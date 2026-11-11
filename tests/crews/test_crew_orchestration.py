"""End-to-end CrewAI orchestration tests (offline mock LLM + optional live Gemini)."""

import pytest

from app.core.config import Settings
from app.core.llm import resolve_gemini_api_key
from app.crews import SmartLogisticsCrew
from app.optimization import load_dataset_from_csv
from app.schemas.logistics import OrchestrationRequest, TrafficDisruptionEvent


def _build_sample_request(
    *,
    scenario: str = "auto",
    unavailable_vehicle_ids: list[str] | None = None,
    simulated_disruption: TrafficDisruptionEvent | None = None,
) -> OrchestrationRequest:
    depots, vehicles, customers = load_dataset_from_csv("data")
    return OrchestrationRequest(
        depots=depots,
        vehicles=vehicles,
        orders=[c.model_dump() for c in customers],
        scenario=scenario,  # type: ignore[arg-type]
        unavailable_vehicle_ids=unavailable_vehicle_ids or [],
        simulated_disruption=simulated_disruption,
        time_limit_seconds=5,
    )


def test_crew_end_to_end_vrptw_on_synthetic_dataset() -> None:
    """Execute full 6-agent CrewAI workflow with OR-Tools VRPTW on the 25-customer dataset."""
    crew_runner = SmartLogisticsCrew(llm_mode="mock")
    response = crew_runner.run(_build_sample_request(scenario="auto"))

    assert response.execution_mode == "mock"
    assert response.order_validation.valid_count == 25
    assert response.order_validation.rejected_count == 0
    assert response.fleet_capacity.capacity_sufficient is True
    assert response.optimization_preparation.recommended_scenario == "VRPTW"
    assert response.route_optimization.status == "feasible"
    assert response.route_optimization.optimization_result is not None
    assert response.traffic_exception.is_simulated is True
    assert response.traffic_exception.has_disruption is False

    plan = response.dispatch_plan
    assert plan.dispatch_status == "approved"
    assert plan.route_source == "initial_ortools"
    assert plan.ortools_provenance_verified is True
    assert len(plan.all_serviced_customer_ids) == 25
    assert len(set(plan.all_serviced_customer_ids)) == 25
    assert plan.dispatched_routes == response.route_optimization.optimization_result.routes

    perf = response.performance_report
    assert perf.report_status == "completed"
    assert perf.metrics is not None
    assert perf.metrics.total_deliveries_completed == 25
    assert perf.metrics.time_window_violations == 0
    assert perf.fuel_estimate is not None
    assert perf.fuel_estimate.estimated_fuel_liters > 0


def test_crew_end_to_end_cvrp_scenario() -> None:
    """Explicit CVRP scenario invokes CvrpOptimizationTool and avoids faking TW violations."""
    crew_runner = SmartLogisticsCrew(llm_mode="mock")
    response = crew_runner.run(_build_sample_request(scenario="CVRP"))

    assert response.optimization_preparation.recommended_scenario == "CVRP"
    assert response.route_optimization.problem_type == "CVRP"
    assert response.dispatch_plan.dispatch_status == "approved"
    assert response.dispatch_plan.problem_type == "CVRP"
    assert len(response.dispatch_plan.all_serviced_customer_ids) == 25
    assert response.performance_report.metrics is not None
    assert response.performance_report.metrics.time_window_violations is None


def test_crew_simulated_disruption_and_dynamic_replanning_preserve_completed() -> None:
    """Simulated traffic disruption triggers OR-Tools replanning while preserving finished stops."""
    baseline = SmartLogisticsCrew(llm_mode="mock").run(_build_sample_request(scenario="auto"))
    initial_routes = baseline.dispatch_plan.dispatched_routes
    assert len(initial_routes) >= 2

    # Mark first route's earliest stops completed and simulate disabling that vehicle
    disrupted_vehicle = initial_routes[0].vehicle_id
    completed_stops = list(initial_routes[0].ordered_customer_ids[:2])
    disruption = TrafficDisruptionEvent(
        event_id="SIM-CONGESTION-01",
        is_simulated=True,
        severity="high",
        description="Simulated highway lane restriction and vehicle breakdown",
        disabled_vehicle_ids=[disrupted_vehicle],
        affected_vehicle_ids=[initial_routes[1].vehicle_id],
        completed_customer_ids=completed_stops,
        extra_delay_minutes=10,
        speed_reduction_factor=0.85,
        elapsed_time_minutes=25,
    )

    response = SmartLogisticsCrew(llm_mode="mock").run(
        _build_sample_request(scenario="auto", simulated_disruption=disruption)
    )

    assert response.traffic_exception.has_disruption is True
    assert response.traffic_exception.is_simulated is True
    assert response.traffic_exception.replanning_recommended is True
    replan = response.traffic_exception.replanning_report
    assert replan is not None
    assert replan.status == "replanned"
    assert replan.disabled_vehicle_ids == [disrupted_vehicle]
    assert {c.customer_id for c in replan.preserved_completed_deliveries} == set(
        completed_stops
    )

    plan = response.dispatch_plan
    assert plan.dispatch_status == "approved"
    assert plan.route_source == "replanned_ortools"
    assert plan.is_simulated_traffic_used is True
    assert all(r.vehicle_id != disrupted_vehicle for r in plan.dispatched_routes)
    assert len(set(plan.all_serviced_customer_ids)) == 25

    perf = response.performance_report
    assert perf.is_replanned_workflow is True
    assert perf.completed_before_replanning_count == len(completed_stops)
    assert perf.completed_demand_before_replanning > 0
    assert perf.remaining_deliveries_count == 25 - len(completed_stops)
    assert perf.total_deliveries_accounted_for == 25
    assert perf.completion_rate == 1.0

    fc = perf.fleet_comparison
    assert fc is not None
    assert fc.disabled_vehicle_ids == [disrupted_vehicle]
    assert disrupted_vehicle not in fc.active_remaining_vehicle_ids
    assert fc.active_remaining_fleet_capacity < fc.original_fleet_capacity
    assert perf.metrics is not None
    assert perf.metrics.total_vehicles_available == fc.active_remaining_vehicle_count
    assert disrupted_vehicle not in perf.metrics.vehicle_utilization_by_id
    assert perf.metrics.fleet_capacity_utilization == fc.active_remaining_fleet_utilization

    db = perf.distance_breakdown
    assert db is not None
    assert db.distance_travelled_before_disruption_meters is None
    assert db.estimated_total_shift_distance_meters is None
    assert db.total_shift_distance_status == "unavailable_missing_pre_disruption_distance"

    assert perf.fuel_estimate is not None
    assert perf.fuel_estimate.estimate_scope == "remaining_routes"
    assert perf.total_shift_fuel_estimate is None


def test_crew_targeted_delay_and_observed_telemetry_kpis() -> None:
    """Targeted vehicle delay spares unaffected vehicles and enables complete-shift fuel KPIs when observed."""
    req = _build_sample_request(scenario="auto")
    baseline = SmartLogisticsCrew(llm_mode="mock").run(req)
    initial_routes = baseline.dispatch_plan.dispatched_routes
    assert len(initial_routes) >= 2

    initial_by_vid = {r.vehicle_id: r for r in initial_routes}
    unaffected_vid = initial_routes[0].vehicle_id
    delayed_vid = initial_routes[1].vehicle_id
    completed_stops = list(initial_routes[0].ordered_customer_ids[:1])

    delayed_depot_id = next(
        v.start_depot_id for v in req.vehicles if v.vehicle_id == delayed_vid
    )
    depot_start = next(
        d.time_window_start for d in req.depots if d.depot_id == delayed_depot_id
    )
    elapsed = depot_start + 20
    extra_delay = 15

    disruption = TrafficDisruptionEvent(
        event_id="SIM-TARGETED-CREW-01",
        is_simulated=True,
        severity="high",
        description="Targeted lane delay on one active vehicle with odometer telemetry",
        affected_vehicle_ids=[delayed_vid],
        completed_customer_ids=completed_stops,
        extra_delay_minutes=extra_delay,
        speed_reduction_factor=0.85,
        elapsed_time_minutes=elapsed,
        actual_distance_before_disruption_meters=8500,
    )

    response = SmartLogisticsCrew(llm_mode="mock").run(
        _build_sample_request(scenario="auto", simulated_disruption=disruption)
    )
    replan = response.traffic_exception.replanning_report
    assert replan is not None
    assert replan.status == "replanned"
    assert replan.vehicle_earliest_start_minutes[delayed_vid] == elapsed + extra_delay
    assert (
        replan.vehicle_earliest_start_minutes[unaffected_vid]
        == initial_by_vid[unaffected_vid].depot_departure_time_minutes
    )

    replanned_by_vid = {r.vehicle_id: r for r in response.dispatch_plan.dispatched_routes}
    if delayed_vid in replanned_by_vid:
        dep_min = replanned_by_vid[delayed_vid].depot_departure_time_minutes
        assert dep_min is not None
        assert dep_min >= elapsed + extra_delay

    perf = response.performance_report
    db = perf.distance_breakdown
    assert db is not None
    assert db.distance_travelled_before_disruption_meters == 8500
    assert db.remaining_optimized_route_distance_meters is not None
    assert (
        db.estimated_total_shift_distance_meters
        == 8500 + db.remaining_optimized_route_distance_meters
    )
    assert db.total_shift_distance_status == "available_with_observed_pre_disruption"
    assert perf.fuel_estimate is not None
    assert perf.fuel_estimate.estimate_scope == "remaining_routes"
    assert perf.total_shift_fuel_estimate is not None
    assert perf.total_shift_fuel_estimate.estimate_scope == "complete_shift"
    assert (
        perf.total_shift_fuel_estimate.distance_basis_km
        == db.estimated_total_shift_distance_km
    )


def test_crew_infeasible_replanning_blocks_dispatch_safely() -> None:
    """Infeasible post-disruption constraints return status='infeasible' and block dispatch."""
    req = _build_sample_request(scenario="auto")
    baseline = SmartLogisticsCrew(llm_mode="mock").run(req)
    max_depot_end = max(d.time_window_end for d in req.depots)
    active_vids = [r.vehicle_id for r in baseline.dispatch_plan.dispatched_routes]

    disruption = TrafficDisruptionEvent(
        event_id="SIM-INFEASIBLE-CREW-01",
        is_simulated=True,
        severity="critical",
        description="Extreme congestion past depot close across active fleet",
        affected_vehicle_ids=active_vids,
        completed_customer_ids=[],
        extra_delay_minutes=120,
        speed_reduction_factor=0.5,
        elapsed_time_minutes=max_depot_end + 60,
    )

    response = SmartLogisticsCrew(llm_mode="mock").run(
        _build_sample_request(scenario="auto", simulated_disruption=disruption)
    )
    replan = response.traffic_exception.replanning_report
    assert replan is not None
    assert replan.status == "infeasible"
    assert replan.replanned_result is None
    assert response.dispatch_plan.dispatch_status == "blocked"
    assert response.dispatch_plan.dispatched_routes == []
    assert response.performance_report.report_status == "blocked"


def test_crew_blocks_dispatch_on_insufficient_capacity_invalid_orders_and_infeasible_tw() -> None:
    """Verify safe blocking (no fabricated routes) across all failure modes."""
    depots, vehicles, customers = load_dataset_from_csv("data")
    crew_runner = SmartLogisticsCrew(llm_mode="mock")

    # 1. All orders invalid
    bad_orders_req = OrchestrationRequest(
        depots=depots,
        vehicles=vehicles,
        orders=[{"customer_id": "BAD_1", "latitude": 120.0, "longitude": 77.6, "demand": -3}],
    )
    bad_resp = crew_runner.run(bad_orders_req)
    assert bad_resp.order_validation.is_valid_for_optimization is False
    assert bad_resp.route_optimization.status == "blocked"
    assert bad_resp.dispatch_plan.dispatch_status == "blocked"
    assert bad_resp.dispatch_plan.dispatched_routes == []
    assert bad_resp.performance_report.report_status == "blocked"

    # 2. Insufficient fleet capacity (too many vehicles unavailable)
    short_fleet_resp = crew_runner.run(
        _build_sample_request(unavailable_vehicle_ids=["VEH_1", "VEH_2", "VEH_3"])
    )
    assert short_fleet_resp.fleet_capacity.capacity_sufficient is False
    assert short_fleet_resp.route_optimization.status == "blocked"
    assert short_fleet_resp.dispatch_plan.dispatch_status == "blocked"
    assert short_fleet_resp.dispatch_plan.dispatched_routes == []

    # 3. Infeasible VRPTW time windows
    impossible_order = customers[0].model_copy(
        update={"time_window_start": 0, "time_window_end": 1}
    )
    infeasible_req = OrchestrationRequest(
        depots=depots,
        vehicles=vehicles,
        orders=[impossible_order.model_dump()],
    )
    infeasible_resp = crew_runner.run(infeasible_req)
    assert infeasible_resp.route_optimization.status == "infeasible"
    assert infeasible_resp.dispatch_plan.dispatch_status == "blocked"
    assert infeasible_resp.dispatch_plan.dispatched_routes == []


def test_crew_guardrail_blocks_fabricated_route_during_kickoff() -> None:
    """CrewAI runtime guardrail refuses an LLM attempt to inject fabricated routes."""
    import json
    from app.core.llm import MockLogisticsLLM

    class FabricatingMockLLM(MockLogisticsLLM):
        def _handle_dispatch_coordination(self, full_prompt: str) -> str:
            base = super()._handle_dispatch_coordination(full_prompt)
            payload_str = base.split("Final Answer:", 1)[1].strip()
            payload = json.loads(payload_str)
            # Tamper with OR-Tools route output (or invent one on guardrail retries)
            if payload["dispatched_routes"]:
                first_route = payload["dispatched_routes"][0]
                if len(first_route["ordered_customer_ids"]) >= 2:
                    first_route["ordered_customer_ids"].reverse()
                else:
                    first_route["ordered_customer_ids"].append("CUST_FABRICATED")
                first_route["total_distance_meters"] += 9999
            else:
                payload["dispatch_status"] = "approved"
                payload["route_source"] = "initial_ortools"
                payload["dispatched_routes"] = [
                    {
                        "vehicle_id": "VEH_FAKE",
                        "vehicle_capacity": 100,
                        "starting_depot_id": "DEPOT_1",
                        "ending_depot_id": "DEPOT_1",
                        "ordered_customer_ids": ["CUST_01"],
                        "stops": [],
                        "total_distance_meters": 1000,
                        "total_distance_km": 1.0,
                        "estimated_travel_time_minutes": 2,
                        "total_route_duration_minutes": 10,
                        "arrival_times_minutes": {},
                        "total_demand": 10,
                        "capacity_utilization": 0.1,
                    }
                ]
            return (
                "Thought: Returning tampered dispatch routes.\n"
                f"Final Answer: {json.dumps(payload)}"
            )

    depots, vehicles, customers = load_dataset_from_csv("data")
    subset_orders = [c.model_dump() for c in customers[:4]]
    req = OrchestrationRequest(
        depots=depots,
        vehicles=vehicles,
        orders=subset_orders,
        scenario="VRPTW",
    )
    bad_llm = FabricatingMockLLM(
        context_payload={
            "depots": [d.model_dump() for d in depots],
            "vehicles": [v.model_dump() for v in vehicles],
            "orders": subset_orders,
            "unavailable_vehicle_ids": [],
            "scenario": "VRPTW",
            "time_limit_seconds": 5,
            "default_horizon_minutes": 1440,
            "use_guided_local_search": False,
            "simulated_disruption": None,
            "fuel_efficiency_km_per_liter": 8.0,
            "fuel_price_per_liter": 1.5,
        }
    )
    crew_runner = SmartLogisticsCrew(llm=bad_llm)
    with pytest.raises(Exception, match="guardrail"):
        crew_runner.run(req)


@pytest.mark.skipif(
    resolve_gemini_api_key(Settings()) == "",
    reason="Live Gemini API key (GEMINI_API_KEY / GOOGLE_API_KEY) not configured.",
)
def test_optional_live_gemini_crew_smoke_run() -> None:
    """Optional live Google Gemini test, skipped unless GEMINI_API_KEY/GOOGLE_API_KEY is set."""
    depots, vehicles, customers = load_dataset_from_csv("data")
    small_request = OrchestrationRequest(
        depots=depots,
        vehicles=vehicles[:2],
        orders=[c.model_dump() for c in customers[:3]],
        scenario="VRPTW",
    )
    crew_runner = SmartLogisticsCrew(llm_mode="gemini")
    response = crew_runner.run(small_request)
    assert response.execution_mode == "gemini"
    assert response.dispatch_plan.dispatch_status == "approved"
    assert response.dispatch_plan.ortools_provenance_verified is True
