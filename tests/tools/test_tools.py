"""Unit tests for all seven CrewAI Phase III custom logistics tools."""

import json

from app.optimization import load_dataset_from_csv
from app.schemas.logistics import (
    DynamicReplanningReport,
    FleetCapacityReport,
    LogisticsPerformanceReport,
    OrderValidationReport,
    RouteOptimizationReport,
    TrafficExceptionReport,
)
from app.tools import (
    CvrpOptimizationTool,
    DynamicRouteReplanningTool,
    FleetCapacityAnalysisTool,
    LogisticsPerformanceAnalysisTool,
    OrderValidationTool,
    TrafficDisruptionSimulationTool,
    VrptwOptimizationTool,
    verify_dispatch_routes_match_ortools,
)


def _load_sample_dicts() -> tuple[list[dict], list[dict], list[dict]]:
    depots, vehicles, customers = load_dataset_from_csv("data")
    return (
        [d.model_dump() for d in depots],
        [v.model_dump() for v in vehicles],
        [c.model_dump() for c in customers],
    )


def test_order_validation_tool_partitions_valid_and_rejected_orders() -> None:
    """OrderValidationTool rejects bad records safely without halting valid orders."""
    tool = OrderValidationTool()
    orders = [
        {
            "customer_id": "CUST_OK",
            "latitude": 12.95,
            "longitude": 77.60,
            "demand": 15,
            "service_duration_minutes": 10,
            "time_window_start": 30,
            "time_window_end": 120,
        },
        {
            "customer_id": "CUST_OK",  # duplicate ID
            "latitude": 12.96,
            "longitude": 77.61,
            "demand": 10,
        },
        {
            "customer_id": "CUST_NEG_DEMAND",
            "latitude": 12.96,
            "longitude": 77.61,
            "demand": -5,
        },
        {
            "customer_id": "CUST_BAD_COORD",
            "latitude": 99.0,
            "longitude": 77.61,
            "demand": 5,
        },
        {
            "customer_id": "CUST_BAD_TW",
            "latitude": 12.96,
            "longitude": 77.61,
            "demand": 5,
            "time_window_start": 200,
            "time_window_end": 100,
        },
    ]

    report = OrderValidationReport.model_validate_json(tool.run(orders=orders))
    assert report.total_submitted == 5
    assert report.valid_count == 1
    assert report.rejected_count == 4
    assert report.valid_orders[0].customer_id == "CUST_OK"
    assert report.total_valid_demand == 15
    assert report.has_time_windows is True
    assert report.is_valid_for_optimization is True

    empty_report = OrderValidationReport.model_validate_json(tool.run(orders=[]))
    assert empty_report.is_valid_for_optimization is False
    assert empty_report.valid_count == 0


def test_fleet_capacity_tool_checks_availability_and_insufficiency() -> None:
    """FleetCapacityAnalysisTool evaluates depot mappings, offline vehicles, and capacity."""
    depots, vehicles, orders = _load_sample_dicts()
    tool = FleetCapacityAnalysisTool()

    ok_report = FleetCapacityReport.model_validate_json(
        tool.run(depots=depots, vehicles=vehicles, valid_orders=orders)
    )
    assert ok_report.capacity_sufficient is True
    assert ok_report.total_fleet_capacity == 520
    assert ok_report.total_required_demand == 389
    assert ok_report.capacity_slack == 131
    assert ok_report.depot_vehicle_counts == {"DEPOT_1": 3, "DEPOT_2": 2}

    shortage_report = FleetCapacityReport.model_validate_json(
        tool.run(
            depots=depots,
            vehicles=vehicles,
            valid_orders=orders,
            unavailable_vehicle_ids=["VEH_1", "VEH_4"],
        )
    )
    assert shortage_report.capacity_sufficient is False
    assert shortage_report.total_fleet_capacity == 290
    assert shortage_report.capacity_slack == -99
    assert len(shortage_report.unavailable_vehicles) == 2


def test_cvrp_and_vrptw_tools_return_ortools_routes_and_handle_infeasible() -> None:
    """Optimization tools invoke real OR-Tools and block infeasible problems without fake routes."""
    depots, vehicles, orders = _load_sample_dicts()
    cvrp_tool = CvrpOptimizationTool()
    vrptw_tool = VrptwOptimizationTool()

    cvrp_report = RouteOptimizationReport.model_validate_json(
        cvrp_tool.run(depots=depots, vehicles=vehicles, customers=orders)
    )
    assert cvrp_report.status == "feasible"
    assert cvrp_report.optimization_result is not None
    assert cvrp_report.optimization_result.problem_type == "CVRP"
    assert cvrp_report.optimization_result.metrics.total_deliveries_completed == 25
    assert cvrp_report.optimization_result.metrics.time_window_violations is None

    vrptw_report = RouteOptimizationReport.model_validate_json(
        vrptw_tool.run(depots=depots, vehicles=vehicles, customers=orders)
    )
    assert vrptw_report.status == "feasible"
    assert vrptw_report.optimization_result is not None
    assert vrptw_report.optimization_result.problem_type == "VRPTW"
    assert vrptw_report.optimization_result.metrics.total_deliveries_completed == 25
    assert vrptw_report.optimization_result.metrics.time_window_violations == 0

    tight_orders = [dict(orders[0], time_window_start=1, time_window_end=2)]
    infeasible_report = RouteOptimizationReport.model_validate_json(
        vrptw_tool.run(depots=depots, vehicles=vehicles, customers=tight_orders)
    )
    assert infeasible_report.status == "infeasible"
    assert infeasible_report.optimization_result is None
    assert infeasible_report.error_message


def test_traffic_simulation_and_dynamic_replanning_preserve_completed_deliveries() -> None:
    """Traffic simulation flags simulated events and dynamic replanning preserves finished stops."""
    depots, vehicles, orders = _load_sample_dicts()
    vrptw_tool = VrptwOptimizationTool()
    initial_report = RouteOptimizationReport.model_validate_json(
        vrptw_tool.run(depots=depots, vehicles=vehicles, customers=orders)
    )
    assert initial_report.optimization_result is not None

    first_route = initial_report.optimization_result.routes[0]
    second_route = initial_report.optimization_result.routes[1]
    completed_ids = (
        first_route.ordered_customer_ids[:2] + second_route.ordered_customer_ids[:1]
    )
    disabled_vehicle = first_route.vehicle_id

    disruption_event = {
        "event_id": "SIM-INCIDENT-42",
        "is_simulated": True,
        "severity": "high",
        "description": "Simulated arterial lane closure disabling one truck mid-shift",
        "disabled_vehicle_ids": [disabled_vehicle],
        "affected_vehicle_ids": [second_route.vehicle_id],
        "completed_customer_ids": completed_ids,
        "extra_delay_minutes": 15,
        "speed_reduction_factor": 0.8,
        "elapsed_time_minutes": 30,
    }

    sim_tool = TrafficDisruptionSimulationTool()
    sim_report = TrafficExceptionReport.model_validate_json(
        sim_tool.run(
            optimization_result=initial_report.optimization_result.model_dump(),
            orders=orders,
            disruption_event=disruption_event,
        )
    )
    assert sim_report.has_disruption is True
    assert sim_report.is_simulated is True
    assert "SIMULATED" in sim_report.simulation_notice
    assert sim_report.replanning_recommended is True

    replan_tool = DynamicRouteReplanningTool()
    replan_report = DynamicReplanningReport.model_validate_json(
        replan_tool.run(
            depots=depots,
            vehicles=vehicles,
            orders=orders,
            initial_optimization_result=initial_report.optimization_result.model_dump(),
            disruption_event=disruption_event,
        )
    )
    assert replan_report.status == "replanned"
    assert {c.customer_id for c in replan_report.preserved_completed_deliveries} == set(
        completed_ids
    )
    assert replan_report.replanned_result is not None
    replanned_ids = [
        cid
        for r in replan_report.replanned_result.routes
        for cid in r.ordered_customer_ids
    ]
    assert set(replanned_ids).isdisjoint(set(completed_ids))
    assert len(replanned_ids) + len(completed_ids) == 25
    assert all(
        r.vehicle_id != disabled_vehicle for r in replan_report.replanned_result.routes
    )


def test_performance_analysis_tool_and_route_provenance_guardrail() -> None:
    """Performance tool computes KPIs + fuel estimates and provenance rejects fabricated routes."""
    depots, vehicles, orders = _load_sample_dicts()
    vrptw_tool = VrptwOptimizationTool()
    vrptw_report = RouteOptimizationReport.model_validate_json(
        vrptw_tool.run(depots=depots, vehicles=vehicles, customers=orders)
    )
    opt_result = vrptw_report.optimization_result
    assert opt_result is not None

    perf_tool = LogisticsPerformanceAnalysisTool()
    perf_report = LogisticsPerformanceReport.model_validate_json(
        perf_tool.run(
            optimization_result=opt_result.model_dump(),
            completed_deliveries=[],
            total_submitted_orders=25,
            dispatch_status="approved",
            problem_type="VRPTW",
            vehicles=vehicles,
            fuel_efficiency_km_per_liter=10.0,
            fuel_price_per_liter=2.0,
        )
    )
    assert perf_report.report_status == "completed"
    assert perf_report.metrics is not None
    assert perf_report.completion_rate == 1.0
    assert perf_report.fuel_estimate is not None
    expected_liters = round(perf_report.metrics.total_distance_km / 10.0, 4)
    assert perf_report.fuel_estimate.estimated_fuel_liters == expected_liters

    ok, errors = verify_dispatch_routes_match_ortools(
        opt_result.routes, opt_result, expected_problem_type="VRPTW"
    )
    assert ok is True
    assert errors == []

    tampered = json.loads(opt_result.model_dump_json())
    tampered["routes"][0]["ordered_customer_ids"].reverse()
    from app.schemas.route import OptimizationResult

    tampered_obj = OptimizationResult.model_validate(tampered)
    ok_bad, bad_errors = verify_dispatch_routes_match_ortools(
        tampered_obj.routes, opt_result, expected_problem_type="VRPTW"
    )
    assert ok_bad is False
    assert any("match" in err.lower() for err in bad_errors)


def test_dynamic_replanning_targeted_delay_preserves_unaffected_and_handles_infeasible() -> None:
    """Targeted delay shifts only affected vehicle departure and reports explicit infeasibility."""
    from app.schemas.logistics import (
        DynamicReplanningReport,
        TrafficDisruptionEvent,
    )
    from app.tools.traffic_tools import DynamicRouteReplanningTool

    depots = [
        {
            "depot_id": "DEPOT_1",
            "latitude": 13.0,
            "longitude": 77.5,
            "time_window_start": 0,
            "time_window_end": 480,
        }
    ]
    vehicles = [
        {
            "vehicle_id": "V1",
            "capacity": 30,
            "start_depot_id": "DEPOT_1",
            "max_route_duration_minutes": 480,
            "speed_kmh": 40.0,
        },
        {
            "vehicle_id": "V2",
            "capacity": 30,
            "start_depot_id": "DEPOT_1",
            "max_route_duration_minutes": 480,
            "speed_kmh": 40.0,
        },
    ]
    orders = [
        {
            "customer_id": "C_DONE",
            "latitude": 13.005,
            "longitude": 77.5,
            "demand": 5,
            "service_duration_minutes": 5,
            "time_window_start": 0,
            "time_window_end": 20,
        },
        {
            "customer_id": "C_EARLY",
            "latitude": 13.015,
            "longitude": 77.5,
            "demand": 25,
            "service_duration_minutes": 5,
            "time_window_start": 0,
            "time_window_end": 60,
        },
        {
            "customer_id": "C_LATE",
            "latitude": 13.0,
            "longitude": 77.515,
            "demand": 25,
            "service_duration_minutes": 5,
            "time_window_start": 100,
            "time_window_end": 300,
        },
    ]

    vrptw_tool = VrptwOptimizationTool()
    initial_report = RouteOptimizationReport.model_validate_json(
        vrptw_tool.run(depots=depots, vehicles=vehicles, customers=orders)
    )
    initial_opt = initial_report.optimization_result
    assert initial_opt is not None

    initial_by_vid = {r.vehicle_id: r for r in initial_opt.routes}
    late_vid = next(
        r.vehicle_id for r in initial_opt.routes if "C_LATE" in r.ordered_customer_ids
    )
    early_vid = next(
        r.vehicle_id for r in initial_opt.routes if "C_EARLY" in r.ordered_customer_ids
    )
    assert late_vid != early_vid

    replan_tool = DynamicRouteReplanningTool()
    targeted_event = TrafficDisruptionEvent(
        event_id="SIM-TARGETED-01",
        is_simulated=True,
        severity="high",
        description="Targeted lane closure affecting late route vehicle only",
        affected_vehicle_ids=[late_vid],
        completed_customer_ids=["C_DONE"],
        extra_delay_minutes=30,
        speed_reduction_factor=0.8,
        elapsed_time_minutes=35,
    )
    replan_report = DynamicReplanningReport.model_validate_json(
        replan_tool.run(
            depots=depots,
            vehicles=vehicles,
            orders=orders,
            initial_optimization_result=initial_opt.model_dump(),
            disruption_event=targeted_event.model_dump(),
        )
    )

    assert replan_report.status == "replanned"
    assert replan_report.replanned_result is not None
    assert {c.customer_id for c in replan_report.preserved_completed_deliveries} == {
        "C_DONE"
    }
    assert replan_report.vehicle_earliest_start_minutes[late_vid] == 65
    assert (
        replan_report.vehicle_earliest_start_minutes[early_vid]
        == initial_by_vid[early_vid].depot_departure_time_minutes
    )

    replanned_by_vid = {
        r.vehicle_id: r for r in replan_report.replanned_result.routes
    }
    assert replanned_by_vid[early_vid].ordered_customer_ids == ["C_EARLY"]
    assert (
        replanned_by_vid[early_vid].depot_departure_time_minutes
        == initial_by_vid[early_vid].depot_departure_time_minutes
    )
    assert replanned_by_vid[late_vid].ordered_customer_ids == ["C_LATE"]
    assert replanned_by_vid[late_vid].depot_departure_time_minutes is not None
    assert replanned_by_vid[late_vid].depot_departure_time_minutes >= 65

    infeasible_event = TrafficDisruptionEvent(
        event_id="SIM-INFEASIBLE-01",
        is_simulated=True,
        severity="critical",
        description="Extreme delay exceeding depot closing time",
        affected_vehicle_ids=[early_vid, late_vid],
        completed_customer_ids=["C_DONE"],
        extra_delay_minutes=100,
        speed_reduction_factor=0.5,
        elapsed_time_minutes=400,
    )
    infeasible_report = DynamicReplanningReport.model_validate_json(
        replan_tool.run(
            depots=depots,
            vehicles=vehicles,
            orders=orders,
            initial_optimization_result=initial_opt.model_dump(),
            disruption_event=infeasible_event.model_dump(),
        )
    )
    assert infeasible_report.status == "infeasible"
    assert infeasible_report.replanned_result is None
    assert infeasible_report.error_type is not None
    assert infeasible_report.error_message is not None
    assert {c.customer_id for c in infeasible_report.preserved_completed_deliveries} == {
        "C_DONE"
    }


def test_performance_analysis_excludes_disabled_vehicles_and_scopes_fuel_and_distance() -> None:
    """Performance report separates original vs active remaining fleet and avoids fabricating telemetry."""
    from app.schemas.logistics import CompletedDeliveryRecord

    depots = [
        {
            "depot_id": "DEPOT_1",
            "latitude": 13.0,
            "longitude": 77.5,
            "time_window_start": 0,
            "time_window_end": 480,
        }
    ]
    all_vehicles = [
        {"vehicle_id": "V1", "capacity": 50, "start_depot_id": "DEPOT_1"},
        {"vehicle_id": "V2", "capacity": 50, "start_depot_id": "DEPOT_1"},
        {"vehicle_id": "V3", "capacity": 50, "start_depot_id": "DEPOT_1"},
    ]
    active_vehicles = [v for v in all_vehicles if v["vehicle_id"] != "V3"]
    all_orders = [
        {
            "customer_id": "C1",
            "latitude": 13.01,
            "longitude": 77.5,
            "demand": 20,
            "service_duration_minutes": 5,
            "time_window_start": 0,
            "time_window_end": 240,
        },
        {
            "customer_id": "C2",
            "latitude": 13.02,
            "longitude": 77.51,
            "demand": 25,
            "service_duration_minutes": 5,
            "time_window_start": 0,
            "time_window_end": 240,
        },
        {
            "customer_id": "C3",
            "latitude": 13.03,
            "longitude": 77.52,
            "demand": 30,
            "service_duration_minutes": 5,
            "time_window_start": 0,
            "time_window_end": 240,
        },
    ]
    remaining_orders = [o for o in all_orders if o["customer_id"] != "C1"]

    vrptw_tool = VrptwOptimizationTool()
    initial_opt = RouteOptimizationReport.model_validate_json(
        vrptw_tool.run(depots=depots, vehicles=all_vehicles, customers=all_orders)
    ).optimization_result
    replanned_opt = RouteOptimizationReport.model_validate_json(
        vrptw_tool.run(depots=depots, vehicles=active_vehicles, customers=remaining_orders)
    ).optimization_result
    assert initial_opt is not None
    assert replanned_opt is not None

    completed = [
        CompletedDeliveryRecord(customer_id="C1", vehicle_id="V3", demand=20)
    ]
    perf_tool = LogisticsPerformanceAnalysisTool()

    no_telemetry_report = LogisticsPerformanceReport.model_validate_json(
        perf_tool.run(
            optimization_result=replanned_opt.model_dump(),
            original_optimization_result=initial_opt.model_dump(),
            completed_deliveries=[c.model_dump() for c in completed],
            total_submitted_orders=3,
            dispatch_status="approved",
            problem_type="VRPTW",
            vehicles=all_vehicles,
            original_vehicles=all_vehicles,
            active_remaining_vehicles=active_vehicles,
            disabled_vehicle_ids=["V3"],
            is_replanned=True,
            fuel_efficiency_km_per_liter=10.0,
            fuel_price_per_liter=2.0,
        )
    )
    assert no_telemetry_report.is_replanned_workflow is True
    assert no_telemetry_report.metrics is not None
    assert no_telemetry_report.metrics.total_vehicles_available == 2
    assert "V3" not in no_telemetry_report.metrics.vehicle_utilization_by_id
    assert no_telemetry_report.metrics.fleet_capacity_utilization == round(55 / 100, 4)

    fc = no_telemetry_report.fleet_comparison
    assert fc is not None
    assert fc.original_vehicle_count == 3
    assert fc.original_fleet_capacity == 150
    assert fc.disabled_vehicle_ids == ["V3"]
    assert fc.active_remaining_vehicle_ids == ["V1", "V2"]
    assert fc.active_remaining_vehicle_count == 2
    assert fc.active_remaining_fleet_capacity == 100
    assert fc.active_remaining_demand_delivered == 55
    assert fc.active_remaining_fleet_utilization == round(55 / 100, 4)

    assert no_telemetry_report.completed_before_replanning_count == 1
    assert no_telemetry_report.completed_demand_before_replanning == 20
    assert no_telemetry_report.remaining_deliveries_count == 2
    assert no_telemetry_report.remaining_demand_delivered == 55
    assert no_telemetry_report.total_deliveries_accounted_for == 3
    assert no_telemetry_report.completion_rate == 1.0

    db = no_telemetry_report.distance_breakdown
    assert db is not None
    assert (
        db.original_total_route_distance_meters
        == initial_opt.metrics.total_distance_meters
    )
    assert (
        db.remaining_optimized_route_distance_meters
        == replanned_opt.metrics.total_distance_meters
    )
    assert db.distance_travelled_before_disruption_meters is None
    assert db.estimated_total_shift_distance_meters is None
    assert (
        db.total_shift_distance_status
        == "unavailable_missing_pre_disruption_distance"
    )
    assert no_telemetry_report.fuel_estimate is not None
    assert no_telemetry_report.fuel_estimate.estimate_scope == "remaining_routes"
    assert no_telemetry_report.total_shift_fuel_estimate is None

    with_telemetry_report = LogisticsPerformanceReport.model_validate_json(
        perf_tool.run(
            optimization_result=replanned_opt.model_dump(),
            original_optimization_result=initial_opt.model_dump(),
            completed_deliveries=[c.model_dump() for c in completed],
            total_submitted_orders=3,
            dispatch_status="approved",
            problem_type="VRPTW",
            vehicles=all_vehicles,
            original_vehicles=all_vehicles,
            active_remaining_vehicles=active_vehicles,
            disabled_vehicle_ids=["V3"],
            is_replanned=True,
            distance_travelled_before_disruption_meters=12000,
            fuel_efficiency_km_per_liter=10.0,
            fuel_price_per_liter=2.0,
        )
    )
    db_obs = with_telemetry_report.distance_breakdown
    assert db_obs is not None
    assert db_obs.distance_travelled_before_disruption_meters == 12000
    assert (
        db_obs.estimated_total_shift_distance_meters
        == 12000 + replanned_opt.metrics.total_distance_meters
    )
    assert db_obs.total_shift_distance_status == "available_with_observed_pre_disruption"
    assert with_telemetry_report.fuel_estimate is not None
    assert with_telemetry_report.fuel_estimate.estimate_scope == "remaining_routes"
    assert with_telemetry_report.total_shift_fuel_estimate is not None
    assert (
        with_telemetry_report.total_shift_fuel_estimate.estimate_scope
        == "complete_shift"
    )
    assert (
        with_telemetry_report.total_shift_fuel_estimate.distance_basis_km
        == db_obs.estimated_total_shift_distance_km
    )

