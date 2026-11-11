"""End-to-end Phase III demonstration script using the synthetic logistics CSV dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.schemas.logistics import OrchestrationResponse, TrafficDisruptionEvent
from app.services import LogisticsOrchestrationService


def _print_section(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2))


def main() -> None:
    """Run nominal VRPTW and simulated traffic replanning workflows end-to-end."""
    service = LogisticsOrchestrationService(llm_mode="mock")

    # 1. Baseline multi-depot 5-vehicle 25-customer VRPTW workflow
    nominal: OrchestrationResponse = service.run_from_csv("data", scenario="auto")
    _print_section(
        "1. Order Validation & Fleet Capacity (Baseline VRPTW)",
        {
            "order_validation_summary": nominal.order_validation.summary,
            "fleet_capacity_summary": nominal.fleet_capacity.summary,
            "recommended_scenario": nominal.optimization_preparation.recommended_scenario,
            "route_optimization_summary": nominal.route_optimization.summary,
        },
    )
    _print_section(
        "2. Final Dispatch Plan & KPIs (Baseline VRPTW)",
        {
            "dispatch_status": nominal.dispatch_plan.dispatch_status,
            "route_source": nominal.dispatch_plan.route_source,
            "ortools_provenance_verified": nominal.dispatch_plan.ortools_provenance_verified,
            "routes": [
                {
                    "vehicle_id": route.vehicle_id,
                    "depot": f"{route.starting_depot_id}->{route.ending_depot_id}",
                    "departure_min": route.depot_departure_time_minutes,
                    "customers": route.ordered_customer_ids,
                    "demand": f"{route.total_demand}/{route.vehicle_capacity}",
                    "distance_km": route.total_distance_km,
                    "travel_time_min": route.estimated_travel_time_minutes,
                }
                for route in nominal.dispatch_plan.dispatched_routes
            ],
            "fleet_comparison": (
                nominal.performance_report.fleet_comparison.model_dump()
                if nominal.performance_report.fleet_comparison
                else None
            ),
            "distance_breakdown": (
                nominal.performance_report.distance_breakdown.model_dump()
                if nominal.performance_report.distance_breakdown
                else None
            ),
            "performance_metrics": (
                nominal.performance_report.metrics.model_dump()
                if nominal.performance_report.metrics
                else None
            ),
            "fuel_estimate": (
                nominal.performance_report.fuel_estimate.model_dump()
                if nominal.performance_report.fuel_estimate
                else None
            ),
        },
    )

    # 2. Controlled simulated traffic disruption + dynamic replanning preserving completed deliveries
    first_route = nominal.dispatch_plan.dispatched_routes[0]
    second_route = nominal.dispatch_plan.dispatched_routes[1]
    disruption = TrafficDisruptionEvent(
        event_id="SIM-DEMO-TRAFFIC-01",
        is_simulated=True,
        severity="high",
        description="Simulated highway lane closure disabling one active truck mid-shift",
        disabled_vehicle_ids=[first_route.vehicle_id],
        affected_vehicle_ids=[second_route.vehicle_id],
        completed_customer_ids=list(first_route.ordered_customer_ids[:2]),
        extra_delay_minutes=10,
        speed_reduction_factor=0.85,
        elapsed_time_minutes=25,
    )
    replanned: OrchestrationResponse = service.run_from_csv(
        "data",
        scenario="auto",
        simulated_disruption=disruption,
    )
    _print_section(
        "3. Simulated Traffic Exception & Dynamic OR-Tools Replanning",
        {
            "is_simulated": replanned.traffic_exception.is_simulated,
            "simulation_notice": replanned.traffic_exception.simulation_notice,
            "replanning_summary": (
                replanned.traffic_exception.replanning_report.summary
                if replanned.traffic_exception.replanning_report
                else None
            ),
            "preserved_completed_deliveries": [
                c.model_dump()
                for c in replanned.dispatch_plan.preserved_completed_deliveries
            ],
            "dispatch_status": replanned.dispatch_plan.dispatch_status,
            "route_source": replanned.dispatch_plan.route_source,
            "replanned_routes": [
                {
                    "vehicle_id": route.vehicle_id,
                    "departure_min": route.depot_departure_time_minutes,
                    "customers": route.ordered_customer_ids,
                    "demand": f"{route.total_demand}/{route.vehicle_capacity}",
                    "distance_km": route.total_distance_km,
                    "travel_time_min": route.estimated_travel_time_minutes,
                }
                for route in replanned.dispatch_plan.dispatched_routes
            ],
            "fleet_comparison": (
                replanned.performance_report.fleet_comparison.model_dump()
                if replanned.performance_report.fleet_comparison
                else None
            ),
            "distance_breakdown": (
                replanned.performance_report.distance_breakdown.model_dump()
                if replanned.performance_report.distance_breakdown
                else None
            ),
            "fuel_estimate": (
                replanned.performance_report.fuel_estimate.model_dump()
                if replanned.performance_report.fuel_estimate
                else None
            ),
            "total_shift_fuel_estimate": (
                replanned.performance_report.total_shift_fuel_estimate.model_dump()
                if replanned.performance_report.total_shift_fuel_estimate
                else None
            ),
            "performance_summary": replanned.performance_report.summary,
            "completion_rate": replanned.performance_report.completion_rate,
        },
    )


if __name__ == "__main__":
    main()
