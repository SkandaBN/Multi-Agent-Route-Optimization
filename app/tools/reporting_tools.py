"""CrewAI custom tool for logistics performance metrics and fuel cost reporting."""

from __future__ import annotations

from typing import Any, Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.optimization.metrics import calculate_performance_metrics
from app.schemas.logistics import (
    CompletedDeliveryRecord,
    DistanceBreakdownReport,
    FleetUtilizationComparison,
    FuelEstimateAssumptions,
    LogisticsPerformanceReport,
)
from app.schemas.route import OptimizationResult
from app.schemas.vehicle import Vehicle


class LogisticsPerformanceAnalysisInput(BaseModel):
    """Input schema for LogisticsPerformanceAnalysisTool."""

    optimization_result: dict[str, Any] | None = Field(
        default=None,
        description="Final active OR-Tools OptimizationResult (initial or remaining replanned routes).",
    )
    original_optimization_result: dict[str, Any] | None = Field(
        default=None,
        description="Optional initial pre-disruption OR-Tools OptimizationResult for comparison.",
    )
    completed_deliveries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Completed deliveries preserved before dynamic replanning.",
    )
    total_submitted_orders: int = Field(
        default=0,
        ge=0,
        description="Total customer orders originally submitted.",
    )
    dispatch_status: Literal["approved", "blocked"] = Field(
        description="Whether final dispatch was approved or blocked."
    )
    problem_type: Literal["CVRP", "VRPTW"] = "VRPTW"
    vehicles: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Fallback fleet vehicles if original_vehicles/active_remaining_vehicles omitted.",
    )
    original_vehicles: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Original pre-disruption available fleet vehicles.",
    )
    active_remaining_vehicles: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Active remaining operational vehicles (excluding disabled vehicles).",
    )
    disabled_vehicle_ids: list[str] = Field(
        default_factory=list,
        description="Vehicle IDs disabled by disruption and excluded from remaining fleet capacity.",
    )
    is_replanned: bool = Field(
        default=False,
        description="True when final dispatched routes come from dynamic OR-Tools replanning.",
    )
    distance_travelled_before_disruption_meters: int | None = Field(
        default=None,
        ge=0,
        description="Observed distance travelled (meters) prior to disruption, if available.",
    )
    fuel_efficiency_km_per_liter: float = Field(default=8.0, gt=0.0)
    fuel_price_per_liter: float = Field(default=1.5, ge=0.0)


def _build_fuel_estimate(
    *,
    distance_km: float,
    scope: Literal["complete_shift", "remaining_routes"],
    scope_description: str,
    fuel_efficiency_km_per_liter: float,
    fuel_price_per_liter: float,
) -> FuelEstimateAssumptions:
    """Compute distance-based fuel liters and cost with explicit scope labeling."""
    fuel_liters = round(distance_km / fuel_efficiency_km_per_liter, 4)
    fuel_cost = round(fuel_liters * fuel_price_per_liter, 2)
    return FuelEstimateAssumptions(
        estimate_scope=scope,
        scope_description=scope_description,
        distance_basis_km=round(distance_km, 4),
        fuel_efficiency_km_per_liter=fuel_efficiency_km_per_liter,
        fuel_price_per_liter=fuel_price_per_liter,
        estimated_fuel_liters=fuel_liters,
        estimated_fuel_cost=fuel_cost,
    )


def _vehicles_from_routes(opt_result: OptimizationResult) -> list[Vehicle]:
    """Synthesize minimal Vehicle models from route headers when vehicle inputs are absent."""
    return [
        Vehicle(
            vehicle_id=route.vehicle_id,
            capacity=route.vehicle_capacity,
            start_depot_id=route.starting_depot_id,
            end_depot_id=route.ending_depot_id,
        )
        for route in opt_result.routes
    ]


class LogisticsPerformanceAnalysisTool(BaseTool):
    """Computes final logistics KPIs separating original and active remaining fleets."""

    name: str = "logistics_performance_analysis_tool"
    description: str = (
        "Computes original vs. active remaining fleet utilization (excluding disabled vehicles), "
        "completed vs. remaining deliveries, distance breakdown (original, pre-disruption, "
        "remaining, and total shift where known), VRPTW time-window violations (when evaluated), "
        "and explicitly scoped fuel estimates from verified OR-Tools outputs."
    )
    args_schema: type[BaseModel] = LogisticsPerformanceAnalysisInput

    def _run(
        self,
        optimization_result: dict[str, Any] | None = None,
        original_optimization_result: dict[str, Any] | None = None,
        completed_deliveries: list[dict[str, Any]] | None = None,
        total_submitted_orders: int = 0,
        dispatch_status: Literal["approved", "blocked"] = "approved",
        problem_type: Literal["CVRP", "VRPTW"] = "VRPTW",
        vehicles: list[dict[str, Any]] | None = None,
        original_vehicles: list[dict[str, Any]] | None = None,
        active_remaining_vehicles: list[dict[str, Any]] | None = None,
        disabled_vehicle_ids: list[str] | None = None,
        is_replanned: bool = False,
        distance_travelled_before_disruption_meters: int | None = None,
        fuel_efficiency_km_per_liter: float = 8.0,
        fuel_price_per_liter: float = 1.5,
    ) -> str:
        """Calculate structured performance metrics, fleet comparison, and scoped fuel report."""
        preserved = [
            CompletedDeliveryRecord.model_validate(item)
            for item in (completed_deliveries or [])
        ]
        completed_demand = sum(item.demand for item in preserved)
        orig_opt_obj = (
            OptimizationResult.model_validate(original_optimization_result)
            if original_optimization_result is not None
            else None
        )
        disabled_ids = list(dict.fromkeys(disabled_vehicle_ids or []))
        disabled_set = set(disabled_ids)

        fallback_vehicles = [
            Vehicle.model_validate(item) for item in (vehicles or [])
        ]
        original_fleet = [
            Vehicle.model_validate(item) for item in (original_vehicles or [])
        ] or fallback_vehicles

        if dispatch_status == "blocked":
            orig_dist_m = (
                orig_opt_obj.metrics.total_distance_meters
                if orig_opt_obj is not None
                else None
            )
            orig_dist_km = (
                orig_opt_obj.metrics.total_distance_km
                if orig_opt_obj is not None
                else None
            )
            pre_dist_km = (
                round(distance_travelled_before_disruption_meters / 1000.0, 4)
                if distance_travelled_before_disruption_meters is not None
                else None
            )
            report = LogisticsPerformanceReport(
                report_status="blocked",
                dispatch_status="blocked",
                problem_type=problem_type,
                is_replanned_workflow=is_replanned,
                metrics=None,
                completed_before_replanning_count=len(preserved),
                completed_demand_before_replanning=completed_demand,
                remaining_deliveries_count=0,
                remaining_demand_delivered=0,
                preserved_completed_deliveries=preserved,
                total_deliveries_accounted_for=len(preserved),
                completion_rate=(
                    round(min(1.0, len(preserved) / total_submitted_orders), 4)
                    if total_submitted_orders > 0
                    else 0.0
                ),
                distance_breakdown=DistanceBreakdownReport(
                    original_total_route_distance_meters=orig_dist_m,
                    original_total_route_distance_km=orig_dist_km,
                    distance_travelled_before_disruption_meters=distance_travelled_before_disruption_meters,
                    distance_travelled_before_disruption_km=pre_dist_km,
                    remaining_optimized_route_distance_meters=None,
                    remaining_optimized_route_distance_km=None,
                    estimated_total_shift_distance_meters=None,
                    estimated_total_shift_distance_km=None,
                    total_shift_distance_status="unavailable_blocked",
                ),
                fuel_estimate=None,
                total_shift_fuel_estimate=None,
                notes=[
                    "Dispatch was blocked due to validation, capacity, or OR-Tools infeasibility.",
                    "Zero unverified routes were dispatched; active route metrics are unavailable.",
                ],
                summary="Performance report blocked because no feasible OR-Tools routes were dispatched.",
            )
            return report.model_dump_json()

        opt_result = (
            OptimizationResult.model_validate(optimization_result)
            if optimization_result is not None
            else None
        )
        if not original_fleet:
            if orig_opt_obj is not None:
                original_fleet = _vehicles_from_routes(orig_opt_obj)
            elif opt_result is not None:
                original_fleet = _vehicles_from_routes(opt_result)

        explicit_active = [
            v
            for v in (
                Vehicle.model_validate(item)
                for item in (active_remaining_vehicles or [])
            )
            if v.vehicle_id not in disabled_set
        ]
        if explicit_active:
            active_fleet = explicit_active
        else:
            base_for_active = fallback_vehicles or original_fleet
            active_fleet = [
                v for v in base_for_active if v.vehicle_id not in disabled_set
            ]

        orig_fleet_capacity = sum(v.capacity for v in original_fleet)
        orig_vehicles_used: int | None = None
        orig_fleet_util: float | None = None
        if orig_opt_obj is not None and orig_fleet_capacity > 0:
            orig_used_routes = [
                r for r in orig_opt_obj.routes if r.ordered_customer_ids
            ]
            orig_vehicles_used = len(orig_used_routes)
            orig_fleet_util = round(
                sum(r.total_demand for r in orig_used_routes) / orig_fleet_capacity, 4
            )
        elif not is_replanned and opt_result is not None and orig_fleet_capacity > 0:
            used_routes = [r for r in opt_result.routes if r.ordered_customer_ids]
            orig_vehicles_used = len(used_routes)
            orig_fleet_util = round(
                sum(r.total_demand for r in used_routes) / orig_fleet_capacity, 4
            )

        orig_dist_m = (
            orig_opt_obj.metrics.total_distance_meters
            if orig_opt_obj is not None
            else (
                opt_result.metrics.total_distance_meters
                if (opt_result is not None and not is_replanned)
                else None
            )
        )
        orig_dist_km = (
            round(orig_dist_m / 1000.0, 4) if orig_dist_m is not None else None
        )
        pre_dist_m = distance_travelled_before_disruption_meters
        pre_dist_km = (
            round(pre_dist_m / 1000.0, 4) if pre_dist_m is not None else None
        )

        if opt_result is None:
            completion_rate = (
                round(min(1.0, len(preserved) / total_submitted_orders), 4)
                if total_submitted_orders > 0
                else 1.0
            )
            active_cap = sum(v.capacity for v in active_fleet)
            fleet_comparison = FleetUtilizationComparison(
                original_vehicle_count=len(original_fleet),
                original_fleet_capacity=orig_fleet_capacity,
                original_vehicles_used=orig_vehicles_used,
                original_fleet_capacity_utilization=orig_fleet_util,
                disabled_vehicle_ids=disabled_ids,
                active_remaining_vehicle_ids=[v.vehicle_id for v in active_fleet],
                active_remaining_vehicle_count=len(active_fleet),
                active_remaining_fleet_capacity=active_cap,
                active_remaining_vehicles_used=0,
                active_remaining_demand_delivered=0,
                active_remaining_fleet_utilization=0.0,
                active_vehicle_utilization_by_id={
                    v.vehicle_id: 0.0 for v in active_fleet
                },
            )
            shift_dist_m = pre_dist_m if pre_dist_m is not None else None
            shift_dist_km = pre_dist_km if pre_dist_km is not None else None
            dist_status: Literal[
                "available_complete_shift",
                "available_with_observed_pre_disruption",
                "unavailable_missing_pre_disruption_distance",
                "unavailable_blocked",
            ] = (
                "available_with_observed_pre_disruption"
                if shift_dist_m is not None
                else "unavailable_missing_pre_disruption_distance"
            )
            rem_fuel = _build_fuel_estimate(
                distance_km=0.0,
                scope="remaining_routes",
                scope_description=(
                    "Remaining optimized routes only: 0.0 km remaining because all deliveries completed."
                ),
                fuel_efficiency_km_per_liter=fuel_efficiency_km_per_liter,
                fuel_price_per_liter=fuel_price_per_liter,
            )
            shift_fuel = (
                _build_fuel_estimate(
                    distance_km=shift_dist_km,
                    scope="complete_shift",
                    scope_description=(
                        "Complete shift: observed pre-disruption distance (0.0 km remaining routes)."
                    ),
                    fuel_efficiency_km_per_liter=fuel_efficiency_km_per_liter,
                    fuel_price_per_liter=fuel_price_per_liter,
                )
                if shift_dist_km is not None
                else None
            )
            report = LogisticsPerformanceReport(
                report_status="completed",
                dispatch_status="approved",
                problem_type=problem_type,
                is_replanned_workflow=True,
                metrics=None,
                completed_before_replanning_count=len(preserved),
                completed_demand_before_replanning=completed_demand,
                remaining_deliveries_count=0,
                remaining_demand_delivered=0,
                preserved_completed_deliveries=preserved,
                total_deliveries_accounted_for=len(preserved),
                completion_rate=completion_rate,
                fleet_comparison=fleet_comparison,
                distance_breakdown=DistanceBreakdownReport(
                    original_total_route_distance_meters=orig_dist_m,
                    original_total_route_distance_km=orig_dist_km,
                    distance_travelled_before_disruption_meters=pre_dist_m,
                    distance_travelled_before_disruption_km=pre_dist_km,
                    remaining_optimized_route_distance_meters=0,
                    remaining_optimized_route_distance_km=0.0,
                    estimated_total_shift_distance_meters=shift_dist_m,
                    estimated_total_shift_distance_km=shift_dist_km,
                    total_shift_distance_status=dist_status,
                ),
                fuel_estimate=rem_fuel,
                total_shift_fuel_estimate=shift_fuel,
                notes=[
                    "All orders were already completed prior to replanning; zero remaining route distance needed.",
                    (
                        "Total shift distance computed from observed pre-disruption distance."
                        if shift_dist_km is not None
                        else "Pre-disruption distance telemetry unavailable; full-shift distance not invented."
                    ),
                ],
                summary=(
                    f"Performance report completed: {len(preserved)} already-completed deliveries preserved "
                    f"(demand={completed_demand}); 0 remaining routes needed."
                ),
            )
            return report.model_dump_json()

        if not active_fleet:
            active_fleet = _vehicles_from_routes(opt_result)

        metrics = calculate_performance_metrics(
            opt_result.routes,
            active_fleet,
            evaluate_time_windows=(opt_result.problem_type == "VRPTW"),
        )
        total_accounted = metrics.total_deliveries_completed + len(preserved)
        denominator = total_submitted_orders if total_submitted_orders > 0 else total_accounted
        completion_rate = (
            round(min(1.0, total_accounted / denominator), 4)
            if denominator > 0
            else 0.0
        )

        active_cap = sum(v.capacity for v in active_fleet)
        fleet_comparison = FleetUtilizationComparison(
            original_vehicle_count=len(original_fleet),
            original_fleet_capacity=orig_fleet_capacity,
            original_vehicles_used=orig_vehicles_used,
            original_fleet_capacity_utilization=orig_fleet_util,
            disabled_vehicle_ids=disabled_ids,
            active_remaining_vehicle_ids=[v.vehicle_id for v in active_fleet],
            active_remaining_vehicle_count=len(active_fleet),
            active_remaining_fleet_capacity=active_cap,
            active_remaining_vehicles_used=metrics.vehicles_used,
            active_remaining_demand_delivered=metrics.total_demand_delivered,
            active_remaining_fleet_utilization=metrics.fleet_capacity_utilization,
            active_vehicle_utilization_by_id=metrics.vehicle_utilization_by_id,
        )

        rem_dist_m = metrics.total_distance_meters
        rem_dist_km = metrics.total_distance_km
        if is_replanned:
            if pre_dist_m is not None:
                shift_dist_m = pre_dist_m + rem_dist_m
                shift_dist_km = round(shift_dist_m / 1000.0, 4)
                dist_status = "available_with_observed_pre_disruption"
            else:
                shift_dist_m = None
                shift_dist_km = None
                dist_status = "unavailable_missing_pre_disruption_distance"

            fuel_estimate = _build_fuel_estimate(
                distance_km=rem_dist_km,
                scope="remaining_routes",
                scope_description=(
                    "Remaining optimized routes only: excludes distance travelled prior to disruption."
                ),
                fuel_efficiency_km_per_liter=fuel_efficiency_km_per_liter,
                fuel_price_per_liter=fuel_price_per_liter,
            )
            total_shift_fuel_estimate = (
                _build_fuel_estimate(
                    distance_km=shift_dist_km,
                    scope="complete_shift",
                    scope_description=(
                        "Complete shift: observed pre-disruption distance plus remaining optimized routes."
                    ),
                    fuel_efficiency_km_per_liter=fuel_efficiency_km_per_liter,
                    fuel_price_per_liter=fuel_price_per_liter,
                )
                if shift_dist_km is not None
                else None
            )
        else:
            shift_dist_m = rem_dist_m
            shift_dist_km = rem_dist_km
            dist_status = "available_complete_shift"
            fuel_estimate = _build_fuel_estimate(
                distance_km=rem_dist_km,
                scope="complete_shift",
                scope_description=(
                    "Complete shift: calculated from all dispatched OR-Tools routes."
                ),
                fuel_efficiency_km_per_liter=fuel_efficiency_km_per_liter,
                fuel_price_per_liter=fuel_price_per_liter,
            )
            total_shift_fuel_estimate = fuel_estimate

        distance_breakdown = DistanceBreakdownReport(
            original_total_route_distance_meters=orig_dist_m,
            original_total_route_distance_km=orig_dist_km,
            distance_travelled_before_disruption_meters=pre_dist_m,
            distance_travelled_before_disruption_km=pre_dist_km,
            remaining_optimized_route_distance_meters=rem_dist_m,
            remaining_optimized_route_distance_km=rem_dist_km,
            estimated_total_shift_distance_meters=shift_dist_m,
            estimated_total_shift_distance_km=shift_dist_km,
            total_shift_distance_status=dist_status,
        )

        notes: list[str] = [
            (
                f"Fuel estimate ({fuel_estimate.estimate_scope}): {fuel_estimate.distance_basis_km} km "
                f"at {fuel_efficiency_km_per_liter} km/L and ${fuel_price_per_liter}/L "
                f"({fuel_estimate.estimated_fuel_liters} L, ${fuel_estimate.estimated_fuel_cost:.2f})."
            )
        ]
        if is_replanned:
            notes.append(
                f"Active remaining fleet excludes disabled vehicle(s) {disabled_ids or []}: "
                f"{len(active_fleet)} active vehicle(s), {active_cap} remaining capacity "
                f"(vs. {len(original_fleet)} original vehicles, {orig_fleet_capacity} original capacity)."
            )
            if total_shift_fuel_estimate is not None:
                notes.append(
                    f"Complete-shift fuel estimate (observed pre-disruption {pre_dist_km} km + "
                    f"remaining {rem_dist_km} km = {shift_dist_km} km): "
                    f"{total_shift_fuel_estimate.estimated_fuel_liters} L, "
                    f"${total_shift_fuel_estimate.estimated_fuel_cost:.2f}."
                )
            else:
                notes.append(
                    "Observed pre-disruption distance unavailable; total shift distance and full-shift fuel "
                    "estimate are intentionally left None rather than invented."
                )
        if metrics.time_window_violations is None:
            notes.append(
                "Time-window violations omitted (None) because active optimization scenario is CVRP."
            )
        else:
            notes.append(
                f"Evaluated VRPTW time-window violations on active routes: {metrics.time_window_violations}."
            )
        if preserved:
            notes.append(
                f"Preserved {len(preserved)} already-completed deliveries (demand={completed_demand}) "
                f"separately from {metrics.total_deliveries_completed} remaining deliveries "
                f"(demand={metrics.total_demand_delivered})."
            )

        scope_tag = (
            "remaining routes only"
            if fuel_estimate.estimate_scope == "remaining_routes"
            else "complete shift"
        )
        summary = (
            f"KPI summary ({opt_result.problem_type}, {scope_tag}): "
            f"remaining_distance={rem_dist_km} km, travel_time={metrics.total_estimated_travel_time_minutes} min, "
            f"active_vehicles={metrics.vehicles_used}/{len(active_fleet)} "
            f"(active_fleet_util={metrics.fleet_capacity_utilization:.1%}), "
            f"deliveries={metrics.total_deliveries_completed} remaining + {len(preserved)} completed "
            f"= {total_accounted}, est_fuel_cost={fuel_estimate.estimated_fuel_cost:.2f} [{fuel_estimate.estimate_scope}]."
        )

        report = LogisticsPerformanceReport(
            report_status="completed",
            dispatch_status="approved",
            problem_type=opt_result.problem_type,
            is_replanned_workflow=is_replanned,
            metrics=metrics,
            completed_before_replanning_count=len(preserved),
            completed_demand_before_replanning=completed_demand,
            remaining_deliveries_count=metrics.total_deliveries_completed,
            remaining_demand_delivered=metrics.total_demand_delivered,
            preserved_completed_deliveries=preserved,
            total_deliveries_accounted_for=total_accounted,
            completion_rate=completion_rate,
            fleet_comparison=fleet_comparison,
            distance_breakdown=distance_breakdown,
            fuel_estimate=fuel_estimate,
            total_shift_fuel_estimate=total_shift_fuel_estimate,
            notes=notes,
            summary=summary,
        )
        return report.model_dump_json()

