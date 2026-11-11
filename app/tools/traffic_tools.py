"""CrewAI custom tools for simulated traffic exception monitoring and dynamic replanning."""

from __future__ import annotations

import math
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from app.optimization import OptimizationError, solve_cvrp, solve_vrptw
from app.schemas.logistics import (
    CompletedDeliveryRecord,
    DynamicReplanningReport,
    ImpactedVehicleSummary,
    TrafficDisruptionEvent,
    TrafficExceptionReport,
)
from app.schemas.order import CustomerOrder, Depot
from app.schemas.route import OptimizationResult
from app.schemas.vehicle import Vehicle


class TrafficDisruptionSimulationInput(BaseModel):
    """Input schema for TrafficDisruptionSimulationTool."""

    optimization_result: dict[str, Any] = Field(
        description="Initial OR-Tools OptimizationResult JSON object."
    )
    orders: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Validated customer orders used for time-window violation checks.",
    )
    disruption_event: dict[str, Any] | None = Field(
        default=None,
        description="Controlled simulated TrafficDisruptionEvent payload.",
    )


class DynamicReplanningInput(BaseModel):
    """Input schema for DynamicRouteReplanningTool."""

    depots: list[dict[str, Any]] = Field(description="Depot records.")
    vehicles: list[dict[str, Any]] = Field(description="Available fleet vehicle records.")
    orders: list[dict[str, Any]] = Field(description="Original validated customer orders.")
    initial_optimization_result: dict[str, Any] = Field(
        description="Initial feasible OptimizationResult prior to disruption."
    )
    disruption_event: dict[str, Any] = Field(
        description="Simulated TrafficDisruptionEvent specifying completed and impacted entities."
    )
    time_limit_seconds: int = Field(default=5, gt=0)
    default_horizon_minutes: int = Field(default=1440, gt=0)
    use_guided_local_search: bool = False


class TrafficDisruptionSimulationTool(BaseTool):
    """Evaluates controlled simulated traffic disruptions against planned OR-Tools routes."""

    name: str = "traffic_disruption_simulation_tool"
    description: str = (
        "Evaluates simulated traffic disruptions (clearly marked simulated), identifies "
        "affected vehicles and remaining deliveries, estimates ETA impacts, and determines "
        "whether dynamic OR-Tools replanning is recommended."
    )
    args_schema: type[BaseModel] = TrafficDisruptionSimulationInput

    def _run(
        self,
        optimization_result: dict[str, Any],
        orders: list[dict[str, Any]] | None = None,
        disruption_event: dict[str, Any] | None = None,
    ) -> str:
        """Analyze simulated traffic impacts on existing OR-Tools routes."""
        opt_result = OptimizationResult.model_validate(optimization_result)
        order_lookup = {
            order.customer_id: order
            for order in (CustomerOrder.model_validate(item) for item in (orders or []))
        }

        if disruption_event is None:
            report = TrafficExceptionReport(
                has_disruption=False,
                is_simulated=True,
                simulation_notice=(
                    "SIMULATED TRAFFIC DATA — controlled simulation (nominal baseline)"
                ),
                disruption_event=None,
                impacted_vehicles=[],
                replanning_recommended=False,
                recommendation_reason="No simulated traffic disruption event injected.",
                replanning_report=None,
                summary="Nominal simulated traffic status: no route disruptions active.",
            )
            return report.model_dump_json()

        try:
            event = TrafficDisruptionEvent.model_validate(disruption_event)
        except ValidationError as exc:
            raise ValueError(f"Invalid simulated traffic disruption event: {exc}") from exc

        if event.severity == "none":
            report = TrafficExceptionReport(
                has_disruption=False,
                is_simulated=True,
                simulation_notice=event.simulation_notice,
                disruption_event=event,
                impacted_vehicles=[],
                replanning_recommended=False,
                recommendation_reason="Simulated event severity is 'none'.",
                replanning_report=None,
                summary="Simulated traffic severity is 'none'; no replanning required.",
            )
            return report.model_dump_json()

        completed_set = set(event.completed_customer_ids)
        affected_vehicles_set = set(event.affected_vehicle_ids)
        disabled_vehicles_set = set(event.disabled_vehicle_ids)
        affected_customers_set = set(event.affected_customer_ids)
        global_impact = (
            not affected_vehicles_set
            and not disabled_vehicles_set
            and not affected_customers_set
        )

        impacted_vehicles: list[ImpactedVehicleSummary] = []
        for route in opt_result.routes:
            remaining_ids = [
                cid for cid in route.ordered_customer_ids if cid not in completed_set
            ]
            is_disabled = route.vehicle_id in disabled_vehicles_set
            touched_customers = [
                cid for cid in remaining_ids if cid in affected_customers_set
            ]
            is_vehicle_affected = (
                is_disabled
                or route.vehicle_id in affected_vehicles_set
                or bool(touched_customers)
                or (
                    global_impact
                    and (event.extra_delay_minutes > 0 or event.speed_reduction_factor < 1.0)
                )
            )
            if not is_vehicle_affected:
                continue

            slowdown_penalty = (
                int(
                    math.ceil(
                        route.estimated_travel_time_minutes
                        * ((1.0 / event.speed_reduction_factor) - 1.0)
                    )
                )
                if event.speed_reduction_factor < 1.0
                else 0
            )
            projected_delay = event.extra_delay_minutes + slowdown_penalty

            tw_violations = 0
            for stop in route.stops:
                if stop.customer_id in completed_set:
                    continue
                order = order_lookup.get(stop.customer_id)
                tw_end = (
                    stop.time_window_end
                    if stop.time_window_end is not None
                    else (order.time_window_end if order is not None else None)
                )
                if stop.arrival_time_minutes is not None and tw_end is not None:
                    projected_arrival = (
                        max(stop.arrival_time_minutes, event.elapsed_time_minutes)
                        + projected_delay
                    )
                    if projected_arrival > tw_end:
                        tw_violations += 1

            reasons: list[str] = []
            if is_disabled:
                reasons.append("vehicle disabled during disruption")
            if route.vehicle_id in affected_vehicles_set:
                reasons.append("vehicle on congested corridor")
            if touched_customers:
                reasons.append(f"impacted stops: {', '.join(touched_customers)}")
            if global_impact:
                reasons.append("network-wide simulated congestion")

            impacted_vehicles.append(
                ImpactedVehicleSummary(
                    vehicle_id=route.vehicle_id,
                    reason="; ".join(reasons),
                    affected_remaining_customer_ids=remaining_ids,
                    projected_additional_delay_minutes=projected_delay,
                    projected_time_window_violations=tw_violations,
                    disabled=is_disabled,
                )
            )

        has_remaining_work = any(
            bool(summary.affected_remaining_customer_ids)
            for summary in impacted_vehicles
        )
        has_critical_condition = any(
            summary.disabled or summary.projected_time_window_violations > 0
            for summary in impacted_vehicles
        )
        replanning_recommended = bool(
            impacted_vehicles
            and has_remaining_work
            and (
                has_critical_condition
                or event.severity in {"moderate", "high", "critical"}
            )
        )

        if replanning_recommended:
            rec_reason = (
                f"Simulated {event.severity} disruption ({event.event_id}) impacts "
                f"{len(impacted_vehicles)} vehicle(s) with uncompleted deliveries."
            )
        elif not impacted_vehicles:
            rec_reason = (
                f"Simulated event {event.event_id} does not intersect active remaining routes."
            )
        else:
            rec_reason = (
                f"Simulated {event.severity} event has minor impact without remaining stop conflicts."
            )

        report = TrafficExceptionReport(
            has_disruption=True,
            is_simulated=True,
            simulation_notice=event.simulation_notice,
            disruption_event=event,
            impacted_vehicles=impacted_vehicles,
            replanning_recommended=replanning_recommended,
            recommendation_reason=rec_reason,
            replanning_report=None,
            summary=(
                f"[SIMULATED TRAFFIC] Event {event.event_id} ({event.severity}): "
                f"{len(impacted_vehicles)} vehicle(s) impacted; "
                f"replanning_recommended={replanning_recommended}."
            ),
        )
        return report.model_dump_json()


class DynamicRouteReplanningTool(BaseTool):
    """Re-optimizes remaining deliveries with OR-Tools while preserving completed stops."""

    name: str = "dynamic_route_replanning_tool"
    description: str = (
        "Preserves already completed deliveries from the initial OR-Tools plan, updates "
        "remaining vehicle capacities/speeds/durations for simulated disruptions, and "
        "re-runs the Google OR-Tools CVRP or VRPTW solver strictly on remaining orders."
    )
    args_schema: type[BaseModel] = DynamicReplanningInput

    def _run(
        self,
        depots: list[dict[str, Any]],
        vehicles: list[dict[str, Any]],
        orders: list[dict[str, Any]],
        initial_optimization_result: dict[str, Any],
        disruption_event: dict[str, Any],
        time_limit_seconds: int = 5,
        default_horizon_minutes: int = 1440,
        use_guided_local_search: bool = False,
    ) -> str:
        """Preserve completed deliveries and re-optimize remaining orders via OR-Tools."""
        try:
            parsed_depots = [Depot.model_validate(d) for d in depots]
            parsed_vehicles = [Vehicle.model_validate(v) for v in vehicles]
            parsed_orders = [CustomerOrder.model_validate(o) for o in orders]
            initial_result = OptimizationResult.model_validate(initial_optimization_result)
            event = TrafficDisruptionEvent.model_validate(disruption_event)
        except ValidationError as exc:
            report = DynamicReplanningReport(
                replanning_triggered=True,
                status="blocked",
                is_simulated_disruption=True,
                error_type="ValidationError",
                error_message=str(exc),
                summary=f"Dynamic replanning blocked by invalid input payload: {exc}",
            )
            return report.model_dump_json()

        order_ids = {order.customer_id for order in parsed_orders}
        completed_set = set(event.completed_customer_ids)
        unknown_completed = sorted(completed_set - order_ids)
        if unknown_completed:
            report = DynamicReplanningReport(
                replanning_triggered=True,
                status="blocked",
                is_simulated_disruption=event.is_simulated,
                error_type="InvalidCompletedDeliveriesError",
                error_message=(
                    f"Completed customer IDs not found in validated orders: {unknown_completed}"
                ),
                summary="Dynamic replanning blocked due to unknown completed customer IDs.",
            )
            return report.model_dump_json()

        preserved_completed: list[CompletedDeliveryRecord] = []
        completed_demand_by_vehicle: dict[str, int] = {}
        preserved_ids: set[str] = set()

        for route in initial_result.routes:
            if route.stops:
                for stop in route.stops:
                    if stop.customer_id in completed_set and stop.customer_id not in preserved_ids:
                        preserved_ids.add(stop.customer_id)
                        completed_demand_by_vehicle[route.vehicle_id] = (
                            completed_demand_by_vehicle.get(route.vehicle_id, 0) + stop.demand
                        )
                        preserved_completed.append(
                            CompletedDeliveryRecord(
                                customer_id=stop.customer_id,
                                vehicle_id=route.vehicle_id,
                                demand=stop.demand,
                                arrival_time_minutes=stop.arrival_time_minutes,
                                departure_time_minutes=stop.departure_time_minutes,
                            )
                        )
            else:
                order_map = {o.customer_id: o for o in parsed_orders}
                for cid in route.ordered_customer_ids:
                    if cid in completed_set and cid not in preserved_ids:
                        preserved_ids.add(cid)
                        demand = order_map[cid].demand
                        completed_demand_by_vehicle[route.vehicle_id] = (
                            completed_demand_by_vehicle.get(route.vehicle_id, 0) + demand
                        )
                        preserved_completed.append(
                            CompletedDeliveryRecord(
                                customer_id=cid,
                                vehicle_id=route.vehicle_id,
                                demand=demand,
                            )
                        )

        remaining_orders = [
            order for order in parsed_orders if order.customer_id not in preserved_ids
        ]
        remaining_ids = [order.customer_id for order in remaining_orders]

        disabled_set = set(event.disabled_vehicle_ids)
        disabled_vehicle_ids = [
            v.vehicle_id for v in parsed_vehicles if v.vehicle_id in disabled_set
        ]
        original_vehicle_count = len(parsed_vehicles)
        original_fleet_capacity = sum(v.capacity for v in parsed_vehicles)
        completed_demand_total = sum(c.demand for c in preserved_completed)
        remaining_demand_total = sum(o.demand for o in remaining_orders)
        original_total_distance_meters = initial_result.metrics.total_distance_meters
        original_total_distance_km = initial_result.metrics.total_distance_km
        pre_disruption_dist_m = event.actual_distance_before_disruption_meters
        pre_disruption_dist_km = (
            round(pre_disruption_dist_m / 1000.0, 4)
            if pre_disruption_dist_m is not None
            else None
        )

        affected_set = set(event.affected_vehicle_ids)
        affected_customers_set = set(event.affected_customer_ids)
        touched_by_customer_vehicles = {
            route.vehicle_id
            for route in initial_result.routes
            if any(
                cid not in preserved_ids and cid in affected_customers_set
                for cid in route.ordered_customer_ids
            )
        }
        targeted_affected_vehicles = affected_set | touched_by_customer_vehicles
        global_slowdown = (
            not affected_set and not disabled_set and not affected_customers_set
        )
        initial_routes_by_vehicle = {
            route.vehicle_id: route
            for route in initial_result.routes
            if route.ordered_customer_ids
        }
        depot_by_id = {depot.depot_id: depot for depot in parsed_depots}

        adjusted_vehicles: list[Vehicle] = []
        vehicle_earliest_starts: dict[str, int] = {}
        infeasible_vehicle_reasons: list[str] = []

        for vehicle in parsed_vehicles:
            if vehicle.vehicle_id in disabled_set:
                continue

            used_capacity = completed_demand_by_vehicle.get(vehicle.vehicle_id, 0)
            remaining_capacity = vehicle.capacity - used_capacity
            if remaining_capacity <= 0:
                infeasible_vehicle_reasons.append(
                    f"Vehicle '{vehicle.vehicle_id}' has zero remaining capacity after completed stops."
                )
                continue

            start_depot = depot_by_id.get(vehicle.start_depot_id)
            end_depot = depot_by_id.get(vehicle.end_depot_id or vehicle.start_depot_id)
            if start_depot is None or end_depot is None:
                infeasible_vehicle_reasons.append(
                    f"Vehicle '{vehicle.vehicle_id}' references unknown depot assignment."
                )
                continue

            is_affected = (
                vehicle.vehicle_id in targeted_affected_vehicles or global_slowdown
            )
            if is_affected:
                earliest_start = (
                    max(start_depot.time_window_start, event.elapsed_time_minutes)
                    + event.extra_delay_minutes
                )
                adjusted_speed = max(
                    5.0, round(vehicle.speed_kmh * event.speed_reduction_factor, 2)
                )
                consumed_shift_budget = max(
                    0, earliest_start - start_depot.time_window_start
                )
            elif vehicle.vehicle_id not in initial_routes_by_vehicle:
                earliest_start = max(
                    start_depot.time_window_start, event.elapsed_time_minutes
                )
                adjusted_speed = vehicle.speed_kmh
                consumed_shift_budget = max(
                    0, earliest_start - start_depot.time_window_start
                )
            else:
                initial_route = initial_routes_by_vehicle[vehicle.vehicle_id]
                planned_departure = (
                    initial_route.depot_departure_time_minutes
                    if initial_route.depot_departure_time_minutes is not None
                    else start_depot.time_window_start
                )
                earliest_start = max(start_depot.time_window_start, planned_departure)
                adjusted_speed = vehicle.speed_kmh
                consumed_shift_budget = 0

            adjusted_max_duration = vehicle.max_route_duration_minutes
            if adjusted_max_duration is not None:
                adjusted_max_duration = adjusted_max_duration - consumed_shift_budget
                if adjusted_max_duration <= 0:
                    infeasible_vehicle_reasons.append(
                        f"Vehicle '{vehicle.vehicle_id}' exhausted route duration limit "
                        f"({vehicle.max_route_duration_minutes} min) after {consumed_shift_budget} min elapsed/delay."
                    )
                    continue

            if (
                start_depot.time_window_end is not None
                and earliest_start > start_depot.time_window_end
            ):
                infeasible_vehicle_reasons.append(
                    f"Vehicle '{vehicle.vehicle_id}' delayed departure ({earliest_start}) exceeds "
                    f"depot '{start_depot.depot_id}' closing window ({start_depot.time_window_end})."
                )
                continue
            if (
                end_depot.time_window_end is not None
                and earliest_start > end_depot.time_window_end
            ):
                infeasible_vehicle_reasons.append(
                    f"Vehicle '{vehicle.vehicle_id}' delayed departure ({earliest_start}) exceeds "
                    f"return depot '{end_depot.depot_id}' closing window ({end_depot.time_window_end})."
                )
                continue

            adjusted_vehicles.append(
                Vehicle(
                    vehicle_id=vehicle.vehicle_id,
                    capacity=remaining_capacity,
                    start_depot_id=vehicle.start_depot_id,
                    end_depot_id=vehicle.end_depot_id,
                    max_route_duration_minutes=adjusted_max_duration,
                    speed_kmh=adjusted_speed,
                )
            )
            vehicle_earliest_starts[vehicle.vehicle_id] = earliest_start

        active_remaining_capacity = sum(v.capacity for v in adjusted_vehicles)
        shared_metadata: dict[str, Any] = {
            "disabled_vehicle_ids": disabled_vehicle_ids,
            "active_remaining_vehicles": adjusted_vehicles,
            "original_vehicle_count": original_vehicle_count,
            "original_fleet_capacity": original_fleet_capacity,
            "active_remaining_vehicle_count": len(adjusted_vehicles),
            "active_remaining_fleet_capacity": active_remaining_capacity,
            "completed_demand_before_replanning": completed_demand_total,
            "remaining_demand_to_deliver": remaining_demand_total,
            "original_total_distance_meters": original_total_distance_meters,
            "original_total_distance_km": original_total_distance_km,
            "distance_travelled_before_disruption_meters": pre_disruption_dist_m,
            "distance_travelled_before_disruption_km": pre_disruption_dist_km,
            "vehicle_earliest_start_minutes": vehicle_earliest_starts,
        }

        if not remaining_orders:
            report = DynamicReplanningReport(
                replanning_triggered=True,
                status="replanned",
                is_simulated_disruption=event.is_simulated,
                preserved_completed_deliveries=preserved_completed,
                remaining_customer_ids=[],
                replanned_result=None,
                summary=(
                    f"All {len(preserved_completed)} validated deliveries were already completed; "
                    "no remaining OR-Tools replanning routes needed."
                ),
                **shared_metadata,
            )
            return report.model_dump_json()

        if not adjusted_vehicles:
            reason_text = (
                "; ".join(infeasible_vehicle_reasons)
                if infeasible_vehicle_reasons
                else "No operational vehicles remain available for dynamic replanning."
            )
            report = DynamicReplanningReport(
                replanning_triggered=True,
                status="infeasible",
                is_simulated_disruption=event.is_simulated,
                preserved_completed_deliveries=preserved_completed,
                remaining_customer_ids=remaining_ids,
                replanned_result=None,
                error_type="InfeasibleReplanningConstraintsError",
                error_message=reason_text,
                summary=f"Dynamic replanning infeasible: {reason_text}",
                **shared_metadata,
            )
            return report.model_dump_json()

        try:
            if initial_result.problem_type == "VRPTW":
                replanned_result = solve_vrptw(
                    depots=parsed_depots,
                    vehicles=adjusted_vehicles,
                    customers=remaining_orders,
                    default_horizon_minutes=default_horizon_minutes,
                    time_limit_seconds=time_limit_seconds,
                    use_guided_local_search=use_guided_local_search,
                    vehicle_earliest_start_minutes=vehicle_earliest_starts,
                )
            else:
                replanned_result = solve_cvrp(
                    depots=parsed_depots,
                    vehicles=adjusted_vehicles,
                    customers=remaining_orders,
                    time_limit_seconds=time_limit_seconds,
                    use_guided_local_search=use_guided_local_search,
                )
        except OptimizationError as exc:
            report = DynamicReplanningReport(
                replanning_triggered=True,
                status="infeasible",
                is_simulated_disruption=event.is_simulated,
                preserved_completed_deliveries=preserved_completed,
                remaining_customer_ids=remaining_ids,
                replanned_result=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
                summary=(
                    f"[SIMULATED TRAFFIC] Preserved {len(preserved_completed)} completed deliveries, "
                    f"but OR-Tools dynamic replanning failed for {len(remaining_orders)} remaining order(s): {exc}"
                ),
                **shared_metadata,
            )
            return report.model_dump_json()

        report = DynamicReplanningReport(
            replanning_triggered=True,
            status="replanned",
            is_simulated_disruption=event.is_simulated,
            preserved_completed_deliveries=preserved_completed,
            remaining_customer_ids=remaining_ids,
            replanned_result=replanned_result,
            summary=(
                f"[SIMULATED TRAFFIC] Preserved {len(preserved_completed)} completed delivery/deliveries; "
                f"OR-Tools {replanned_result.problem_type} re-optimized {len(remaining_orders)} remaining order(s) "
                f"across {replanned_result.metrics.vehicles_used} active vehicle(s), "
                f"distance={replanned_result.metrics.total_distance_km} km."
            ),
            **shared_metadata,
        )
        return report.model_dump_json()

