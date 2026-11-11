"""CrewAI custom tools for validating orders and analyzing fleet capacity."""

from __future__ import annotations

from collections import Counter
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from app.schemas.logistics import (
    FleetCapacityReport,
    OrderValidationReport,
    RejectedOrder,
    ValidatedOrderMetadata,
    VehicleDepotAssignment,
)
from app.schemas.order import CustomerOrder, Depot
from app.schemas.vehicle import Vehicle


class OrderValidationInput(BaseModel):
    """Input schema for OrderValidationTool."""

    orders: list[dict[str, Any]] = Field(
        description="List of raw incoming customer order records to validate."
    )


class FleetCapacityAnalysisInput(BaseModel):
    """Input schema for FleetCapacityAnalysisTool."""

    depots: list[dict[str, Any]] = Field(
        description="Available depot definitions."
    )
    vehicles: list[dict[str, Any]] = Field(
        description="Fleet vehicle definitions."
    )
    valid_orders: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Validated customer order records requiring delivery capacity.",
    )
    unavailable_vehicle_ids: list[str] = Field(
        default_factory=list,
        description="Optional vehicle IDs temporarily offline or unavailable.",
    )


def _format_validation_error(exc: ValidationError) -> str:
    """Produce a concise single-line summary of a Pydantic ValidationError."""
    parts = [
        f"{'.'.join(str(loc) for loc in err['loc']) or 'order'}: {err['msg']}"
        for err in exc.errors()
    ]
    return "; ".join(parts)


class OrderValidationTool(BaseTool):
    """Validates incoming delivery orders against Phase II CustomerOrder schemas."""

    name: str = "order_validation_tool"
    description: str = (
        "Validates incoming customer orders (IDs, coordinates, demands, service "
        "durations, and time windows), rejects invalid or duplicate entries safely, "
        "and returns a structured OrderValidationReport JSON payload."
    )
    args_schema: type[BaseModel] = OrderValidationInput

    def _run(self, orders: list[dict[str, Any]]) -> str:
        """Validate customer orders and partition into valid and rejected lists."""
        valid_orders: list[CustomerOrder] = []
        rejected_orders: list[RejectedOrder] = []
        metadata: list[ValidatedOrderMetadata] = []
        seen_ids: set[str] = set()

        for index, raw_item in enumerate(orders):
            if not isinstance(raw_item, dict):
                rejected_orders.append(
                    RejectedOrder(
                        customer_id=f"<index_{index}>",
                        reason="Order payload entry must be a mapping/dictionary.",
                        raw_data={"value": str(raw_item)},
                    )
                )
                continue

            raw_id = str(raw_item.get("customer_id", "")).strip() or f"<index_{index}>"
            try:
                parsed = CustomerOrder.model_validate(raw_item)
            except ValidationError as exc:
                rejected_orders.append(
                    RejectedOrder(
                        customer_id=raw_id,
                        reason=_format_validation_error(exc),
                        raw_data=raw_item,
                    )
                )
                continue

            if parsed.customer_id in seen_ids:
                rejected_orders.append(
                    RejectedOrder(
                        customer_id=parsed.customer_id,
                        reason=f"Duplicate customer_id '{parsed.customer_id}'.",
                        raw_data=raw_item,
                    )
                )
                continue

            seen_ids.add(parsed.customer_id)
            valid_orders.append(parsed)

            has_tw = parsed.time_window_end is not None or parsed.time_window_start > 0
            narrow_tw = (
                parsed.time_window_end is not None
                and (parsed.time_window_end - parsed.time_window_start) <= 120
            )
            priority = "high" if (narrow_tw or parsed.demand >= 25) else "normal"
            metadata.append(
                ValidatedOrderMetadata(
                    customer_id=parsed.customer_id,
                    demand=parsed.demand,
                    has_time_window=has_tw,
                    priority=priority,
                )
            )

        total_demand = sum(o.demand for o in valid_orders)
        has_time_windows = any(m.has_time_window for m in metadata)
        is_valid_for_opt = len(valid_orders) > 0

        if not orders:
            summary = "No orders submitted; routing cannot proceed with an empty order batch."
        elif not valid_orders:
            summary = f"All {len(orders)} submitted orders failed validation; routing blocked."
        elif rejected_orders:
            summary = (
                f"Validated {len(valid_orders)} orders (total demand {total_demand}) "
                f"and safely rejected {len(rejected_orders)} invalid order(s)."
            )
        else:
            summary = (
                f"All {len(valid_orders)} submitted orders passed validation "
                f"(total demand {total_demand}, time_windows={has_time_windows})."
            )

        report = OrderValidationReport(
            valid_orders=valid_orders,
            rejected_orders=rejected_orders,
            order_metadata=metadata,
            total_submitted=len(orders),
            valid_count=len(valid_orders),
            rejected_count=len(rejected_orders),
            total_valid_demand=total_demand,
            has_time_windows=has_time_windows,
            is_valid_for_optimization=is_valid_for_opt,
            summary=summary,
        )
        return report.model_dump_json()


class FleetCapacityAnalysisTool(BaseTool):
    """Analyzes fleet availability, depot assignments, and capacity sufficiency."""

    name: str = "fleet_capacity_analysis_tool"
    description: str = (
        "Evaluates fleet vehicles, depot assignments, availability, total capacity, "
        "and identifies capacity insufficiency before OR-Tools optimization."
    )
    args_schema: type[BaseModel] = FleetCapacityAnalysisInput

    def _run(
        self,
        depots: list[dict[str, Any]],
        vehicles: list[dict[str, Any]],
        valid_orders: list[dict[str, Any]] | None = None,
        unavailable_vehicle_ids: list[str] | None = None,
    ) -> str:
        """Analyze depots, vehicles, and demand capacity requirements."""
        issues: list[str] = []
        parsed_depots: list[Depot] = []
        depot_ids: set[str] = set()

        for raw_depot in depots:
            try:
                depot = Depot.model_validate(raw_depot)
                if depot.depot_id in depot_ids:
                    issues.append(f"Duplicate depot_id '{depot.depot_id}' ignored.")
                    continue
                depot_ids.add(depot.depot_id)
                parsed_depots.append(depot)
            except ValidationError as exc:
                issues.append(f"Invalid depot entry: {_format_validation_error(exc)}")

        parsed_orders: list[CustomerOrder] = []
        for raw_order in valid_orders or []:
            try:
                parsed_orders.append(CustomerOrder.model_validate(raw_order))
            except ValidationError as exc:
                issues.append(f"Invalid order entry in fleet check: {_format_validation_error(exc)}")

        offline_ids = set(unavailable_vehicle_ids or [])
        available_vehicles: list[Vehicle] = []
        unavailable_assignments: list[VehicleDepotAssignment] = []
        all_assignments: list[VehicleDepotAssignment] = []
        seen_vehicle_ids: set[str] = set()

        for raw_vehicle in vehicles:
            try:
                vehicle = Vehicle.model_validate(raw_vehicle)
            except ValidationError as exc:
                issues.append(f"Invalid vehicle entry: {_format_validation_error(exc)}")
                continue

            if vehicle.vehicle_id in seen_vehicle_ids:
                issues.append(f"Duplicate vehicle_id '{vehicle.vehicle_id}' ignored.")
                continue
            seen_vehicle_ids.add(vehicle.vehicle_id)

            end_depot = vehicle.end_depot_id or vehicle.start_depot_id
            reason: str | None = None
            if vehicle.vehicle_id in offline_ids:
                reason = f"Vehicle '{vehicle.vehicle_id}' marked unavailable."
            elif vehicle.start_depot_id not in depot_ids:
                reason = (
                    f"Unknown start_depot_id '{vehicle.start_depot_id}' "
                    f"for vehicle '{vehicle.vehicle_id}'."
                )
            elif end_depot not in depot_ids:
                reason = (
                    f"Unknown end_depot_id '{end_depot}' "
                    f"for vehicle '{vehicle.vehicle_id}'."
                )

            assignment = VehicleDepotAssignment(
                vehicle_id=vehicle.vehicle_id,
                capacity=vehicle.capacity,
                start_depot_id=vehicle.start_depot_id,
                end_depot_id=end_depot,
                max_route_duration_minutes=vehicle.max_route_duration_minutes,
                speed_kmh=vehicle.speed_kmh,
                available=reason is None,
                unavailability_reason=reason,
            )
            all_assignments.append(assignment)
            if reason is None:
                available_vehicles.append(vehicle)
            else:
                unavailable_assignments.append(assignment)
                issues.append(reason)

        depot_counts = dict(
            sorted(Counter(v.start_depot_id for v in available_vehicles).items())
        )
        total_capacity = sum(v.capacity for v in available_vehicles)
        total_demand = sum(o.demand for o in parsed_orders)
        slack = total_capacity - total_demand
        max_vehicle_cap = max((v.capacity for v in available_vehicles), default=0)
        max_order_demand = max((o.demand for o in parsed_orders), default=0)

        if not parsed_depots:
            issues.append("No valid depots available for dispatch.")
        if not available_vehicles:
            issues.append("No available vehicles assigned to valid depots.")
        if total_capacity < total_demand:
            issues.append(
                f"Total available fleet capacity ({total_capacity}) is below total "
                f"required demand ({total_demand})."
            )
        if parsed_orders and max_order_demand > max_vehicle_cap:
            issues.append(
                f"Largest single order demand ({max_order_demand}) exceeds maximum "
                f"single vehicle capacity ({max_vehicle_cap})."
            )

        capacity_sufficient = (
            bool(parsed_depots)
            and bool(available_vehicles)
            and total_capacity >= total_demand
            and (not parsed_orders or max_order_demand <= max_vehicle_cap)
        )

        if capacity_sufficient:
            summary = (
                f"{len(available_vehicles)} vehicle(s) available across {len(parsed_depots)} "
                f"depot(s) providing {total_capacity} capacity for {total_demand} demand "
                f"(slack={slack})."
            )
        else:
            summary = f"Fleet capacity check failed: {'; '.join(issues)}"

        report = FleetCapacityReport(
            depots=parsed_depots,
            available_vehicles=available_vehicles,
            unavailable_vehicles=unavailable_assignments,
            assignments=all_assignments,
            depot_vehicle_counts=depot_counts,
            total_fleet_capacity=total_capacity,
            total_required_demand=total_demand,
            capacity_slack=slack,
            max_single_vehicle_capacity=max_vehicle_cap,
            max_single_order_demand=max_order_demand,
            capacity_sufficient=capacity_sufficient,
            issues=issues,
            summary=summary,
        )
        return report.model_dump_json()

