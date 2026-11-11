"""Performance metrics calculation and schedule verification utilities."""

from collections.abc import Sequence

from app.schemas.route import PerformanceMetrics, VehicleRoute
from app.schemas.vehicle import Vehicle


def validate_vrptw_schedule(routes: Sequence[VehicleRoute]) -> int:
    """Count time-window violations across all stops in VRPTW routes.

    Raises ValueError if any stop lacks computed arrival time or window bounds.
    """
    violations = 0
    for route in routes:
        for stop in route.stops:
            if stop.arrival_time_minutes is None or stop.time_window_start is None:
                raise ValueError(
                    f"Stop {stop.customer_id} on vehicle {route.vehicle_id} is missing "
                    "calculated VRPTW schedule metadata."
                )
            early_violation = stop.arrival_time_minutes < stop.time_window_start
            late_violation = (
                stop.time_window_end is not None
                and stop.arrival_time_minutes > stop.time_window_end
            )
            if early_violation or late_violation:
                violations += 1
    return violations


def calculate_performance_metrics(
    routes: Sequence[VehicleRoute],
    vehicles: Sequence[Vehicle],
    *,
    evaluate_time_windows: bool,
) -> PerformanceMetrics:
    """Compute aggregate routing performance metrics from extracted vehicle routes.

    Time-window violations are only reported (as an integer >= 0) when
    ``evaluate_time_windows=True``; otherwise ``time_window_violations`` remains
    ``None`` so unverified metrics are never reported as zero.
    """
    if not vehicles:
        raise ValueError("vehicles cannot be empty when calculating performance metrics.")

    route_by_vehicle = {route.vehicle_id: route for route in routes}
    used_routes = [route for route in routes if route.ordered_customer_ids]

    total_distance_meters = sum(route.total_distance_meters for route in used_routes)
    total_estimated_travel_time = sum(
        route.estimated_travel_time_minutes for route in used_routes
    )
    total_deliveries = sum(len(route.ordered_customer_ids) for route in used_routes)
    total_demand_delivered = sum(route.total_demand for route in used_routes)
    total_fleet_capacity = sum(vehicle.capacity for vehicle in vehicles)

    utilization_by_id: dict[str, float] = {}
    for vehicle in vehicles:
        route = route_by_vehicle.get(vehicle.vehicle_id)
        assigned_demand = route.total_demand if route is not None else 0
        utilization_by_id[vehicle.vehicle_id] = round(assigned_demand / vehicle.capacity, 4)

    vehicles_used = len(used_routes)
    average_used_utilization = (
        round(sum(route.capacity_utilization for route in used_routes) / vehicles_used, 4)
        if vehicles_used > 0
        else 0.0
    )
    fleet_capacity_utilization = round(total_demand_delivered / total_fleet_capacity, 4)

    time_window_violations: int | None = None
    if evaluate_time_windows:
        time_window_violations = validate_vrptw_schedule(used_routes)

    return PerformanceMetrics(
        total_distance_meters=total_distance_meters,
        total_distance_km=round(total_distance_meters / 1000.0, 4),
        total_estimated_travel_time_minutes=total_estimated_travel_time,
        vehicles_used=vehicles_used,
        total_vehicles_available=len(vehicles),
        vehicle_utilization_by_id=utilization_by_id,
        average_used_vehicle_utilization=average_used_utilization,
        fleet_capacity_utilization=fleet_capacity_utilization,
        total_deliveries_completed=total_deliveries,
        total_demand_delivered=total_demand_delivered,
        time_window_violations=time_window_violations,
    )

