"""Route extraction utilities converting OR-Tools solutions into Pydantic schemas."""

from collections.abc import Sequence

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from app.schemas.order import CustomerOrder, Depot
from app.schemas.route import RouteStop, VehicleRoute
from app.schemas.vehicle import Vehicle

SOLVED_ROUTING_STATUSES = {
    routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS,
    routing_enums_pb2.RoutingSearchStatus.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED,
    routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL,
}


def get_routing_status_name(status_code: int) -> str:
    """Return the human-readable OR-Tools RoutingSearchStatus enum name."""
    try:
        return routing_enums_pb2.RoutingSearchStatus.Value.Name(status_code)
    except ValueError:
        return f"UNKNOWN_STATUS({status_code})"


def extract_vehicle_routes(
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
    active_depots: Sequence[Depot],
    customers: Sequence[CustomerOrder],
    vehicles: Sequence[Vehicle],
    distance_matrix_meters: Sequence[Sequence[int]],
    travel_time_matrices_by_vehicle: Sequence[Sequence[Sequence[int]]],
    *,
    time_dimension: pywrapcp.RoutingDimension | None = None,
    horizon_minutes: int | None = None,
    include_unused_vehicles: bool = False,
) -> list[VehicleRoute]:
    """Extract structured VehicleRoute models from a solved OR-Tools RoutingModel."""
    num_depots = len(active_depots)
    routes: list[VehicleRoute] = []

    for vehicle_idx, vehicle in enumerate(vehicles):
        index = routing.Start(vehicle_idx)
        start_node = manager.IndexToNode(index)
        starting_depot_id = active_depots[start_node].depot_id
        travel_time_matrix = travel_time_matrices_by_vehicle[vehicle_idx]

        ordered_customer_ids: list[str] = []
        stops: list[RouteStop] = []
        arrival_times: dict[str, int] = {}
        total_distance_meters = 0
        estimated_travel_time_minutes = 0
        total_demand = 0

        start_cumul = (
            solution.Min(time_dimension.CumulVar(index)) if time_dimension is not None else None
        )

        while not routing.IsEnd(index):
            from_node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            to_node = manager.IndexToNode(next_index)

            total_distance_meters += distance_matrix_meters[from_node][to_node]
            estimated_travel_time_minutes += travel_time_matrix[from_node][to_node]

            if not routing.IsEnd(next_index):
                customer = customers[to_node - num_depots]
                ordered_customer_ids.append(customer.customer_id)
                total_demand += customer.demand

                if time_dimension is not None:
                    arrival = solution.Min(time_dimension.CumulVar(next_index))
                    tw_end = (
                        customer.time_window_end
                        if customer.time_window_end is not None
                        else horizon_minutes
                    )
                    violated = (
                        arrival < customer.time_window_start
                        or (tw_end is not None and arrival > tw_end)
                    )
                    arrival_times[customer.customer_id] = arrival
                    stops.append(
                        RouteStop(
                            customer_id=customer.customer_id,
                            demand=customer.demand,
                            arrival_time_minutes=arrival,
                            service_duration_minutes=customer.service_duration_minutes,
                            departure_time_minutes=arrival + customer.service_duration_minutes,
                            time_window_start=customer.time_window_start,
                            time_window_end=tw_end,
                            time_window_violated=violated,
                        )
                    )
                else:
                    stops.append(
                        RouteStop(
                            customer_id=customer.customer_id,
                            demand=customer.demand,
                        )
                    )

            index = next_index

        end_node = manager.IndexToNode(index)
        ending_depot_id = active_depots[end_node].depot_id

        if not ordered_customer_ids and not include_unused_vehicles:
            continue

        depot_departure_time: int | None = None
        depot_return_time: int | None = None
        total_route_duration: int | None = None
        if time_dimension is not None and start_cumul is not None:
            end_cumul = solution.Min(time_dimension.CumulVar(index))
            depot_departure_time = start_cumul
            depot_return_time = end_cumul
            total_route_duration = end_cumul - start_cumul

        routes.append(
            VehicleRoute(
                vehicle_id=vehicle.vehicle_id,
                vehicle_capacity=vehicle.capacity,
                starting_depot_id=starting_depot_id,
                ending_depot_id=ending_depot_id,
                ordered_customer_ids=ordered_customer_ids,
                stops=stops,
                total_distance_meters=total_distance_meters,
                total_distance_km=round(total_distance_meters / 1000.0, 4),
                estimated_travel_time_minutes=estimated_travel_time_minutes,
                depot_departure_time_minutes=depot_departure_time,
                depot_return_time_minutes=depot_return_time,
                total_route_duration_minutes=total_route_duration,
                arrival_times_minutes=arrival_times if time_dimension is not None else None,
                total_demand=total_demand,
                capacity_utilization=round(total_demand / vehicle.capacity, 4),
            )
        )

    return routes

