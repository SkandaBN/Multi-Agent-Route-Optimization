"""Vehicle Routing Problem with Time Windows (VRPTW) solver using Google OR-Tools.

Extends the base CVRP model preparation with travel-time + service-duration
transit callbacks, OR-Tools Time dimension cumul variables, customer delivery
time windows, depot operating hours, and vehicle maximum route durations.
"""

from collections.abc import Callable, Mapping, Sequence

from ortools.constraint_solver import pywrapcp

from app.optimization import InfeasibleRoutingError, InvalidRoutingInputError
from app.optimization.cvrp import (
    build_search_parameters,
    prepare_base_routing_model,
    solve_or_raise,
)
from app.optimization.distance_matrix import DistanceMatrixProvider
from app.optimization.metrics import calculate_performance_metrics
from app.optimization.route_extractor import extract_vehicle_routes
from app.schemas.order import CustomerOrder, Depot
from app.schemas.route import OptimizationResult
from app.schemas.vehicle import Vehicle

DEFAULT_HORIZON_MINUTES = 1440


def _compute_planning_horizon(
    active_depots: Sequence[Depot],
    vehicles: Sequence[Vehicle],
    customers: Sequence[CustomerOrder],
    default_horizon_minutes: int,
    vehicle_earliest_start_minutes: Mapping[str, int] | None = None,
) -> int:
    """Determine an upper-bound planning horizon covering all configured windows."""
    if default_horizon_minutes <= 0:
        raise InvalidRoutingInputError("default_horizon_minutes must be positive.")

    explicit_upper_bounds: list[int] = [default_horizon_minutes]
    for depot in active_depots:
        explicit_upper_bounds.append(depot.time_window_start)
        if depot.time_window_end is not None:
            explicit_upper_bounds.append(depot.time_window_end)
    for customer in customers:
        explicit_upper_bounds.append(customer.time_window_start + customer.service_duration_minutes)
        if customer.time_window_end is not None:
            explicit_upper_bounds.append(
                customer.time_window_end + customer.service_duration_minutes
            )
    for vehicle in vehicles:
        if vehicle.max_route_duration_minutes is not None:
            explicit_upper_bounds.append(vehicle.max_route_duration_minutes)
    if vehicle_earliest_start_minutes:
        explicit_upper_bounds.extend(vehicle_earliest_start_minutes.values())

    return max(explicit_upper_bounds)


def _register_time_callbacks(
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    travel_time_matrices_by_vehicle: Sequence[Sequence[Sequence[int]]],
    service_durations: Sequence[int],
) -> list[int]:
    """Register OR-Tools time transit callbacks (travel time + source service time)."""
    callback_indices_by_matrix_id: dict[int, int] = {}
    vehicle_callbacks: list[int] = []

    for tt_matrix in travel_time_matrices_by_vehicle:
        matrix_id = id(tt_matrix)
        if matrix_id not in callback_indices_by_matrix_id:

            def _make_callback(
                matrix: Sequence[Sequence[int]],
            ) -> Callable[[int, int], int]:
                def _time_callback(from_index: int, to_index: int) -> int:
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    return matrix[from_node][to_node] + service_durations[from_node]

                return _time_callback

            cb_idx = routing.RegisterTransitCallback(_make_callback(tt_matrix))
            callback_indices_by_matrix_id[matrix_id] = cb_idx
        vehicle_callbacks.append(callback_indices_by_matrix_id[matrix_id])

    return vehicle_callbacks


def solve_vrptw(
    depots: Sequence[Depot],
    vehicles: Sequence[Vehicle],
    customers: Sequence[CustomerOrder],
    *,
    distance_provider: DistanceMatrixProvider | None = None,
    default_horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
    time_limit_seconds: int = 5,
    use_guided_local_search: bool = False,
    vehicle_earliest_start_minutes: Mapping[str, int] | None = None,
) -> OptimizationResult:
    """Solve a single-depot or multi-depot VRPTW instance using Google OR-Tools."""
    prepared = prepare_base_routing_model(
        depots,
        vehicles,
        customers,
        distance_provider=distance_provider,
    )

    if vehicle_earliest_start_minutes:
        known_vehicle_ids = {v.vehicle_id for v in prepared.vehicles}
        for vid, start_min in vehicle_earliest_start_minutes.items():
            if vid not in known_vehicle_ids:
                raise InvalidRoutingInputError(
                    f"vehicle_earliest_start_minutes references unknown vehicle '{vid}'."
                )
            if start_min < 0:
                raise InvalidRoutingInputError(
                    f"vehicle_earliest_start_minutes for '{vid}' cannot be negative ({start_min})."
                )

    horizon = _compute_planning_horizon(
        prepared.active_depots,
        prepared.vehicles,
        prepared.customers,
        default_horizon_minutes,
        vehicle_earliest_start_minutes=vehicle_earliest_start_minutes,
    )

    num_depots = len(prepared.active_depots)
    service_durations = [0] * num_depots + [
        customer.service_duration_minutes for customer in prepared.customers
    ]

    vehicle_time_callbacks = _register_time_callbacks(
        prepared.manager,
        prepared.routing,
        prepared.travel_time_matrices_by_vehicle,
        service_durations,
    )

    unique_callbacks = set(vehicle_time_callbacks)
    if len(unique_callbacks) == 1:
        prepared.routing.AddDimension(
            vehicle_time_callbacks[0],
            horizon,
            horizon,
            False,
            "Time",
        )
    else:
        prepared.routing.AddDimensionWithVehicleTransits(
            vehicle_time_callbacks,
            horizon,
            horizon,
            False,
            "Time",
        )

    time_dimension = prepared.routing.GetDimensionOrDie("Time")
    depot_by_id = {depot.depot_id: depot for depot in prepared.active_depots}

    for customer_offset, customer in enumerate(prepared.customers):
        node_idx = num_depots + customer_offset
        routing_idx = prepared.manager.NodeToIndex(node_idx)
        tw_end = customer.time_window_end if customer.time_window_end is not None else horizon
        if customer.time_window_start > tw_end:
            raise InvalidRoutingInputError(
                f"Customer {customer.customer_id} has invalid time window "
                f"[{customer.time_window_start}, {tw_end}]."
            )
        time_dimension.CumulVar(routing_idx).SetRange(customer.time_window_start, tw_end)

    for vehicle_idx, vehicle in enumerate(prepared.vehicles):
        start_depot = depot_by_id[vehicle.start_depot_id]
        end_depot = depot_by_id[vehicle.end_depot_id or vehicle.start_depot_id]

        start_idx = prepared.routing.Start(vehicle_idx)
        end_idx = prepared.routing.End(vehicle_idx)

        start_tw_start = start_depot.time_window_start
        if (
            vehicle_earliest_start_minutes
            and vehicle.vehicle_id in vehicle_earliest_start_minutes
        ):
            start_tw_start = max(
                start_tw_start, vehicle_earliest_start_minutes[vehicle.vehicle_id]
            )
        start_tw_end = (
            start_depot.time_window_end if start_depot.time_window_end is not None else horizon
        )
        end_tw_end = (
            end_depot.time_window_end if end_depot.time_window_end is not None else horizon
        )
        if start_tw_start > start_tw_end:
            raise InfeasibleRoutingError(
                f"Vehicle {vehicle.vehicle_id} earliest start time ({start_tw_start}) "
                f"exceeds depot '{start_depot.depot_id}' closing window ({start_tw_end})."
            )
        if start_tw_start > end_tw_end:
            raise InfeasibleRoutingError(
                f"Vehicle {vehicle.vehicle_id} earliest start time ({start_tw_start}) "
                f"exceeds return depot '{end_depot.depot_id}' closing window ({end_tw_end})."
            )
        time_dimension.CumulVar(start_idx).SetRange(start_tw_start, start_tw_end)
        time_dimension.CumulVar(end_idx).SetRange(end_depot.time_window_start, end_tw_end)

        if vehicle.max_route_duration_minutes is not None:
            time_dimension.SetSpanUpperBoundForVehicle(
                vehicle.max_route_duration_minutes, vehicle_idx
            )

        prepared.routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(start_idx))
        prepared.routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(end_idx))

    search_params = build_search_parameters(
        time_limit_seconds=time_limit_seconds,
        use_guided_local_search=use_guided_local_search,
    )
    solution, status_label = solve_or_raise(prepared.routing, search_params)

    routes = extract_vehicle_routes(
        prepared.manager,
        prepared.routing,
        solution,
        prepared.active_depots,
        prepared.customers,
        prepared.vehicles,
        prepared.distance_matrix_meters,
        prepared.travel_time_matrices_by_vehicle,
        time_dimension=time_dimension,
        horizon_minutes=horizon,
    )
    metrics = calculate_performance_metrics(
        routes,
        prepared.vehicles,
        evaluate_time_windows=True,
    )

    if metrics.time_window_violations and metrics.time_window_violations > 0:
        raise InfeasibleRoutingError(
            f"Schedule contains {metrics.time_window_violations} time-window violation(s)."
        )

    vehicle_by_id = {v.vehicle_id: v for v in prepared.vehicles}
    for route in routes:
        max_dur = vehicle_by_id[route.vehicle_id].max_route_duration_minutes
        if (
            max_dur is not None
            and route.total_route_duration_minutes is not None
            and route.total_route_duration_minutes > max_dur
        ):
            raise InfeasibleRoutingError(
                f"Vehicle {route.vehicle_id} exceeded max_route_duration_minutes ({max_dur})."
            )

    return OptimizationResult(
        problem_type="VRPTW",
        solver_status=status_label,
        routes=routes,
        metrics=metrics,
    )

