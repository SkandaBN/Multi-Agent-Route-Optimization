"""Capacitated Vehicle Routing Problem (CVRP) solver using Google OR-Tools.

Supports both single-depot and multi-depot vehicle routing and exposes shared
model preparation helpers extended by the VRPTW solver.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from app.optimization import (
    InfeasibleRoutingError,
    InsufficientCapacityError,
    InvalidRoutingInputError,
)
from app.optimization.distance_matrix import (
    DistanceMatrixProvider,
    build_distance_matrix,
    build_travel_time_matrix,
)
from app.optimization.metrics import calculate_performance_metrics
from app.optimization.route_extractor import (
    SOLVED_ROUTING_STATUSES,
    extract_vehicle_routes,
    get_routing_status_name,
)
from app.schemas.order import CustomerOrder, Depot
from app.schemas.route import OptimizationResult
from app.schemas.vehicle import Vehicle


@dataclass
class PreparedRoutingModel:
    """Shared OR-Tools CVRP model state used by both CVRP and VRPTW solvers."""

    active_depots: list[Depot]
    customers: list[CustomerOrder]
    vehicles: list[Vehicle]
    distance_matrix_meters: list[list[int]]
    travel_time_matrices_by_vehicle: list[list[list[int]]]
    manager: pywrapcp.RoutingIndexManager
    routing: pywrapcp.RoutingModel


def _validate_unique_ids(items: Sequence[str], label: str) -> None:
    if len(set(items)) != len(items):
        raise InvalidRoutingInputError(f"Duplicate {label} identifiers detected.")


def validate_routing_inputs(
    depots: Sequence[Depot],
    vehicles: Sequence[Vehicle],
    customers: Sequence[CustomerOrder],
) -> list[Depot]:
    """Validate depots, vehicles, and customers and return active depots used by vehicles."""
    if not depots:
        raise InvalidRoutingInputError("At least one depot is required.")
    if not vehicles:
        raise InvalidRoutingInputError("At least one vehicle is required.")
    if not customers:
        raise InvalidRoutingInputError("Customer list cannot be empty.")

    _validate_unique_ids([d.depot_id for d in depots], "depot_id")
    _validate_unique_ids([v.vehicle_id for v in vehicles], "vehicle_id")
    _validate_unique_ids([c.customer_id for c in customers], "customer_id")

    depot_map = {d.depot_id: d for d in depots}
    referenced_depot_ids: set[str] = set()

    for vehicle in vehicles:
        if vehicle.start_depot_id not in depot_map:
            raise InvalidRoutingInputError(
                f"Vehicle {vehicle.vehicle_id} references unknown start_depot_id "
                f"'{vehicle.start_depot_id}'."
            )
        end_depot_id = vehicle.end_depot_id or vehicle.start_depot_id
        if end_depot_id not in depot_map:
            raise InvalidRoutingInputError(
                f"Vehicle {vehicle.vehicle_id} references unknown end_depot_id '{end_depot_id}'."
            )
        if vehicle.capacity <= 0:
            raise InvalidRoutingInputError(
                f"Vehicle {vehicle.vehicle_id} capacity must be positive, got {vehicle.capacity}."
            )
        referenced_depot_ids.add(vehicle.start_depot_id)
        referenced_depot_ids.add(end_depot_id)

    for customer in customers:
        if customer.demand < 0:
            raise InvalidRoutingInputError(
                f"Customer {customer.customer_id} has negative demand ({customer.demand})."
            )

    total_demand = sum(customer.demand for customer in customers)
    total_capacity = sum(vehicle.capacity for vehicle in vehicles)
    if total_demand > total_capacity:
        raise InsufficientCapacityError(
            f"Total customer demand ({total_demand}) exceeds total fleet capacity "
            f"({total_capacity})."
        )

    max_customer_demand = max(customer.demand for customer in customers)
    max_vehicle_capacity = max(vehicle.capacity for vehicle in vehicles)
    if max_customer_demand > max_vehicle_capacity:
        raise InsufficientCapacityError(
            f"Maximum single customer demand ({max_customer_demand}) exceeds maximum "
            f"vehicle capacity ({max_vehicle_capacity})."
        )

    return [depot for depot in depots if depot.depot_id in referenced_depot_ids]


def prepare_base_routing_model(
    depots: Sequence[Depot],
    vehicles: Sequence[Vehicle],
    customers: Sequence[CustomerOrder],
    *,
    distance_provider: DistanceMatrixProvider | None = None,
) -> PreparedRoutingModel:
    """Validate inputs, build matrices, and construct the base OR-Tools CVRP model."""
    active_depots = validate_routing_inputs(depots, vehicles, customers)
    customer_list = list(customers)
    vehicle_list = list(vehicles)

    all_locations = [*active_depots, *customer_list]
    distance_matrix = build_distance_matrix(all_locations, provider=distance_provider)

    speed_cache: dict[float, list[list[int]]] = {}
    travel_time_matrices: list[list[list[int]]] = []
    for vehicle in vehicle_list:
        if vehicle.speed_kmh not in speed_cache:
            speed_cache[vehicle.speed_kmh] = build_travel_time_matrix(
                distance_matrix, speed_kmh=vehicle.speed_kmh
            )
        travel_time_matrices.append(speed_cache[vehicle.speed_kmh])

    depot_index_by_id = {depot.depot_id: idx for idx, depot in enumerate(active_depots)}
    num_locations = len(all_locations)
    num_vehicles = len(vehicle_list)

    if len(active_depots) == 1:
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
    else:
        starts = [depot_index_by_id[v.start_depot_id] for v in vehicle_list]
        ends = [
            depot_index_by_id[v.end_depot_id or v.start_depot_id] for v in vehicle_list
        ]
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, starts, ends)

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    num_depots = len(active_depots)
    demands = [0] * num_depots + [customer.demand for customer in customer_list]

    def demand_callback(from_index: int) -> int:
        return demands[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [vehicle.capacity for vehicle in vehicle_list],
        True,
        "Capacity",
    )

    return PreparedRoutingModel(
        active_depots=active_depots,
        customers=customer_list,
        vehicles=vehicle_list,
        distance_matrix_meters=distance_matrix,
        travel_time_matrices_by_vehicle=travel_time_matrices,
        manager=manager,
        routing=routing,
    )


def build_search_parameters(
    *,
    time_limit_seconds: int = 5,
    use_guided_local_search: bool = False,
) -> pywrapcp.DefaultRoutingSearchParameters:
    """Create OR-Tools routing search parameters."""
    if time_limit_seconds <= 0:
        raise InvalidRoutingInputError("time_limit_seconds must be positive.")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    if use_guided_local_search:
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
    else:
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        )
    params.time_limit.FromSeconds(time_limit_seconds)
    return params


def solve_or_raise(
    routing: pywrapcp.RoutingModel,
    search_parameters: pywrapcp.DefaultRoutingSearchParameters,
) -> tuple[pywrapcp.Assignment, str]:
    """Solve the OR-Tools model or raise InfeasibleRoutingError when unsolved."""
    solution = routing.SolveWithParameters(search_parameters)
    status_code = routing.status()
    status_label = get_routing_status_name(status_code)

    if solution is None or status_code not in SOLVED_ROUTING_STATUSES:
        raise InfeasibleRoutingError(
            f"OR-Tools found no feasible routing solution (status={status_label})."
        )
    return solution, status_label


def solve_cvrp(
    depots: Sequence[Depot],
    vehicles: Sequence[Vehicle],
    customers: Sequence[CustomerOrder],
    *,
    distance_provider: DistanceMatrixProvider | None = None,
    time_limit_seconds: int = 5,
    use_guided_local_search: bool = False,
) -> OptimizationResult:
    """Solve a single-depot or multi-depot CVRP instance using Google OR-Tools."""
    prepared = prepare_base_routing_model(
        depots,
        vehicles,
        customers,
        distance_provider=distance_provider,
    )
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
        time_dimension=None,
    )
    metrics = calculate_performance_metrics(
        routes,
        prepared.vehicles,
        evaluate_time_windows=False,
    )

    return OptimizationResult(
        problem_type="CVRP",
        solver_status=status_label,
        routes=routes,
        metrics=metrics,
    )


def load_dataset_from_csv(
    data_dir: Path | str = Path("data"),
) -> tuple[list[Depot], list[Vehicle], list[CustomerOrder]]:
    """Load depots, vehicles, and customer orders from CSV files in ``data_dir``."""
    base_path = Path(data_dir)

    with (base_path / "depots.csv").open("r", encoding="utf-8", newline="") as f:
        depots = [
            Depot(
                depot_id=row["depot_id"],
                name=row.get("name", ""),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                time_window_start=int(row["time_window_start"]),
                time_window_end=(
                    int(row["time_window_end"]) if row.get("time_window_end") else None
                ),
            )
            for row in csv.DictReader(f)
        ]

    with (base_path / "vehicles.csv").open("r", encoding="utf-8", newline="") as f:
        vehicles = [
            Vehicle(
                vehicle_id=row["vehicle_id"],
                capacity=int(row["capacity"]),
                start_depot_id=row["start_depot_id"],
                end_depot_id=row.get("end_depot_id") or None,
                max_route_duration_minutes=(
                    int(row["max_route_duration_minutes"])
                    if row.get("max_route_duration_minutes")
                    else None
                ),
                speed_kmh=float(row.get("speed_kmh") or 40.0),
            )
            for row in csv.DictReader(f)
        ]

    with (base_path / "customers.csv").open("r", encoding="utf-8", newline="") as f:
        customers = [
            CustomerOrder(
                customer_id=row["customer_id"],
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                demand=int(row["demand"]),
                service_duration_minutes=int(row.get("service_duration_minutes") or 0),
                time_window_start=int(row.get("time_window_start") or 0),
                time_window_end=(
                    int(row["time_window_end"]) if row.get("time_window_end") else None
                ),
            )
            for row in csv.DictReader(f)
        ]

    return depots, vehicles, customers

