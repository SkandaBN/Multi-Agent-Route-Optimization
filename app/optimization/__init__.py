"""OR-Tools optimization solvers, route extraction, and performance metrics."""


class OptimizationError(Exception):
    """Base exception raised by the route optimization engine."""


class InvalidRoutingInputError(OptimizationError, ValueError):
    """Raised when routing inputs fail structural or domain validation."""


class InsufficientCapacityError(OptimizationError):
    """Raised when fleet vehicle capacities cannot satisfy customer demand."""


class InfeasibleRoutingError(OptimizationError):
    """Raised when OR-Tools cannot find a feasible solution for the problem."""


from app.optimization.cvrp import load_dataset_from_csv, solve_cvrp
from app.optimization.distance_matrix import (
    DistanceMatrixProvider,
    HaversineDistanceMatrixProvider,
    build_distance_matrix,
    build_distance_matrix_km,
    build_travel_time_matrix,
    haversine_distance_km,
    haversine_distance_meters,
)
from app.optimization.metrics import calculate_performance_metrics, validate_vrptw_schedule
from app.optimization.route_extractor import extract_vehicle_routes, get_routing_status_name
from app.optimization.vrptw import solve_vrptw

__all__ = [
    "DistanceMatrixProvider",
    "HaversineDistanceMatrixProvider",
    "InfeasibleRoutingError",
    "InsufficientCapacityError",
    "InvalidRoutingInputError",
    "OptimizationError",
    "build_distance_matrix",
    "build_distance_matrix_km",
    "build_travel_time_matrix",
    "calculate_performance_metrics",
    "extract_vehicle_routes",
    "get_routing_status_name",
    "haversine_distance_km",
    "haversine_distance_meters",
    "load_dataset_from_csv",
    "solve_cvrp",
    "solve_vrptw",
    "validate_vrptw_schedule",
]
