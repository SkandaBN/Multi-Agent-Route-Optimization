"""Unit and integration tests for the CVRP solver (single-depot and multi-depot)."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.optimization import (
    InfeasibleRoutingError,
    InsufficientCapacityError,
    InvalidRoutingInputError,
    build_distance_matrix,
    load_dataset_from_csv,
    solve_cvrp,
)
from app.schemas.order import CustomerOrder, Depot
from app.schemas.vehicle import Vehicle


def _single_depot_fixture() -> tuple[list[Depot], list[Vehicle], list[CustomerOrder]]:
    depot = Depot(depot_id="DEPOT_1", name="Central", latitude=13.01, longitude=77.58)
    vehicles = [
        Vehicle(vehicle_id="V1", capacity=50, start_depot_id="DEPOT_1"),
        Vehicle(vehicle_id="V2", capacity=40, start_depot_id="DEPOT_1"),
        Vehicle(vehicle_id="V3", capacity=40, start_depot_id="DEPOT_1"),
    ]
    customers = [
        CustomerOrder(customer_id="C1", latitude=13.03, longitude=77.60, demand=20),
        CustomerOrder(customer_id="C2", latitude=13.04, longitude=77.56, demand=25),
        CustomerOrder(customer_id="C3", latitude=12.98, longitude=77.61, demand=15),
        CustomerOrder(customer_id="C4", latitude=12.96, longitude=77.57, demand=20),
        CustomerOrder(customer_id="C5", latitude=13.00, longitude=77.54, demand=18),
        CustomerOrder(customer_id="C6", latitude=13.02, longitude=77.63, demand=12),
    ]
    return [depot], vehicles, customers


def test_single_depot_cvrp_valid_routes_and_constraints() -> None:
    """Single-depot CVRP must assign every customer once and obey all constraints."""
    depots, vehicles, customers = _single_depot_fixture()
    result = solve_cvrp(depots, vehicles, customers)

    assert result.problem_type == "CVRP"
    assert result.solver_status in {"ROUTING_SUCCESS", "ROUTING_OPTIMAL"}
    assert result.metrics.time_window_violations is None

    visited: list[str] = []
    vehicle_caps = {v.vehicle_id: v.capacity for v in vehicles}
    customer_demands = {c.customer_id: c.demand for c in customers}

    # Precompute reference matrix to verify extracted distance exactly
    nodes = [*depots, *customers]
    node_idx = {"DEPOT_1": 0, **{c.customer_id: i + 1 for i, c in enumerate(customers)}}
    dist_mat = build_distance_matrix(nodes)

    total_distance = 0
    for route in result.routes:
        assert route.starting_depot_id == "DEPOT_1"
        assert route.ending_depot_id == "DEPOT_1"
        assert route.arrival_times_minutes is None
        assert len(route.ordered_customer_ids) > 0
        assert route.total_demand <= vehicle_caps[route.vehicle_id]
        assert route.total_demand == sum(
            customer_demands[cid] for cid in route.ordered_customer_ids
        )

        # Verify arc-by-arc distance equals route.total_distance_meters
        seq = [node_idx["DEPOT_1"], *(node_idx[cid] for cid in route.ordered_customer_ids), node_idx["DEPOT_1"]]
        expected_dist = sum(dist_mat[seq[i]][seq[i + 1]] for i in range(len(seq) - 1))
        assert route.total_distance_meters == expected_dist
        total_distance += expected_dist

        visited.extend(route.ordered_customer_ids)

    expected_ids = {c.customer_id for c in customers}
    assert len(visited) == len(customers)
    assert set(visited) == expected_ids
    assert result.metrics.total_distance_meters == total_distance
    assert result.metrics.total_deliveries_completed == len(customers)


def test_multi_depot_cvrp_on_synthetic_dataset() -> None:
    """Multi-depot CVRP solves the 2-depot, 5-vehicle, 25-customer CSV dataset."""
    depots, vehicles, customers = load_dataset_from_csv("data")
    assert len(depots) == 2
    assert len(vehicles) == 5
    assert len(customers) == 25

    result = solve_cvrp(depots, vehicles, customers)

    vehicle_map = {v.vehicle_id: v for v in vehicles}
    visited: list[str] = []

    for route in result.routes:
        vehicle = vehicle_map[route.vehicle_id]
        assert route.starting_depot_id == vehicle.start_depot_id
        assert route.ending_depot_id == vehicle.end_depot_id
        assert route.total_demand <= vehicle.capacity
        assert route.total_distance_meters > 0
        visited.extend(route.ordered_customer_ids)

    assert len(visited) == 25
    assert set(visited) == {c.customer_id for c in customers}
    assert result.metrics.total_deliveries_completed == 25
    assert result.metrics.time_window_violations is None


def test_cvrp_error_handling_validation_and_capacity() -> None:
    """CVRP must reject empty customers, negative demands, invalid coords, and insufficient capacity."""
    depots, vehicles, customers = _single_depot_fixture()

    with pytest.raises(InvalidRoutingInputError):
        solve_cvrp(depots, vehicles, [])

    with pytest.raises(ValidationError):
        CustomerOrder(customer_id="BAD_DEMAND", latitude=13.0, longitude=77.5, demand=-5)

    with pytest.raises(ValidationError):
        CustomerOrder(customer_id="BAD_COORD", latitude=95.0, longitude=77.5, demand=5)

    # Total capacity insufficient (130 < 150)
    heavy_customers = [
        CustomerOrder(customer_id=f"H{i}", latitude=13.0 + 0.01 * i, longitude=77.5, demand=30)
        for i in range(5)
    ]
    with pytest.raises(InsufficientCapacityError):
        solve_cvrp(depots, vehicles, heavy_customers)

    # Single customer exceeds largest single vehicle capacity (60 > 50)
    oversized = [CustomerOrder(customer_id="BIG", latitude=13.02, longitude=77.59, demand=60)]
    with pytest.raises(InsufficientCapacityError):
        solve_cvrp(depots, vehicles, oversized)

    # Unknown depot on vehicle
    bad_vehicle = [Vehicle(vehicle_id="V_BAD", capacity=50, start_depot_id="NO_SUCH_DEPOT")]
    with pytest.raises(InvalidRoutingInputError):
        solve_cvrp(depots, bad_vehicle, customers[:1])


def test_cvrp_raises_when_ortools_returns_no_solution() -> None:
    """If OR-Tools fails to return a solution, raise InfeasibleRoutingError without fallback routes."""
    depots, vehicles, customers = _single_depot_fixture()
    with patch(
        "ortools.constraint_solver.pywrapcp.RoutingModel.SolveWithParameters",
        return_value=None,
    ):
        with pytest.raises(InfeasibleRoutingError):
            solve_cvrp(depots, vehicles, customers)

