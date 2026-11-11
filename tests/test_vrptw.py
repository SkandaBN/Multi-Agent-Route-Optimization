"""Unit and integration tests for the VRPTW solver."""

import pytest

from app.optimization import (
    InfeasibleRoutingError,
    build_distance_matrix,
    build_travel_time_matrix,
    load_dataset_from_csv,
    solve_vrptw,
)
from app.schemas.order import CustomerOrder, Depot
from app.schemas.vehicle import Vehicle


def _single_depot_vrptw_fixture() -> tuple[list[Depot], list[Vehicle], list[CustomerOrder]]:
    depot = Depot(
        depot_id="DEPOT_1",
        name="Central",
        latitude=13.01,
        longitude=77.58,
        time_window_start=0,
        time_window_end=480,
    )
    vehicles = [
        Vehicle(
            vehicle_id="V1",
            capacity=60,
            start_depot_id="DEPOT_1",
            max_route_duration_minutes=240,
            speed_kmh=40.0,
        ),
        Vehicle(
            vehicle_id="V2",
            capacity=60,
            start_depot_id="DEPOT_1",
            max_route_duration_minutes=240,
            speed_kmh=40.0,
        ),
    ]
    customers = [
        CustomerOrder(
            customer_id="C1",
            latitude=13.03,
            longitude=77.60,
            demand=20,
            service_duration_minutes=10,
            time_window_start=15,
            time_window_end=90,
        ),
        CustomerOrder(
            customer_id="C2",
            latitude=13.04,
            longitude=77.56,
            demand=20,
            service_duration_minutes=15,
            time_window_start=40,
            time_window_end=150,
        ),
        CustomerOrder(
            customer_id="C3",
            latitude=12.98,
            longitude=77.61,
            demand=20,
            service_duration_minutes=10,
            time_window_start=60,
            time_window_end=180,
        ),
        CustomerOrder(
            customer_id="C4",
            latitude=12.96,
            longitude=77.57,
            demand=15,
            service_duration_minutes=10,
            time_window_start=100,
            time_window_end=210,
        ),
    ]
    return [depot], vehicles, customers


def test_single_depot_vrptw_schedule_and_windows() -> None:
    """Single-depot VRPTW schedules must satisfy time windows, causality, and max duration."""
    depots, vehicles, customers = _single_depot_vrptw_fixture()
    result = solve_vrptw(depots, vehicles, customers)

    assert result.problem_type == "VRPTW"
    assert result.solver_status in {"ROUTING_SUCCESS", "ROUTING_OPTIMAL"}
    assert result.metrics.time_window_violations == 0
    assert result.metrics.total_deliveries_completed == len(customers)

    nodes = [*depots, *customers]
    node_idx = {"DEPOT_1": 0, **{c.customer_id: i + 1 for i, c in enumerate(customers)}}
    dist_mat = build_distance_matrix(nodes)
    time_mat = build_travel_time_matrix(dist_mat, speed_kmh=40.0)
    vehicle_map = {v.vehicle_id: v for v in vehicles}

    visited: list[str] = []
    for route in result.routes:
        vehicle = vehicle_map[route.vehicle_id]
        assert route.arrival_times_minutes is not None
        assert route.total_route_duration_minutes is not None
        if vehicle.max_route_duration_minutes is not None:
            assert route.total_route_duration_minutes <= vehicle.max_route_duration_minutes

        prev_node = 0
        prev_departure = 0
        for stop in route.stops:
            assert stop.arrival_time_minutes is not None
            assert stop.departure_time_minutes is not None
            assert stop.time_window_start is not None
            assert stop.time_window_end is not None
            assert stop.time_window_violated is False
            assert stop.time_window_start <= stop.arrival_time_minutes <= stop.time_window_end

            cur_node = node_idx[stop.customer_id]
            assert stop.arrival_time_minutes >= prev_departure + time_mat[prev_node][cur_node]
            assert (
                stop.departure_time_minutes
                == stop.arrival_time_minutes + (stop.service_duration_minutes or 0)
            )
            assert route.arrival_times_minutes[stop.customer_id] == stop.arrival_time_minutes

            prev_node = cur_node
            prev_departure = stop.departure_time_minutes
            visited.append(stop.customer_id)

    assert set(visited) == {c.customer_id for c in customers}


def test_multi_depot_vrptw_on_synthetic_dataset() -> None:
    """Multi-depot VRPTW solves the 2-depot, 5-vehicle, 25-customer CSV dataset."""
    depots, vehicles, customers = load_dataset_from_csv("data")
    result = solve_vrptw(depots, vehicles, customers)

    assert result.problem_type == "VRPTW"
    assert result.metrics.total_deliveries_completed == 25
    assert result.metrics.time_window_violations == 0

    vehicle_map = {v.vehicle_id: v for v in vehicles}
    visited: list[str] = []
    for route in result.routes:
        vehicle = vehicle_map[route.vehicle_id]
        assert route.starting_depot_id == vehicle.start_depot_id
        assert route.ending_depot_id == vehicle.end_depot_id
        assert route.total_demand <= vehicle.capacity
        assert route.arrival_times_minutes is not None
        assert route.total_route_duration_minutes is not None
        if vehicle.max_route_duration_minutes is not None:
            assert route.total_route_duration_minutes <= vehicle.max_route_duration_minutes
        for stop in route.stops:
            assert stop.time_window_violated is False
        visited.extend(route.ordered_customer_ids)

    assert len(visited) == 25
    assert set(visited) == {c.customer_id for c in customers}


def test_vrptw_infeasible_time_window_raises() -> None:
    """Impossible customer time window (before travel from depot) must raise InfeasibleRoutingError."""
    depot = Depot(
        depot_id="DEPOT_1",
        latitude=13.0,
        longitude=77.5,
        time_window_start=0,
        time_window_end=480,
    )
    vehicle = Vehicle(
        vehicle_id="V1",
        capacity=50,
        start_depot_id="DEPOT_1",
        speed_kmh=40.0,
    )
    # ~111 km away (> 160 min travel at 40 km/h), yet time window ends at 10 min
    far_customer = CustomerOrder(
        customer_id="FAR_1",
        latitude=14.0,
        longitude=77.5,
        demand=10,
        service_duration_minutes=5,
        time_window_start=0,
        time_window_end=10,
    )
    with pytest.raises(InfeasibleRoutingError):
        solve_vrptw([depot], [vehicle], [far_customer])


def test_vrptw_infeasible_max_route_duration_raises() -> None:
    """Max route duration smaller than minimum round-trip + service must raise InfeasibleRoutingError."""
    depot = Depot(
        depot_id="DEPOT_1",
        latitude=13.0,
        longitude=77.5,
        time_window_start=0,
        time_window_end=480,
    )
    vehicle = Vehicle(
        vehicle_id="V1",
        capacity=50,
        start_depot_id="DEPOT_1",
        max_route_duration_minutes=15,
        speed_kmh=40.0,
    )
    # ~11 km away (~17 min each way + 10 min service = ~44 min > 15 min)
    customer = CustomerOrder(
        customer_id="C_DUR",
        latitude=13.1,
        longitude=77.5,
        demand=10,
        service_duration_minutes=10,
        time_window_start=0,
        time_window_end=200,
    )
    with pytest.raises(InfeasibleRoutingError):
        solve_vrptw([depot], [vehicle], [customer])

