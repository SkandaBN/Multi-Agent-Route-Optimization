"""Unit tests for route extraction metrics and time-window violation checks."""

import math

import pytest

from app.optimization import (
    calculate_performance_metrics,
    load_dataset_from_csv,
    solve_cvrp,
    solve_vrptw,
    validate_vrptw_schedule,
)
from app.schemas.route import RouteStop, VehicleRoute
from app.schemas.vehicle import Vehicle


def test_metrics_calculation_without_time_window_evaluation() -> None:
    """CVRP metrics must compute utilization accurately and keep time_window_violations=None."""
    vehicles = [
        Vehicle(vehicle_id="V1", capacity=100, start_depot_id="D1"),
        Vehicle(vehicle_id="V2", capacity=50, start_depot_id="D1"),
        Vehicle(vehicle_id="V3", capacity=50, start_depot_id="D1"),
    ]
    routes = [
        VehicleRoute(
            vehicle_id="V1",
            vehicle_capacity=100,
            starting_depot_id="D1",
            ending_depot_id="D1",
            ordered_customer_ids=["C1", "C2"],
            stops=[
                RouteStop(customer_id="C1", demand=40),
                RouteStop(customer_id="C2", demand=35),
            ],
            total_distance_meters=12500,
            total_distance_km=12.5,
            estimated_travel_time_minutes=20,
            total_demand=75,
            capacity_utilization=0.75,
        ),
        VehicleRoute(
            vehicle_id="V2",
            vehicle_capacity=50,
            starting_depot_id="D1",
            ending_depot_id="D1",
            ordered_customer_ids=["C3"],
            stops=[RouteStop(customer_id="C3", demand=25)],
            total_distance_meters=5000,
            total_distance_km=5.0,
            estimated_travel_time_minutes=8,
            total_demand=25,
            capacity_utilization=0.5,
        ),
    ]

    metrics = calculate_performance_metrics(routes, vehicles, evaluate_time_windows=False)

    assert metrics.total_distance_meters == 17500
    assert math.isclose(metrics.total_distance_km, 17.5)
    assert metrics.total_estimated_travel_time_minutes == 28
    assert metrics.vehicles_used == 2
    assert metrics.total_vehicles_available == 3
    assert metrics.vehicle_utilization_by_id == {"V1": 0.75, "V2": 0.5, "V3": 0.0}
    assert math.isclose(metrics.average_used_vehicle_utilization, 0.625)
    # 100 delivered / 200 total fleet capacity = 0.5
    assert math.isclose(metrics.fleet_capacity_utilization, 0.5)
    assert metrics.total_deliveries_completed == 3
    assert metrics.total_demand_delivered == 100
    assert metrics.time_window_violations is None


def test_metrics_counts_actual_time_window_violations() -> None:
    """VRPTW violation validation must count both early and late arrival violations."""
    vehicles = [Vehicle(vehicle_id="V1", capacity=100, start_depot_id="D1")]
    routes = [
        VehicleRoute(
            vehicle_id="V1",
            vehicle_capacity=100,
            starting_depot_id="D1",
            ending_depot_id="D1",
            ordered_customer_ids=["C1", "C2", "C3"],
            stops=[
                RouteStop(
                    customer_id="C1",
                    demand=20,
                    arrival_time_minutes=10,  # early (< 20)
                    time_window_start=20,
                    time_window_end=60,
                    time_window_violated=True,
                ),
                RouteStop(
                    customer_id="C2",
                    demand=20,
                    arrival_time_minutes=70,  # valid
                    time_window_start=50,
                    time_window_end=100,
                    time_window_violated=False,
                ),
                RouteStop(
                    customer_id="C3",
                    demand=20,
                    arrival_time_minutes=130,  # late (> 120)
                    time_window_start=80,
                    time_window_end=120,
                    time_window_violated=True,
                ),
            ],
            total_distance_meters=10000,
            total_distance_km=10.0,
            estimated_travel_time_minutes=15,
            arrival_times_minutes={"C1": 10, "C2": 70, "C3": 130},
            total_demand=60,
            capacity_utilization=0.6,
        )
    ]

    assert validate_vrptw_schedule(routes) == 2
    metrics = calculate_performance_metrics(routes, vehicles, evaluate_time_windows=True)
    assert metrics.time_window_violations == 2


def test_metrics_refuses_uncalculated_vrptw_schedule() -> None:
    """Evaluating time windows on stops without calculated arrival times must raise ValueError."""
    vehicles = [Vehicle(vehicle_id="V1", capacity=100, start_depot_id="D1")]
    routes = [
        VehicleRoute(
            vehicle_id="V1",
            vehicle_capacity=100,
            starting_depot_id="D1",
            ending_depot_id="D1",
            ordered_customer_ids=["C1"],
            stops=[RouteStop(customer_id="C1", demand=10)],
            total_distance_meters=1000,
            total_distance_km=1.0,
            estimated_travel_time_minutes=2,
            total_demand=10,
            capacity_utilization=0.1,
        )
    ]
    with pytest.raises(ValueError):
        calculate_performance_metrics(routes, vehicles, evaluate_time_windows=True)

    with pytest.raises(ValueError):
        calculate_performance_metrics(routes, [], evaluate_time_windows=False)


def test_end_to_end_metrics_consistency_on_sample_dataset() -> None:
    """Verify aggregate metrics stay consistent across CVRP and VRPTW on sample CSV data."""
    depots, vehicles, customers = load_dataset_from_csv("data")
    total_demand = sum(c.demand for c in customers)

    cvrp_res = solve_cvrp(depots, vehicles, customers)
    vrptw_res = solve_vrptw(depots, vehicles, customers)

    assert cvrp_res.metrics.total_demand_delivered == total_demand
    assert vrptw_res.metrics.total_demand_delivered == total_demand
    assert cvrp_res.metrics.time_window_violations is None
    assert vrptw_res.metrics.time_window_violations == 0
    assert vrptw_res.metrics.total_distance_meters >= cvrp_res.metrics.total_distance_meters

