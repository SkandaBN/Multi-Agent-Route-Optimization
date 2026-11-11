"""Pydantic schemas for extracted vehicle routes and optimization performance metrics."""

from typing import Literal

from pydantic import BaseModel, Field


class RouteStop(BaseModel):
    """Single customer delivery stop along a vehicle route."""

    customer_id: str
    demand: int = Field(..., ge=0)
    arrival_time_minutes: int | None = None
    service_duration_minutes: int | None = None
    departure_time_minutes: int | None = None
    time_window_start: int | None = None
    time_window_end: int | None = None
    time_window_violated: bool | None = None


class VehicleRoute(BaseModel):
    """Structured route assigned to a single vehicle."""

    vehicle_id: str
    vehicle_capacity: int = Field(..., gt=0)
    starting_depot_id: str
    ending_depot_id: str
    ordered_customer_ids: list[str]
    stops: list[RouteStop]
    total_distance_meters: int = Field(..., ge=0)
    total_distance_km: float = Field(..., ge=0.0)
    estimated_travel_time_minutes: int = Field(..., ge=0)
    total_route_duration_minutes: int | None = None
    arrival_times_minutes: dict[str, int] | None = None
    total_demand: int = Field(..., ge=0)
    capacity_utilization: float = Field(..., ge=0.0)


class PerformanceMetrics(BaseModel):
    """Aggregate performance metrics across the optimized routing solution."""

    total_distance_meters: int = Field(..., ge=0)
    total_distance_km: float = Field(..., ge=0.0)
    total_estimated_travel_time_minutes: int = Field(..., ge=0)
    vehicles_used: int = Field(..., ge=0)
    total_vehicles_available: int = Field(..., ge=0)
    vehicle_utilization_by_id: dict[str, float]
    average_used_vehicle_utilization: float = Field(..., ge=0.0)
    fleet_capacity_utilization: float = Field(..., ge=0.0)
    total_deliveries_completed: int = Field(..., ge=0)
    total_demand_delivered: int = Field(..., ge=0)
    time_window_violations: int | None = None


class OptimizationResult(BaseModel):
    """Complete structured output from a CVRP or VRPTW optimization run."""

    problem_type: Literal["CVRP", "VRPTW"]
    solver_status: str
    routes: list[VehicleRoute]
    metrics: PerformanceMetrics
    unassigned_customer_ids: list[str] = Field(default_factory=list)

