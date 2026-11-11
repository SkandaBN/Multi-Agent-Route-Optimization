"""Pydantic schemas for request, domain, and response payloads."""

from app.schemas.health import HealthResponse
from app.schemas.order import CustomerOrder, Depot, GeoPoint, Order
from app.schemas.route import OptimizationResult, PerformanceMetrics, RouteStop, VehicleRoute
from app.schemas.vehicle import Vehicle

__all__ = [
    "CustomerOrder",
    "Depot",
    "GeoPoint",
    "HealthResponse",
    "OptimizationResult",
    "Order",
    "PerformanceMetrics",
    "RouteStop",
    "Vehicle",
    "VehicleRoute",
]
