"""Pydantic schemas for request, domain, and response payloads."""

from app.schemas.health import HealthResponse
from app.schemas.logistics import (
    CompletedDeliveryRecord,
    DispatchPlan,
    DistanceBreakdownReport,
    DynamicReplanningReport,
    FleetCapacityReport,
    FleetUtilizationComparison,
    FuelEstimateAssumptions,
    ImpactedVehicleSummary,
    LogisticsPerformanceReport,
    OptimizationPreparationPlan,
    OrchestrationRequest,
    OrchestrationResponse,
    OrderValidationReport,
    RejectedOrder,
    RouteOptimizationReport,
    TrafficDisruptionEvent,
    TrafficExceptionReport,
    ValidatedOrderMetadata,
    VehicleDepotAssignment,
)
from app.schemas.order import CustomerOrder, Depot, GeoPoint, Order
from app.schemas.route import OptimizationResult, PerformanceMetrics, RouteStop, VehicleRoute
from app.schemas.vehicle import Vehicle

__all__ = [
    "CompletedDeliveryRecord",
    "CustomerOrder",
    "Depot",
    "DispatchPlan",
    "DistanceBreakdownReport",
    "DynamicReplanningReport",
    "FleetCapacityReport",
    "FleetUtilizationComparison",
    "FuelEstimateAssumptions",
    "GeoPoint",
    "HealthResponse",
    "ImpactedVehicleSummary",
    "LogisticsPerformanceReport",
    "OptimizationPreparationPlan",
    "OptimizationResult",
    "OrchestrationRequest",
    "OrchestrationResponse",
    "Order",
    "OrderValidationReport",
    "PerformanceMetrics",
    "RejectedOrder",
    "RouteOptimizationReport",
    "RouteStop",
    "TrafficDisruptionEvent",
    "TrafficExceptionReport",
    "ValidatedOrderMetadata",
    "Vehicle",
    "VehicleDepotAssignment",
    "VehicleRoute",
]
