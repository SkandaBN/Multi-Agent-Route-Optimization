"""Reusable CrewAI custom tools connecting agents to the OR-Tools engine."""

from app.tools.optimization_tools import (
    CvrpOptimizationInput,
    CvrpOptimizationTool,
    VrptwOptimizationInput,
    VrptwOptimizationTool,
)
from app.tools.provenance import (
    validate_route_constraints,
    verify_dispatch_routes_match_ortools,
)
from app.tools.reporting_tools import (
    LogisticsPerformanceAnalysisInput,
    LogisticsPerformanceAnalysisTool,
)
from app.tools.traffic_tools import (
    DynamicReplanningInput,
    DynamicRouteReplanningTool,
    TrafficDisruptionSimulationInput,
    TrafficDisruptionSimulationTool,
)
from app.tools.validation_tools import (
    FleetCapacityAnalysisInput,
    FleetCapacityAnalysisTool,
    OrderValidationInput,
    OrderValidationTool,
)


def get_all_logistics_tools() -> list[object]:
    """Instantiate all seven Phase III CrewAI logistics tools."""
    return [
        OrderValidationTool(),
        FleetCapacityAnalysisTool(),
        CvrpOptimizationTool(),
        VrptwOptimizationTool(),
        TrafficDisruptionSimulationTool(),
        DynamicRouteReplanningTool(),
        LogisticsPerformanceAnalysisTool(),
    ]


__all__ = [
    "CvrpOptimizationInput",
    "CvrpOptimizationTool",
    "DynamicReplanningInput",
    "DynamicRouteReplanningTool",
    "FleetCapacityAnalysisInput",
    "FleetCapacityAnalysisTool",
    "LogisticsPerformanceAnalysisInput",
    "LogisticsPerformanceAnalysisTool",
    "OrderValidationInput",
    "OrderValidationTool",
    "TrafficDisruptionSimulationInput",
    "TrafficDisruptionSimulationTool",
    "VrptwOptimizationInput",
    "VrptwOptimizationTool",
    "get_all_logistics_tools",
    "validate_route_constraints",
    "verify_dispatch_routes_match_ortools",
]
