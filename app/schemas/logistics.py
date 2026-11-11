"""Pydantic schemas for CrewAI multi-agent logistics orchestration (Phase III)."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.order import CustomerOrder, Depot
from app.schemas.route import OptimizationResult, PerformanceMetrics, VehicleRoute
from app.schemas.vehicle import Vehicle


class RejectedOrder(BaseModel):
    """Order rejected during pre-optimization validation."""

    customer_id: str = Field(default="<missing>")
    reason: str
    raw_data: dict[str, Any] = Field(default_factory=dict)


class ValidatedOrderMetadata(BaseModel):
    """Operational priority metadata derived for a validated customer order."""

    customer_id: str
    demand: int = Field(ge=0)
    has_time_window: bool
    priority: Literal["high", "normal"] = "normal"


class OrderValidationReport(BaseModel):
    """Structured output produced by the Order Analysis and Validation Agent."""

    valid_orders: list[CustomerOrder] = Field(default_factory=list)
    rejected_orders: list[RejectedOrder] = Field(default_factory=list)
    order_metadata: list[ValidatedOrderMetadata] = Field(default_factory=list)
    total_submitted: int = Field(ge=0)
    valid_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    total_valid_demand: int = Field(ge=0)
    has_time_windows: bool = False
    is_valid_for_optimization: bool = False
    summary: str


class VehicleDepotAssignment(BaseModel):
    """Availability and depot assignment summary for a fleet vehicle."""

    vehicle_id: str
    capacity: int = Field(gt=0)
    start_depot_id: str
    end_depot_id: str
    max_route_duration_minutes: int | None = None
    speed_kmh: float = Field(gt=0.0)
    available: bool = True
    unavailability_reason: str | None = None


class FleetCapacityReport(BaseModel):
    """Structured output produced by the Fleet and Capacity Management Agent."""

    depots: list[Depot] = Field(default_factory=list)
    available_vehicles: list[Vehicle] = Field(default_factory=list)
    unavailable_vehicles: list[VehicleDepotAssignment] = Field(default_factory=list)
    assignments: list[VehicleDepotAssignment] = Field(default_factory=list)
    depot_vehicle_counts: dict[str, int] = Field(default_factory=dict)
    total_fleet_capacity: int = Field(ge=0)
    total_required_demand: int = Field(ge=0)
    capacity_slack: int = 0
    max_single_vehicle_capacity: int = Field(ge=0)
    max_single_order_demand: int = Field(ge=0)
    capacity_sufficient: bool = False
    issues: list[str] = Field(default_factory=list)
    summary: str


class OptimizationPreparationPlan(BaseModel):
    """Scenario selection and OR-Tools solver preparation blueprint."""

    ready_for_optimization: bool
    recommended_scenario: Literal["CVRP", "VRPTW"] = "VRPTW"
    rationale: str
    active_depot_ids: list[str] = Field(default_factory=list)
    active_vehicle_ids: list[str] = Field(default_factory=list)
    valid_customer_count: int = Field(ge=0)
    total_demand: int = Field(ge=0)
    time_limit_seconds: int = Field(default=5, gt=0)
    default_horizon_minutes: int = Field(default=1440, gt=0)
    use_guided_local_search: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)


class RouteOptimizationReport(BaseModel):
    """Structured output backed strictly by Google OR-Tools solver results."""

    status: Literal["feasible", "infeasible", "blocked"]
    problem_type: Literal["CVRP", "VRPTW"]
    solver_engine: Literal["OR-Tools"] = "OR-Tools"
    optimization_result: OptimizationResult | None = None
    error_type: str | None = None
    error_message: str | None = None
    summary: str

    @model_validator(mode="after")
    def validate_solver_result_consistency(self) -> "RouteOptimizationReport":
        """Ensure feasible reports contain actual OR-Tools routes and non-feasible omit routes."""
        if self.status == "feasible" and self.optimization_result is None:
            raise ValueError(
                "Feasible RouteOptimizationReport must contain an optimization_result."
            )
        if self.status != "feasible" and self.optimization_result is not None:
            raise ValueError(
                "Infeasible or blocked RouteOptimizationReport must not carry routes."
            )
        return self


class TrafficDisruptionEvent(BaseModel):
    """Controlled simulated traffic disruption specification (always marked simulated)."""

    event_id: str = Field(default="SIM-TRAFFIC-001", min_length=1)
    is_simulated: bool = Field(default=True)
    data_source: Literal["simulated"] = "simulated"
    simulation_notice: str = Field(
        default="SIMULATED TRAFFIC DATA — controlled event for Phase III testing"
    )
    severity: Literal["none", "low", "moderate", "high", "critical"] = "moderate"
    description: str = "Simulated arterial congestion and temporary route slowdown"
    affected_vehicle_ids: list[str] = Field(default_factory=list)
    disabled_vehicle_ids: list[str] = Field(default_factory=list)
    affected_customer_ids: list[str] = Field(default_factory=list)
    completed_customer_ids: list[str] = Field(default_factory=list)
    extra_delay_minutes: int = Field(default=0, ge=0)
    speed_reduction_factor: float = Field(default=1.0, gt=0.0, le=1.0)
    elapsed_time_minutes: int = Field(default=0, ge=0)
    actual_distance_before_disruption_meters: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional observed distance (meters) travelled prior to disruption; "
            "left None when actual telemetry is unavailable."
        ),
    )

    @field_validator("is_simulated")
    @classmethod
    def require_simulated_flag(cls, value: bool) -> bool:
        """Guarantee Phase III traffic events are explicitly flagged as simulated."""
        if not value:
            raise ValueError(
                "Live traffic feeds are not enabled in Phase III; is_simulated must be True."
            )
        return value


class ImpactedVehicleSummary(BaseModel):
    """Impact assessment for a single vehicle under simulated traffic."""

    vehicle_id: str
    reason: str
    affected_remaining_customer_ids: list[str] = Field(default_factory=list)
    projected_additional_delay_minutes: int = Field(ge=0)
    projected_time_window_violations: int = Field(default=0, ge=0)
    disabled: bool = False


class CompletedDeliveryRecord(BaseModel):
    """Preserved completed delivery excluded from dynamic route replanning."""

    customer_id: str
    vehicle_id: str
    demand: int = Field(ge=0)
    arrival_time_minutes: int | None = None
    departure_time_minutes: int | None = None
    preserved_status: Literal["completed"] = "completed"


class DynamicReplanningReport(BaseModel):
    """Result of dynamic OR-Tools route replanning for uncompleted orders."""

    replanning_triggered: bool
    status: Literal["not_needed", "replanned", "infeasible", "blocked"]
    solver_engine: Literal["OR-Tools"] = "OR-Tools"
    is_simulated_disruption: bool = True
    preserved_completed_deliveries: list[CompletedDeliveryRecord] = Field(
        default_factory=list
    )
    remaining_customer_ids: list[str] = Field(default_factory=list)
    disabled_vehicle_ids: list[str] = Field(default_factory=list)
    active_remaining_vehicles: list[Vehicle] = Field(default_factory=list)
    original_vehicle_count: int | None = Field(default=None, ge=0)
    original_fleet_capacity: int | None = Field(default=None, ge=0)
    active_remaining_vehicle_count: int | None = Field(default=None, ge=0)
    active_remaining_fleet_capacity: int | None = Field(default=None, ge=0)
    completed_demand_before_replanning: int = Field(default=0, ge=0)
    remaining_demand_to_deliver: int = Field(default=0, ge=0)
    original_total_distance_meters: int | None = Field(default=None, ge=0)
    original_total_distance_km: float | None = Field(default=None, ge=0.0)
    distance_travelled_before_disruption_meters: int | None = Field(default=None, ge=0)
    distance_travelled_before_disruption_km: float | None = Field(default=None, ge=0.0)
    vehicle_earliest_start_minutes: dict[str, int] = Field(default_factory=dict)
    replanned_result: OptimizationResult | None = None
    error_type: str | None = None
    error_message: str | None = None
    summary: str


class TrafficExceptionReport(BaseModel):
    """Structured output produced by the Traffic and Delay Exception Monitoring Agent."""

    has_disruption: bool
    is_simulated: bool = True
    simulation_notice: str = Field(
        default="SIMULATED TRAFFIC DATA — controlled simulation for Phase III validation"
    )
    disruption_event: TrafficDisruptionEvent | None = None
    impacted_vehicles: list[ImpactedVehicleSummary] = Field(default_factory=list)
    replanning_recommended: bool = False
    recommendation_reason: str
    replanning_report: DynamicReplanningReport | None = None
    summary: str


class DispatchPlan(BaseModel):
    """Final dispatch decision produced by the Logistics Dispatch Coordinator Agent."""

    dispatch_status: Literal["approved", "blocked"]
    route_source: Literal["initial_ortools", "replanned_ortools", "none"]
    problem_type: Literal["CVRP", "VRPTW"]
    is_simulated_traffic_used: bool = False
    dispatched_routes: list[VehicleRoute] = Field(default_factory=list)
    preserved_completed_deliveries: list[CompletedDeliveryRecord] = Field(
        default_factory=list
    )
    all_serviced_customer_ids: list[str] = Field(default_factory=list)
    rejected_customer_ids: list[str] = Field(default_factory=list)
    unassigned_customer_ids: list[str] = Field(default_factory=list)
    solver_status: str
    blocking_reasons: list[str] = Field(default_factory=list)
    ortools_provenance_verified: bool = False
    summary: str


class FuelEstimateAssumptions(BaseModel):
    """Distance-based fuel consumption and cost estimate with explicit scope."""

    estimate_scope: Literal["complete_shift", "remaining_routes"] = "complete_shift"
    scope_description: str = (
        "Complete shift: calculated from all dispatched OR-Tools routes."
    )
    distance_basis_km: float = Field(default=0.0, ge=0.0)
    fuel_efficiency_km_per_liter: float = Field(gt=0.0)
    fuel_price_per_liter: float = Field(ge=0.0)
    estimated_fuel_liters: float = Field(ge=0.0)
    estimated_fuel_cost: float = Field(ge=0.0)


class FleetUtilizationComparison(BaseModel):
    """Separates original pre-disruption fleet capacity from active remaining fleet."""

    original_vehicle_count: int = Field(ge=0)
    original_fleet_capacity: int = Field(ge=0)
    original_vehicles_used: int | None = Field(default=None, ge=0)
    original_fleet_capacity_utilization: float | None = Field(default=None, ge=0.0)
    disabled_vehicle_ids: list[str] = Field(default_factory=list)
    active_remaining_vehicle_ids: list[str] = Field(default_factory=list)
    active_remaining_vehicle_count: int = Field(ge=0)
    active_remaining_fleet_capacity: int = Field(ge=0)
    active_remaining_vehicles_used: int = Field(ge=0)
    active_remaining_demand_delivered: int = Field(ge=0)
    active_remaining_fleet_utilization: float = Field(ge=0.0)
    active_vehicle_utilization_by_id: dict[str, float] = Field(default_factory=dict)


class DistanceBreakdownReport(BaseModel):
    """Distinguishes original, pre-disruption, remaining, and total shift distances."""

    original_total_route_distance_meters: int | None = Field(default=None, ge=0)
    original_total_route_distance_km: float | None = Field(default=None, ge=0.0)
    distance_travelled_before_disruption_meters: int | None = Field(
        default=None, ge=0
    )
    distance_travelled_before_disruption_km: float | None = Field(
        default=None, ge=0.0
    )
    remaining_optimized_route_distance_meters: int | None = Field(default=None, ge=0)
    remaining_optimized_route_distance_km: float | None = Field(
        default=None, ge=0.0
    )
    estimated_total_shift_distance_meters: int | None = Field(default=None, ge=0)
    estimated_total_shift_distance_km: float | None = Field(default=None, ge=0.0)
    total_shift_distance_status: Literal[
        "available_complete_shift",
        "available_with_observed_pre_disruption",
        "unavailable_missing_pre_disruption_distance",
        "unavailable_blocked",
    ]


class LogisticsPerformanceReport(BaseModel):
    """Structured KPI and fuel report produced by the Performance Reporting Agent."""

    report_status: Literal["completed", "blocked"]
    dispatch_status: Literal["approved", "blocked"]
    problem_type: Literal["CVRP", "VRPTW"]
    is_replanned_workflow: bool = False
    metrics: PerformanceMetrics | None = None
    completed_before_replanning_count: int = Field(default=0, ge=0)
    completed_demand_before_replanning: int = Field(default=0, ge=0)
    remaining_deliveries_count: int = Field(default=0, ge=0)
    remaining_demand_delivered: int = Field(default=0, ge=0)
    preserved_completed_deliveries: list[CompletedDeliveryRecord] = Field(
        default_factory=list
    )
    total_deliveries_accounted_for: int = Field(default=0, ge=0)
    completion_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    fleet_comparison: FleetUtilizationComparison | None = None
    distance_breakdown: DistanceBreakdownReport | None = None
    fuel_estimate: FuelEstimateAssumptions | None = None
    total_shift_fuel_estimate: FuelEstimateAssumptions | None = None
    notes: list[str] = Field(default_factory=list)
    summary: str


class OrchestrationRequest(BaseModel):
    """Input payload for running the full CrewAI multi-agent logistics pipeline."""

    depots: list[Depot]
    vehicles: list[Vehicle]
    orders: list[dict[str, Any]]
    scenario: Literal["auto", "CVRP", "VRPTW"] = "auto"
    unavailable_vehicle_ids: list[str] = Field(default_factory=list)
    simulated_disruption: TrafficDisruptionEvent | None = None
    time_limit_seconds: int = Field(default=5, gt=0)
    default_horizon_minutes: int = Field(default=1440, gt=0)
    use_guided_local_search: bool = False
    fuel_efficiency_km_per_liter: float | None = Field(default=None, gt=0.0)
    fuel_price_per_liter: float | None = Field(default=None, ge=0.0)


class OrchestrationResponse(BaseModel):
    """End-to-end result returned by SmartLogisticsCrew / LogisticsOrchestrationService."""

    execution_mode: Literal["mock", "gemini"]
    model_name: str
    order_validation: OrderValidationReport
    fleet_capacity: FleetCapacityReport
    optimization_preparation: OptimizationPreparationPlan
    route_optimization: RouteOptimizationReport
    traffic_exception: TrafficExceptionReport
    dispatch_plan: DispatchPlan
    performance_report: LogisticsPerformanceReport
    crew_raw_output: str = ""

