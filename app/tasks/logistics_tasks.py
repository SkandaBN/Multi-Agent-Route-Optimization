"""Definitions of the seven sequential CrewAI logistics tasks and safety guardrails."""

from dataclasses import dataclass
from typing import Any, Tuple

from crewai import Task
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.guardrail_types import GuardrailCallable
from pydantic import ValidationError

from app.agents.logistics_agents import LogisticsAgentSuite
from app.schemas.logistics import (
    DispatchPlan,
    FleetCapacityReport,
    LogisticsPerformanceReport,
    OptimizationPreparationPlan,
    OrderValidationReport,
    RouteOptimizationReport,
    TrafficExceptionReport,
)
from app.tools import verify_dispatch_routes_match_ortools


@dataclass(frozen=True)
class LogisticsTaskSuite:
    """Ordered suite of the seven Phase III logistics tasks."""

    order_validation: Task
    fleet_capacity: Task
    optimization_prep: Task
    route_optimization: Task
    traffic_exception: Task
    dispatch_coordination: Task
    performance_reporting: Task

    def as_list(self) -> list[Task]:
        """Return tasks in sequential CrewAI execution order."""
        return [
            self.order_validation,
            self.fleet_capacity,
            self.optimization_prep,
            self.route_optimization,
            self.traffic_exception,
            self.dispatch_coordination,
            self.performance_reporting,
        ]


def _extract_model(task: Task, model_cls: type[Any]) -> Any:
    """Safely extract or parse a Pydantic model from an upstream CrewAI Task."""
    task_output = getattr(task, "output", None)
    if task_output is None:
        return None
    if getattr(task_output, "pydantic", None) is not None and isinstance(
        task_output.pydantic, model_cls
    ):
        return task_output.pydantic
    raw = getattr(task_output, "raw", "")
    if not raw:
        return None
    try:
        return model_cls.model_validate_json(raw)
    except ValidationError:
        return None


def build_dispatch_verification_guardrail(
    route_optimization_task: Task,
    traffic_exception_task: Task,
) -> GuardrailCallable:
    """Create a CrewAI Task guardrail enforcing strict OR-Tools route provenance."""

    def _verify_dispatch_plan(output: TaskOutput) -> Tuple[bool, Any]:
        try:
            if output.pydantic is not None and isinstance(output.pydantic, DispatchPlan):
                plan = output.pydantic
            else:
                plan = DispatchPlan.model_validate_json(output.raw)
        except ValidationError as exc:
            return (False, f"DispatchPlan JSON failed schema validation: {exc}")

        route_report: RouteOptimizationReport | None = _extract_model(
            route_optimization_task, RouteOptimizationReport
        )
        traffic_report: TrafficExceptionReport | None = _extract_model(
            traffic_exception_task, TrafficExceptionReport
        )

        replanning = (
            traffic_report.replanning_report
            if traffic_report is not None and traffic_report.replanning_report is not None
            else None
        )

        if replanning is not None and replanning.replanning_triggered:
            if replanning.status != "replanned":
                if plan.dispatch_status == "approved" or plan.dispatched_routes:
                    return (
                        False,
                        "Guardrail blocked dispatch: OR-Tools dynamic replanning did not succeed; "
                        "cannot approve routes or invent fallback routes.",
                    )
                return (True, plan.model_dump_json())

            expected_result = replanning.replanned_result
            if plan.dispatch_status != "approved":
                return (
                    False,
                    "Guardrail error: dynamic OR-Tools replanning succeeded, dispatch should not discard valid plan.",
                )
            if plan.route_source != "replanned_ortools":
                return (
                    False,
                    "Guardrail blocked dispatch: route_source must be 'replanned_ortools' after replanning.",
                )
            ok, errors = verify_dispatch_routes_match_ortools(
                plan.dispatched_routes,
                expected_result,
                expected_problem_type=plan.problem_type,
            )
            if not ok:
                return (
                    False,
                    f"Guardrail blocked unverified replanned routes: {'; '.join(errors)}",
                )
            expected_completed = [
                c.model_dump() for c in replanning.preserved_completed_deliveries
            ]
            actual_completed = [
                c.model_dump() for c in plan.preserved_completed_deliveries
            ]
            if actual_completed != expected_completed:
                return (
                    False,
                    "Guardrail blocked dispatch: preserved_completed_deliveries do not match "
                    "recorded completed stops.",
                )
            return (True, plan.model_dump_json())

        if (
            route_report is None
            or route_report.status != "feasible"
            or route_report.optimization_result is None
        ):
            if plan.dispatch_status == "approved" or plan.dispatched_routes:
                return (
                    False,
                    "Guardrail blocked dispatch: no feasible Google OR-Tools solution exists; "
                    "cannot approve or fabricate routes.",
                )
            return (True, plan.model_dump_json())

        if plan.dispatch_status == "approved":
            ok, errors = verify_dispatch_routes_match_ortools(
                plan.dispatched_routes,
                route_report.optimization_result,
                expected_problem_type=plan.problem_type,
            )
            if not ok:
                return (
                    False,
                    f"Guardrail blocked fabricated or modified dispatch routes: {'; '.join(errors)}",
                )

        return (True, plan.model_dump_json())

    return _verify_dispatch_plan


def create_logistics_tasks(agents: LogisticsAgentSuite) -> LogisticsTaskSuite:
    """Create all seven sequential CrewAI tasks with explicit context dependencies."""
    order_validation_task = Task(
        name="order_validation_task",
        description=(
            "[TASK_1_ORDER_VALIDATION] Analyze and validate the incoming delivery orders "
            "using `order_validation_tool`. Reject invalid or duplicate orders safely.\n"
            "Orders JSON: {orders_json}"
        ),
        expected_output=(
            "A valid JSON string conforming strictly to the OrderValidationReport schema."
        ),
        agent=agents.order_validator,
        output_pydantic=OrderValidationReport,
    )

    fleet_capacity_task = Task(
        name="fleet_capacity_task",
        description=(
            "[TASK_2_FLEET_CAPACITY] Evaluate fleet vehicle availability, depot assignments, "
            "and total capacity sufficiency against the validated orders from Task 1 using "
            "`fleet_capacity_analysis_tool`.\n"
            "Depots JSON: {depots_json}\n"
            "Vehicles JSON: {vehicles_json}\n"
            "Unavailable Vehicle IDs JSON: {unavailable_vehicle_ids_json}"
        ),
        expected_output=(
            "A valid JSON string conforming strictly to the FleetCapacityReport schema."
        ),
        agent=agents.fleet_manager,
        context=[order_validation_task],
        output_pydantic=FleetCapacityReport,
    )

    optimization_prep_task = Task(
        name="optimization_prep_task",
        description=(
            "[TASK_3_OPTIMIZATION_PREP] Review Task 1 (OrderValidationReport) and Task 2 "
            "(FleetCapacityReport) to determine optimization readiness, active depots/vehicles, "
            "and recommended scenario (`VRPTW` or `CVRP`). Do not invoke route optimization yet.\n"
            "Requested Scenario: {scenario}\n"
            "Time Limit Seconds: {time_limit_seconds}\n"
            "Default Horizon Minutes: {default_horizon_minutes}\n"
            "Use Guided Local Search: {use_guided_local_search}"
        ),
        expected_output=(
            "A valid JSON string conforming strictly to the OptimizationPreparationPlan schema."
        ),
        agent=agents.route_optimizer,
        context=[order_validation_task, fleet_capacity_task],
        tools=[],
        output_pydantic=OptimizationPreparationPlan,
    )

    route_optimization_task = Task(
        name="route_optimization_task",
        description=(
            "[TASK_4_ROUTE_OPTIMIZATION] Execute Google OR-Tools route optimization according "
            "to Task 3 (`OptimizationPreparationPlan`). If `ready_for_optimization` is true, "
            "invoke `vrptw_optimization_tool` (for VRPTW) or `cvrp_optimization_tool` (for CVRP) "
            "using the validated orders, depots, and available vehicles. If blocked, return "
            "RouteOptimizationReport with status='blocked' and optimization_result=null. NEVER "
            "invent or alter delivery routes."
        ),
        expected_output=(
            "A valid JSON string conforming strictly to the RouteOptimizationReport schema."
        ),
        agent=agents.route_optimizer,
        context=[order_validation_task, fleet_capacity_task, optimization_prep_task],
        output_pydantic=RouteOptimizationReport,
    )

    traffic_exception_task = Task(
        name="traffic_exception_task",
        description=(
            "[TASK_5_TRAFFIC_EXCEPTION] Evaluate simulated traffic conditions against the "
            "initial OR-Tools routes from Task 4. Clearly mark all traffic telemetry as "
            "simulated. When a disruption recommends replanning, invoke "
            "`dynamic_route_replanning_tool` to preserve completed deliveries and re-optimize "
            "remaining deliveries with OR-Tools.\n"
            "Simulated Disruption JSON: {simulated_disruption_json}"
        ),
        expected_output=(
            "A valid JSON string conforming strictly to the TrafficExceptionReport schema."
        ),
        agent=agents.traffic_monitor,
        context=[order_validation_task, fleet_capacity_task, route_optimization_task],
        output_pydantic=TrafficExceptionReport,
    )

    dispatch_guardrail = build_dispatch_verification_guardrail(
        route_optimization_task=route_optimization_task,
        traffic_exception_task=traffic_exception_task,
    )

    dispatch_coordination_task = Task(
        name="dispatch_coordination_task",
        description=(
            "[TASK_6_DISPATCH_COORDINATION] Consolidate outputs from order validation, fleet "
            "analysis, OR-Tools optimization, and simulated traffic monitoring into the final "
            "DispatchPlan. Approve dispatch ONLY using exact OR-Tools routes (replanned if "
            "triggered and feasible, otherwise initial feasible routes), and safely block "
            "dispatch when infeasible or blocked."
        ),
        expected_output=(
            "A valid JSON string conforming strictly to the DispatchPlan schema."
        ),
        agent=agents.dispatch_coordinator,
        context=[
            order_validation_task,
            fleet_capacity_task,
            optimization_prep_task,
            route_optimization_task,
            traffic_exception_task,
        ],
        output_pydantic=DispatchPlan,
        guardrail=dispatch_guardrail,
        guardrail_max_retries=2,
    )

    performance_reporting_task = Task(
        name="performance_reporting_task",
        description=(
            "[TASK_7_PERFORMANCE_REPORT] Use `logistics_performance_analysis_tool` on the "
            "final DispatchPlan from Task 6 (together with original optimization, original vs. "
            "active remaining vehicles, disabled vehicle IDs, and any observed pre-disruption "
            "distance from Tasks 2, 4, and 5) to compute verified active vs. original fleet "
            "utilization, separated completed vs. remaining deliveries, distance breakdown, "
            "time-window violations (when evaluated), and explicitly scoped fuel estimates.\n"
            "Fuel Efficiency (km/L): {fuel_efficiency_km_per_liter}\n"
            "Fuel Price ($/L): {fuel_price_per_liter}"
        ),
        expected_output=(
            "A valid JSON string conforming strictly to the LogisticsPerformanceReport schema."
        ),
        agent=agents.performance_reporter,
        context=[
            order_validation_task,
            fleet_capacity_task,
            route_optimization_task,
            traffic_exception_task,
            dispatch_coordination_task,
        ],
        output_pydantic=LogisticsPerformanceReport,
    )

    return LogisticsTaskSuite(
        order_validation=order_validation_task,
        fleet_capacity=fleet_capacity_task,
        optimization_prep=optimization_prep_task,
        route_optimization=route_optimization_task,
        traffic_exception=traffic_exception_task,
        dispatch_coordination=dispatch_coordination_task,
        performance_reporting=performance_reporting_task,
    )
