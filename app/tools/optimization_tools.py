"""CrewAI custom tools wrapping the Phase II Google OR-Tools solvers."""

from __future__ import annotations

from typing import Any, Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from app.optimization import OptimizationError, solve_cvrp, solve_vrptw
from app.schemas.logistics import RouteOptimizationReport
from app.schemas.order import CustomerOrder, Depot
from app.schemas.vehicle import Vehicle


class CvrpOptimizationInput(BaseModel):
    """Input schema for CvrpOptimizationTool."""

    depots: list[dict[str, Any]] = Field(description="Depot records.")
    vehicles: list[dict[str, Any]] = Field(description="Available fleet vehicle records.")
    customers: list[dict[str, Any]] = Field(description="Validated customer orders.")
    time_limit_seconds: int = Field(default=5, gt=0)
    use_guided_local_search: bool = False


class VrptwOptimizationInput(CvrpOptimizationInput):
    """Input schema for VrptwOptimizationTool."""

    default_horizon_minutes: int = Field(default=1440, gt=0)


def _run_ortools_solver(
    *,
    problem_type: Literal["CVRP", "VRPTW"],
    depots_raw: list[dict[str, Any]],
    vehicles_raw: list[dict[str, Any]],
    customers_raw: list[dict[str, Any]],
    time_limit_seconds: int,
    use_guided_local_search: bool,
    default_horizon_minutes: int = 1440,
) -> str:
    """Validate inputs and invoke the shared Phase II Google OR-Tools solver."""
    try:
        depots = [Depot.model_validate(d) for d in depots_raw]
        vehicles = [Vehicle.model_validate(v) for v in vehicles_raw]
        customers = [CustomerOrder.model_validate(c) for c in customers_raw]
    except ValidationError as exc:
        report = RouteOptimizationReport(
            status="blocked",
            problem_type=problem_type,
            optimization_result=None,
            error_type="ValidationError",
            error_message=str(exc),
            summary=f"Blocked {problem_type} optimization due to schema validation error.",
        )
        return report.model_dump_json()

    try:
        if problem_type == "CVRP":
            result = solve_cvrp(
                depots=depots,
                vehicles=vehicles,
                customers=customers,
                time_limit_seconds=time_limit_seconds,
                use_guided_local_search=use_guided_local_search,
            )
        else:
            result = solve_vrptw(
                depots=depots,
                vehicles=vehicles,
                customers=customers,
                default_horizon_minutes=default_horizon_minutes,
                time_limit_seconds=time_limit_seconds,
                use_guided_local_search=use_guided_local_search,
            )
    except OptimizationError as exc:
        report = RouteOptimizationReport(
            status="infeasible",
            problem_type=problem_type,
            optimization_result=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
            summary=(
                f"OR-Tools {problem_type} solver failed to find a feasible route plan: {exc}"
            ),
        )
        return report.model_dump_json()

    report = RouteOptimizationReport(
        status="feasible",
        problem_type=problem_type,
        optimization_result=result,
        summary=(
            f"OR-Tools {problem_type} solved ({result.solver_status}) using "
            f"{result.metrics.vehicles_used}/{result.metrics.total_vehicles_available} "
            f"vehicles, distance={result.metrics.total_distance_km} km, "
            f"deliveries={result.metrics.total_deliveries_completed}."
        ),
    )
    return report.model_dump_json()


class CvrpOptimizationTool(BaseTool):
    """Executes the Phase II Google OR-Tools CVRP solver."""

    name: str = "cvrp_optimization_tool"
    description: str = (
        "Invokes the Google OR-Tools CVRP optimization engine on validated depots, "
        "vehicles, and customer orders. Returns a structured RouteOptimizationReport JSON."
    )
    args_schema: type[BaseModel] = CvrpOptimizationInput

    def _run(
        self,
        depots: list[dict[str, Any]],
        vehicles: list[dict[str, Any]],
        customers: list[dict[str, Any]],
        time_limit_seconds: int = 5,
        use_guided_local_search: bool = False,
    ) -> str:
        """Run Google OR-Tools CVRP solver directly."""
        return _run_ortools_solver(
            problem_type="CVRP",
            depots_raw=depots,
            vehicles_raw=vehicles,
            customers_raw=customers,
            time_limit_seconds=time_limit_seconds,
            use_guided_local_search=use_guided_local_search,
        )


class VrptwOptimizationTool(BaseTool):
    """Executes the Phase II Google OR-Tools VRPTW solver."""

    name: str = "vrptw_optimization_tool"
    description: str = (
        "Invokes the Google OR-Tools VRPTW optimization engine on validated depots, "
        "vehicles, and time-windowed customer orders. Returns a structured RouteOptimizationReport JSON."
    )
    args_schema: type[BaseModel] = VrptwOptimizationInput

    def _run(
        self,
        depots: list[dict[str, Any]],
        vehicles: list[dict[str, Any]],
        customers: list[dict[str, Any]],
        time_limit_seconds: int = 5,
        default_horizon_minutes: int = 1440,
        use_guided_local_search: bool = False,
    ) -> str:
        """Run Google OR-Tools VRPTW solver directly."""
        return _run_ortools_solver(
            problem_type="VRPTW",
            depots_raw=depots,
            vehicles_raw=vehicles,
            customers_raw=customers,
            time_limit_seconds=time_limit_seconds,
            default_horizon_minutes=default_horizon_minutes,
            use_guided_local_search=use_guided_local_search,
        )

