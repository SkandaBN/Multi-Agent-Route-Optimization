"""Unit tests for the seven sequential CrewAI tasks and dispatch guardrail."""

from crewai.tasks.task_output import TaskOutput

from app.agents import create_all_logistics_agents
from app.core.llm import MockLogisticsLLM
from app.optimization import load_dataset_from_csv, solve_vrptw
from app.schemas.logistics import DispatchPlan, RouteOptimizationReport, TrafficExceptionReport
from app.tasks import build_dispatch_verification_guardrail, create_logistics_tasks


def test_create_logistics_tasks_builds_ordered_dag_for_six_agents() -> None:
    """All seven tasks are ordered sequentially with proper context and output models."""
    agents = create_all_logistics_agents(MockLogisticsLLM())
    tasks = create_logistics_tasks(agents)

    ordered = tasks.as_list()
    assert len(ordered) == 7
    assert [t.name for t in ordered] == [
        "order_validation_task",
        "fleet_capacity_task",
        "optimization_prep_task",
        "route_optimization_task",
        "traffic_exception_task",
        "dispatch_coordination_task",
        "performance_reporting_task",
    ]
    assert ordered[0].agent is agents.order_validator
    assert ordered[1].agent is agents.fleet_manager
    assert ordered[2].agent is agents.route_optimizer
    assert ordered[3].agent is agents.route_optimizer
    assert ordered[4].agent is agents.traffic_monitor
    assert ordered[5].agent is agents.dispatch_coordinator
    assert ordered[6].agent is agents.performance_reporter
    assert ordered[5].guardrail is not None


def test_dispatch_verification_guardrail_blocks_fabricated_and_infeasible_routes() -> None:
    """Dispatch guardrail approves exact OR-Tools routes and rejects tampered/infeasible plans."""
    depots, vehicles, customers = load_dataset_from_csv("data")
    opt_result = solve_vrptw(depots, vehicles, customers)

    agents = create_all_logistics_agents(MockLogisticsLLM())
    tasks = create_logistics_tasks(agents)

    tasks.route_optimization.output = TaskOutput(
        description="route opt",
        agent=agents.route_optimizer.role,
        raw=RouteOptimizationReport(
            status="feasible",
            problem_type="VRPTW",
            optimization_result=opt_result,
            summary="Solved",
        ).model_dump_json(),
    )
    tasks.traffic_exception.output = TaskOutput(
        description="traffic",
        agent=agents.traffic_monitor.role,
        raw=TrafficExceptionReport(
            has_disruption=False,
            is_simulated=True,
            replanning_recommended=False,
            recommendation_reason="None",
            summary="Nominal",
        ).model_dump_json(),
    )

    guardrail = build_dispatch_verification_guardrail(
        tasks.route_optimization, tasks.traffic_exception
    )

    valid_plan = DispatchPlan(
        dispatch_status="approved",
        route_source="initial_ortools",
        problem_type="VRPTW",
        dispatched_routes=opt_result.routes,
        all_serviced_customer_ids=[
            cid for r in opt_result.routes for cid in r.ordered_customer_ids
        ],
        solver_status=opt_result.solver_status,
        ortools_provenance_verified=True,
        summary="Verified",
    )
    ok, _ = guardrail(
        TaskOutput(
            description="dispatch",
            agent=agents.dispatch_coordinator.role,
            raw=valid_plan.model_dump_json(),
            pydantic=valid_plan,
        )
    )
    assert ok is True

    # Tamper with route sequence (simulate LLM fabrication)
    tampered_routes = [r.model_copy(deep=True) for r in opt_result.routes]
    tampered_routes[0].ordered_customer_ids = list(
        reversed(tampered_routes[0].ordered_customer_ids)
    )
    fabricated_plan = valid_plan.model_copy(update={"dispatched_routes": tampered_routes})
    ok_tampered, err_msg = guardrail(
        TaskOutput(
            description="dispatch",
            agent=agents.dispatch_coordinator.role,
            raw=fabricated_plan.model_dump_json(),
            pydantic=fabricated_plan,
        )
    )
    assert ok_tampered is False
    assert "fabricated or modified" in str(err_msg).lower()

    # Infeasible OR-Tools output cannot approve dispatch
    tasks.route_optimization.output = TaskOutput(
        description="route opt",
        agent=agents.route_optimizer.role,
        raw=RouteOptimizationReport(
            status="infeasible",
            problem_type="VRPTW",
            optimization_result=None,
            error_type="InfeasibleRoutingError",
            error_message="No solution",
            summary="Infeasible",
        ).model_dump_json(),
    )
    ok_inf, inf_msg = guardrail(
        TaskOutput(
            description="dispatch",
            agent=agents.dispatch_coordinator.role,
            raw=valid_plan.model_dump_json(),
            pydantic=valid_plan,
        )
    )
    assert ok_inf is False
    assert "no feasible" in str(inf_msg).lower()
