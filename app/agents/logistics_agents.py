"""Definitions of the six CrewAI agents for Smart Logistics orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from crewai import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool

from app.tools import (
    CvrpOptimizationTool,
    DynamicRouteReplanningTool,
    FleetCapacityAnalysisTool,
    LogisticsPerformanceAnalysisTool,
    OrderValidationTool,
    TrafficDisruptionSimulationTool,
    VrptwOptimizationTool,
)


@dataclass(frozen=True)
class LogisticsAgentSuite:
    """Container holding all six specialized CrewAI logistics agents."""

    order_validator: Agent
    fleet_manager: Agent
    route_optimizer: Agent
    traffic_monitor: Agent
    dispatch_coordinator: Agent
    performance_reporter: Agent


def _build_agent(
    *,
    role: str,
    goal: str,
    backstory: str,
    llm: BaseLLM,
    tools: list[BaseTool],
    verbose: bool = False,
    max_iter: int = 6,
) -> Agent:
    """Construct a CrewAI Agent with consistent delegation and iteration defaults."""
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=verbose,
        max_iter=max_iter,
    )


def create_order_validation_agent(llm: BaseLLM, *, verbose: bool = False) -> Agent:
    """Create Agent 1: Order Analysis and Validation Agent."""
    return _build_agent(
        role="Order Analysis and Validation Agent",
        goal=(
            "Validate incoming delivery orders strictly against geospatial, demand, "
            "and time-window rules, rejecting invalid or duplicate records safely."
        ),
        backstory=(
            "You are a logistics data quality specialist. You always invoke "
            "order_validation_tool on incoming orders and return exact validated and "
            "rejected order counts without altering customer coordinates or demands."
        ),
        llm=llm,
        tools=[OrderValidationTool()],
        verbose=verbose,
    )


def create_fleet_capacity_agent(llm: BaseLLM, *, verbose: bool = False) -> Agent:
    """Create Agent 2: Fleet and Capacity Management Agent."""
    return _build_agent(
        role="Fleet and Capacity Management Agent",
        goal=(
            "Assess vehicle availability, depot assignments, and aggregate or per-order "
            "capacity sufficiency prior to mathematical route optimization."
        ),
        backstory=(
            "You manage fleet dispatch readiness across depots. You invoke "
            "fleet_capacity_analysis_tool to verify active vehicles and catch "
            "insufficient capacity before solver execution."
        ),
        llm=llm,
        tools=[FleetCapacityAnalysisTool()],
        verbose=verbose,
    )


def create_route_optimization_agent(llm: BaseLLM, *, verbose: bool = False) -> Agent:
    """Create Agent 3: Route Optimization Agent backed exclusively by Google OR-Tools."""
    return _build_agent(
        role="Route Optimization Agent",
        goal=(
            "Select the appropriate CVRP or VRPTW scenario and compute exact, "
            "constraint-satisfying vehicle routes exclusively via Google OR-Tools tools."
        ),
        backstory=(
            "You are a mathematical routing specialist. CRITICAL RULE: you MUST never "
            "invent, guess, or manually construct routes; every route sequence, "
            "distance, travel time, and arrival schedule must come strictly from "
            "cvrp_optimization_tool or vrptw_optimization_tool."
        ),
        llm=llm,
        tools=[CvrpOptimizationTool(), VrptwOptimizationTool()],
        verbose=verbose,
    )


def create_traffic_exception_agent(llm: BaseLLM, *, verbose: bool = False) -> Agent:
    """Create Agent 4: Traffic and Delay Exception Monitoring Agent."""
    return _build_agent(
        role="Traffic and Delay Exception Monitoring Agent",
        goal=(
            "Analyze controlled simulated traffic disruptions, clearly label them as "
            "simulated, and invoke dynamic OR-Tools replanning while preserving completed deliveries."
        ),
        backstory=(
            "You monitor simulated traffic incidents affecting active routes. Whenever a "
            "simulated disruption requires replanning, you invoke "
            "traffic_disruption_simulation_tool and dynamic_route_replanning_tool so "
            "completed stops remain preserved and remaining deliveries stay feasible."
        ),
        llm=llm,
        tools=[TrafficDisruptionSimulationTool(), DynamicRouteReplanningTool()],
        verbose=verbose,
    )


def create_dispatch_coordinator_agent(llm: BaseLLM, *, verbose: bool = False) -> Agent:
    """Create Agent 5: Logistics Dispatch Coordinator Agent."""
    return _build_agent(
        role="Logistics Dispatch Coordinator Agent",
        goal=(
            "Synthesize upstream validation, fleet, OR-Tools optimization, and traffic "
            "replanning outputs to approve only verified OR-Tools routes or block infeasible runs."
        ),
        backstory=(
            "You oversee final dispatch governance. You reject any unverified or "
            "fabricated routes, prioritize valid dynamically replanned OR-Tools routes "
            "when triggered, and block dispatch cleanly on infeasibility."
        ),
        llm=llm,
        tools=[],
        verbose=verbose,
    )


def create_performance_reporting_agent(llm: BaseLLM, *, verbose: bool = False) -> Agent:
    """Create Agent 6: Performance Monitoring and Reporting Agent."""
    return _build_agent(
        role="Performance Monitoring and Reporting Agent",
        goal=(
            "Generate accurate logistics KPIs, utilization metrics, time-window compliance, "
            "delivery completion rates, and fuel cost estimates from verified dispatch outputs."
        ),
        backstory=(
            "You audit final dispatch plans using logistics_performance_analysis_tool, "
            "ensuring time-window violation metrics are only reported when evaluated and "
            "every KPI matches OR-Tools ground-truth calculations."
        ),
        llm=llm,
        tools=[LogisticsPerformanceAnalysisTool()],
        verbose=verbose,
    )


def create_all_logistics_agents(
    llm: BaseLLM, *, verbose: bool = False
) -> LogisticsAgentSuite:
    """Build and return all six specialized CrewAI agents sharing the configured LLM."""
    return LogisticsAgentSuite(
        order_validator=create_order_validation_agent(llm, verbose=verbose),
        fleet_manager=create_fleet_capacity_agent(llm, verbose=verbose),
        route_optimizer=create_route_optimization_agent(llm, verbose=verbose),
        traffic_monitor=create_traffic_exception_agent(llm, verbose=verbose),
        dispatch_coordinator=create_dispatch_coordinator_agent(llm, verbose=verbose),
        performance_reporter=create_performance_reporting_agent(llm, verbose=verbose),
    )

