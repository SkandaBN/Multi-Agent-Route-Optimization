"""Unit tests for the six Phase III CrewAI logistics agents."""

from app.agents import (
    LogisticsAgentSuite,
    create_all_logistics_agents,
    create_dispatch_coordinator_agent,
    create_fleet_capacity_agent,
    create_order_validation_agent,
    create_performance_reporting_agent,
    create_route_optimization_agent,
    create_traffic_exception_agent,
)
from app.core.llm import MockLogisticsLLM


def test_create_all_logistics_agents_assigns_expected_tools_and_roles() -> None:
    """Verify all six CrewAI agents are constructed with the right roles and tools."""
    mock_llm = MockLogisticsLLM()
    suite = create_all_logistics_agents(mock_llm)

    assert isinstance(suite, LogisticsAgentSuite)
    assert "Order Analysis" in suite.order_validator.role
    assert [t.name for t in suite.order_validator.tools] == ["order_validation_tool"]

    assert "Fleet" in suite.fleet_manager.role
    assert [t.name for t in suite.fleet_manager.tools] == ["fleet_capacity_analysis_tool"]

    assert "Route Optimization" in suite.route_optimizer.role
    assert [t.name for t in suite.route_optimizer.tools] == [
        "cvrp_optimization_tool",
        "vrptw_optimization_tool",
    ]

    assert "Traffic" in suite.traffic_monitor.role
    assert [t.name for t in suite.traffic_monitor.tools] == [
        "traffic_disruption_simulation_tool",
        "dynamic_route_replanning_tool",
    ]

    assert "Dispatch Coordinator" in suite.dispatch_coordinator.role
    assert suite.dispatch_coordinator.tools == []

    assert "Performance" in suite.performance_reporter.role
    assert [t.name for t in suite.performance_reporter.tools] == [
        "logistics_performance_analysis_tool"
    ]

    for agent in (
        suite.order_validator,
        suite.fleet_manager,
        suite.route_optimizer,
        suite.traffic_monitor,
        suite.dispatch_coordinator,
        suite.performance_reporter,
    ):
        assert agent.llm is mock_llm
        assert agent.allow_delegation is False


def test_individual_agent_factories_create_distinct_agents() -> None:
    """Verify individual agent factory functions remain directly usable."""
    mock_llm = MockLogisticsLLM()
    assert create_order_validation_agent(mock_llm).llm is mock_llm
    assert create_fleet_capacity_agent(mock_llm).llm is mock_llm
    assert create_route_optimization_agent(mock_llm).llm is mock_llm
    assert create_traffic_exception_agent(mock_llm).llm is mock_llm
    assert create_dispatch_coordinator_agent(mock_llm).llm is mock_llm
    assert create_performance_reporting_agent(mock_llm).llm is mock_llm

