"""CrewAI sequential multi-agent orchestration crew for Smart Logistics."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from crewai import Crew, Process
from crewai.llms.base_llm import BaseLLM

from app.agents import LogisticsAgentSuite, create_all_logistics_agents
from app.core.config import Settings, get_settings
from app.core.llm import MockLogisticsLLM, create_llm
from app.schemas.logistics import (
    DispatchPlan,
    FleetCapacityReport,
    LogisticsPerformanceReport,
    OptimizationPreparationPlan,
    OrchestrationRequest,
    OrchestrationResponse,
    OrderValidationReport,
    RouteOptimizationReport,
    TrafficExceptionReport,
)
from app.tasks import LogisticsTaskSuite, create_logistics_tasks


def _require_task_model(task_output: Any, model_cls: type[Any], task_name: str) -> Any:
    """Extract or validate the Pydantic result from a completed CrewAI TaskOutput."""
    if task_output is None:
        raise RuntimeError(f"CrewAI task '{task_name}' produced no output.")
    pydantic_val = getattr(task_output, "pydantic", None)
    if pydantic_val is not None and isinstance(pydantic_val, model_cls):
        return pydantic_val
    raw = getattr(task_output, "raw", "")
    if not raw:
        raise RuntimeError(f"CrewAI task '{task_name}' returned an empty raw output.")
    return model_cls.model_validate_json(raw)


def build_context_payload(
    request: OrchestrationRequest, settings: Settings
) -> dict[str, Any]:
    """Serialize orchestration inputs for CrewAI prompt templates and MockLogisticsLLM."""
    fuel_eff = (
        request.fuel_efficiency_km_per_liter
        if request.fuel_efficiency_km_per_liter is not None
        else settings.fuel_efficiency_km_per_liter
    )
    fuel_price = (
        request.fuel_price_per_liter
        if request.fuel_price_per_liter is not None
        else settings.fuel_price_per_liter
    )
    return {
        "depots": [depot.model_dump() for depot in request.depots],
        "vehicles": [vehicle.model_dump() for vehicle in request.vehicles],
        "orders": list(request.orders),
        "unavailable_vehicle_ids": list(request.unavailable_vehicle_ids),
        "scenario": request.scenario,
        "time_limit_seconds": request.time_limit_seconds,
        "default_horizon_minutes": request.default_horizon_minutes,
        "use_guided_local_search": request.use_guided_local_search,
        "simulated_disruption": (
            request.simulated_disruption.model_dump()
            if request.simulated_disruption is not None
            else None
        ),
        "fuel_efficiency_km_per_liter": fuel_eff,
        "fuel_price_per_liter": fuel_price,
    }


def prepare_crew_inputs(
    request: OrchestrationRequest, settings: Settings
) -> dict[str, str]:
    """Format template inputs dictionary expected by Crew.kickoff(inputs=...)."""
    payload = build_context_payload(request, settings)
    return {
        "orders_json": json.dumps(payload["orders"]),
        "depots_json": json.dumps(payload["depots"]),
        "vehicles_json": json.dumps(payload["vehicles"]),
        "unavailable_vehicle_ids_json": json.dumps(payload["unavailable_vehicle_ids"]),
        "scenario": str(payload["scenario"]),
        "time_limit_seconds": str(payload["time_limit_seconds"]),
        "default_horizon_minutes": str(payload["default_horizon_minutes"]),
        "use_guided_local_search": str(payload["use_guided_local_search"]).lower(),
        "simulated_disruption_json": json.dumps(payload["simulated_disruption"]),
        "fuel_efficiency_km_per_liter": str(payload["fuel_efficiency_km_per_liter"]),
        "fuel_price_per_liter": str(payload["fuel_price_per_liter"]),
    }


class SmartLogisticsCrew:
    """Assembles and executes the 6-agent CrewAI Smart Logistics optimization pipeline."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: BaseLLM | None = None,
        llm_mode: Literal["auto", "gemini", "mock"] | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        if self.settings.crewai_telemetry_opt_out:
            os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
        self._explicit_llm = llm
        self.llm_mode = llm_mode
        self.model_name = model_name
        self.api_key = api_key
        self.verbose = verbose

    def _resolve_llm(self, context_payload: dict[str, Any]) -> BaseLLM:
        if self._explicit_llm is not None:
            return self._explicit_llm
        return create_llm(
            self.settings,
            mode=self.llm_mode,
            model_name=self.model_name,
            api_key=self.api_key,
            context_payload=context_payload,
        )

    def build(
        self, request: OrchestrationRequest
    ) -> tuple[Crew, LogisticsAgentSuite, LogisticsTaskSuite, BaseLLM]:
        """Construct the CrewAI Crew, agents, and sequential tasks for a request."""
        payload = build_context_payload(request, self.settings)
        llm = self._resolve_llm(payload)
        agents = create_all_logistics_agents(llm, verbose=self.verbose)
        tasks = create_logistics_tasks(agents)
        crew = Crew(
            agents=[
                agents.order_validator,
                agents.fleet_manager,
                agents.route_optimizer,
                agents.traffic_monitor,
                agents.dispatch_coordinator,
                agents.performance_reporter,
            ],
            tasks=tasks.as_list(),
            process=Process.sequential,
            memory=False,
            cache=False,
            verbose=self.verbose,
        )
        return crew, agents, tasks, llm

    def run(self, request: OrchestrationRequest) -> OrchestrationResponse:
        """Execute the full sequential multi-agent logistics pipeline."""
        crew, _, tasks, llm = self.build(request)
        crew_inputs = prepare_crew_inputs(request, self.settings)
        crew_output = crew.kickoff(inputs=crew_inputs)

        order_validation = _require_task_model(
            tasks.order_validation.output, OrderValidationReport, "order_validation_task"
        )
        fleet_capacity = _require_task_model(
            tasks.fleet_capacity.output, FleetCapacityReport, "fleet_capacity_task"
        )
        optimization_prep = _require_task_model(
            tasks.optimization_prep.output,
            OptimizationPreparationPlan,
            "optimization_prep_task",
        )
        route_optimization = _require_task_model(
            tasks.route_optimization.output,
            RouteOptimizationReport,
            "route_optimization_task",
        )
        traffic_exception = _require_task_model(
            tasks.traffic_exception.output,
            TrafficExceptionReport,
            "traffic_exception_task",
        )
        dispatch_plan = _require_task_model(
            tasks.dispatch_coordination.output,
            DispatchPlan,
            "dispatch_coordination_task",
        )
        performance_report = _require_task_model(
            tasks.performance_reporting.output,
            LogisticsPerformanceReport,
            "performance_reporting_task",
        )

        execution_mode: Literal["mock", "gemini"] = (
            "mock" if isinstance(llm, MockLogisticsLLM) else "gemini"
        )
        return OrchestrationResponse(
            execution_mode=execution_mode,
            model_name=getattr(llm, "model", self.settings.gemini_model_name),
            order_validation=order_validation,
            fleet_capacity=fleet_capacity,
            optimization_preparation=optimization_prep,
            route_optimization=route_optimization,
            traffic_exception=traffic_exception,
            dispatch_plan=dispatch_plan,
            performance_report=performance_report,
            crew_raw_output=getattr(crew_output, "raw", "") or "",
        )

