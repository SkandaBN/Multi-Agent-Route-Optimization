"""Reusable orchestration service facade for Smart Logistics workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from crewai.llms.base_llm import BaseLLM

from app.core.config import Settings, get_settings
from app.crews import SmartLogisticsCrew
from app.optimization import load_dataset_from_csv
from app.schemas.logistics import (
    OrchestrationRequest,
    OrchestrationResponse,
    TrafficDisruptionEvent,
)


class LogisticsOrchestrationService:
    """High-level application service exposing CrewAI + OR-Tools orchestration."""

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
        self.crew_runner = SmartLogisticsCrew(
            settings=self.settings,
            llm=llm,
            llm_mode=llm_mode,
            model_name=model_name,
            api_key=api_key,
            verbose=verbose,
        )

    def run(self, request: OrchestrationRequest) -> OrchestrationResponse:
        """Run the end-to-end CrewAI + OR-Tools logistics workflow on a request."""
        return self.crew_runner.run(request)

    def run_from_csv(
        self,
        data_dir: str | Path = "data",
        *,
        scenario: Literal["auto", "CVRP", "VRPTW"] = "auto",
        unavailable_vehicle_ids: list[str] | None = None,
        simulated_disruption: TrafficDisruptionEvent | None = None,
        time_limit_seconds: int | None = None,
        default_horizon_minutes: int = 1440,
        use_guided_local_search: bool = False,
        fuel_efficiency_km_per_liter: float | None = None,
        fuel_price_per_liter: float | None = None,
    ) -> OrchestrationResponse:
        """Load depot, vehicle, and customer CSV datasets and orchestrate dispatch."""
        depots, vehicles, customers = load_dataset_from_csv(data_dir)
        request = OrchestrationRequest(
            depots=depots,
            vehicles=vehicles,
            orders=[customer.model_dump() for customer in customers],
            scenario=scenario,
            unavailable_vehicle_ids=unavailable_vehicle_ids or [],
            simulated_disruption=simulated_disruption,
            time_limit_seconds=(
                time_limit_seconds
                if time_limit_seconds is not None
                else self.settings.ortools_solver_time_limit_seconds
            ),
            default_horizon_minutes=default_horizon_minutes,
            use_guided_local_search=use_guided_local_search,
            fuel_efficiency_km_per_liter=fuel_efficiency_km_per_liter,
            fuel_price_per_liter=fuel_price_per_liter,
        )
        return self.run(request)


def run_logistics_orchestration(
    request: OrchestrationRequest,
    *,
    llm_mode: Literal["auto", "gemini", "mock"] = "auto",
    verbose: bool = False,
) -> OrchestrationResponse:
    """Convenience entry point to execute the Smart Logistics multi-agent workflow."""
    service = LogisticsOrchestrationService(llm_mode=llm_mode, verbose=verbose)
    return service.run(request)

