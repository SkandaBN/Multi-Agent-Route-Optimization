"""LLM configuration and deterministic CrewAI mock provider for Smart Logistics."""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any, Literal

from crewai import LLM
from crewai.llms.base_llm import BaseLLM
from pydantic import Field

from app.core.config import Settings, get_settings

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage

PLACEHOLDER_KEYS = {
    "",
    "your_gemini_api_key_here",
    "your_openai_api_key_here",
    "your_api_key_here",
    "YOUR_API_KEY",
}


def resolve_gemini_api_key(
    settings: Settings | None = None,
    explicit_api_key: str | None = None,
) -> str:
    """Resolve Gemini API key from explicit value, env vars, or settings.

    Args:
        settings: Optional application settings instance.
        explicit_api_key: Optional directly provided API key.

    Returns:
        Non-placeholder API key string, or empty string if unavailable.
    """
    candidates = [
        explicit_api_key or "",
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GOOGLE_API_KEY", ""),
    ]
    if settings is not None:
        candidates.extend([settings.gemini_api_key, settings.google_api_key])
    else:
        cfg = get_settings()
        candidates.extend([cfg.gemini_api_key, cfg.google_api_key])

    for candidate in candidates:
        cleaned = (candidate or "").strip()
        if cleaned and cleaned not in PLACEHOLDER_KEYS:
            return cleaned
    return ""


def normalize_gemini_model_name(model_name: str) -> str:
    """Normalize Gemini model identifier for CrewAI's native Gemini provider."""
    cleaned = model_name.strip()
    if not cleaned:
        return "gemini/gemini-2.5-flash"
    if cleaned.startswith(("gemini/", "google/")):
        return cleaned
    return f"gemini/{cleaned}"


def create_gemini_llm(
    settings: Settings | None = None,
    *,
    model_name: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
) -> BaseLLM:
    """Create a CrewAI LLM configured for Google Gemini.

    Args:
        settings: Optional runtime Settings object.
        model_name: Optional override for the Gemini model name.
        api_key: Optional override for the Gemini/Google API key.
        temperature: Optional temperature override.

    Returns:
        Configured CrewAI LLM instance targeting Google Gemini.

    Raises:
        ValueError: If no valid GEMINI_API_KEY or GOOGLE_API_KEY is available.
    """
    cfg = settings or get_settings()
    resolved_key = resolve_gemini_api_key(cfg, explicit_api_key=api_key)
    if not resolved_key:
        raise ValueError(
            "Gemini API key is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )

    resolved_model = normalize_gemini_model_name(model_name or cfg.gemini_model_name)
    resolved_temp = cfg.gemini_temperature if temperature is None else temperature
    return LLM(
        model=resolved_model,
        api_key=resolved_key,
        temperature=resolved_temp,
    )


def extract_json_block(text: str, marker: str) -> dict[str, Any] | None:
    """Extract a JSON dictionary enclosed in `[MARKER]...[/MARKER]` tags."""
    pattern = re.compile(
        rf"\[{re.escape(marker)}\]\s*(.*?)\s*\[/{re.escape(marker)}\]",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    raw_json = match.group(1).strip()
    if not raw_json:
        return None
    parsed = json.loads(raw_json)
    return parsed if isinstance(parsed, dict) else None


def extract_observations_from_messages(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Extract `(tool_name, observation_json)` pairs from CrewAI ReAct messages."""
    observations: list[tuple[str, str]] = []
    action_pattern = re.compile(r"Action:\s*([^\r\n]+)", re.IGNORECASE)
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or "\nObservation:" not in content:
            continue
        action_match = action_pattern.search(content)
        tool_name = action_match.group(1).strip() if action_match else ""
        obs_body = content.split("\nObservation:", 1)[1].strip()
        observations.append((tool_name, obs_body))
    return observations


def _try_parse_json(raw: str) -> dict[str, Any] | None:
    """Parse JSON object from tool observation text, handling trailing suffix lines."""
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    start = raw.find("{")
    if start == -1:
        return None
    try:
        loaded, _ = decoder.raw_decode(raw[start:])
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        return None


class MockLogisticsLLM(BaseLLM):
    """Deterministic CrewAI BaseLLM for offline Phase III multi-agent testing.

    Drives authentic CrewAI ReAct tool calls (`Action:` / `Action Input:`) and
    processes real tool `Observation:` responses without making external API calls.
    """

    llm_type: str = "mock"
    model: str = "mock/gemini-2.5-flash"
    provider: str = "mock"
    context_payload: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, model: str = "mock/gemini-2.5-flash", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)

    def supports_function_calling(self) -> bool:
        """Use CrewAI's ReAct text action loop for deterministic mock tool calls."""
        return False

    def supports_stop_words(self) -> bool:
        """Support stop word truncation configured by CrewAI executor."""
        return True

    def get_context_window_size(self) -> int:
        """Provide generous local context window size."""
        return 128_000

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Respond deterministically based on task tag, context payload, and observations."""
        msg_list: list[dict[str, Any]] = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else list(messages)
        )
        full_prompt = "\n".join(
            str(m.get("content", "")) for m in msg_list if isinstance(m, dict)
        )
        observations = extract_observations_from_messages(msg_list)
        task_name = getattr(from_task, "name", "") or ""

        if task_name == "order_validation_task" or "[TASK_1_ORDER_VALIDATION]" in full_prompt:
            response = self._handle_order_validation(observations)
        elif task_name == "fleet_capacity_task" or "[TASK_2_FLEET_CAPACITY]" in full_prompt:
            response = self._handle_fleet_capacity(full_prompt, observations)
        elif task_name == "optimization_prep_task" or "[TASK_3_OPTIMIZATION_PREP]" in full_prompt:
            response = self._handle_optimization_prep(full_prompt)
        elif task_name == "route_optimization_task" or "[TASK_4_ROUTE_OPTIMIZATION]" in full_prompt:
            response = self._handle_route_optimization(full_prompt, observations)
        elif task_name == "traffic_exception_task" or "[TASK_5_TRAFFIC_EXCEPTION]" in full_prompt:
            response = self._handle_traffic_exception(full_prompt, observations)
        elif task_name == "dispatch_coordination_task" or "[TASK_6_DISPATCH_COORDINATION]" in full_prompt:
            response = self._handle_dispatch_coordination(full_prompt)
        elif task_name == "performance_reporting_task" or "[TASK_7_PERFORMANCE_REPORT]" in full_prompt:
            response = self._handle_performance_report(full_prompt, observations)
        else:
            response = (
                "Thought: No specialized logistics task marker detected.\n"
                "Final Answer: {\"status\": \"ok\"}"
            )
        return self._apply_stop_words(response)

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async delegation to deterministic synchronous handler."""
        return self.call(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    def _handle_order_validation(self, observations: list[tuple[str, str]]) -> str:
        if not observations:
            action_input = {"orders": self.context_payload.get("orders", [])}
            return (
                "Thought: I need to validate all submitted orders using order_validation_tool.\n"
                "Action: order_validation_tool\n"
                f"Action Input: {json.dumps(action_input)}"
            )
        parsed = _try_parse_json(observations[-1][1])
        final_payload = json.dumps(parsed) if parsed is not None else observations[-1][1]
        return f"Thought: Order validation tool returned structured results.\nFinal Answer: {final_payload}"

    def _handle_fleet_capacity(
        self, full_prompt: str, observations: list[tuple[str, str]]
    ) -> str:
        if not observations:
            order_report = self._extract_order_report_from_prompt(full_prompt)
            valid_orders = (
                order_report.get("valid_orders", [])
                if order_report
                else self.context_payload.get("valid_orders", [])
            )
            action_input = {
                "depots": self.context_payload.get("depots", []),
                "vehicles": self.context_payload.get("vehicles", []),
                "valid_orders": valid_orders,
                "unavailable_vehicle_ids": self.context_payload.get(
                    "unavailable_vehicle_ids", []
                ),
            }
            return (
                "Thought: Analyzing fleet availability, depot assignments, and capacity using fleet_capacity_analysis_tool.\n"
                "Action: fleet_capacity_analysis_tool\n"
                f"Action Input: {json.dumps(action_input)}"
            )
        parsed = _try_parse_json(observations[-1][1])
        final_payload = json.dumps(parsed) if parsed is not None else observations[-1][1]
        return f"Thought: Fleet capacity assessment finished.\nFinal Answer: {final_payload}"

    def _handle_optimization_prep(self, full_prompt: str) -> str:
        order_report = self._extract_order_report_from_prompt(full_prompt)
        fleet_report = self._extract_fleet_report_from_prompt(full_prompt)

        valid_for_opt = bool(order_report and order_report.get("is_valid_for_optimization"))
        cap_ok = bool(fleet_report and fleet_report.get("capacity_sufficient"))
        requested_scenario = self.context_payload.get("scenario", "auto")
        has_time_windows = bool(order_report and order_report.get("has_time_windows"))

        if requested_scenario in ("CVRP", "VRPTW"):
            scenario: Literal["CVRP", "VRPTW"] = requested_scenario
        else:
            scenario = "VRPTW" if has_time_windows else "CVRP"

        blocking_reasons: list[str] = []
        if not valid_for_opt:
            blocking_reasons.append(
                (order_report or {}).get("summary", "Order validation failed.")
            )
        if not cap_ok:
            issues = (fleet_report or {}).get("issues", [])
            blocking_reasons.extend(issues or ["Insufficient or invalid fleet capacity."])

        depots = (fleet_report or {}).get("depots", [])
        avail_vehicles = (fleet_report or {}).get("available_vehicles", [])
        prep_plan = {
            "ready_for_optimization": bool(valid_for_opt and cap_ok),
            "recommended_scenario": scenario,
            "rationale": (
                f"Selected {scenario} based on scenario='{requested_scenario}' and "
                f"has_time_windows={has_time_windows}."
            ),
            "active_depot_ids": [d["depot_id"] for d in depots],
            "active_vehicle_ids": [v["vehicle_id"] for v in avail_vehicles],
            "valid_customer_count": (order_report or {}).get("valid_count", 0),
            "total_demand": (order_report or {}).get("total_valid_demand", 0),
            "time_limit_seconds": self.context_payload.get("time_limit_seconds", 5),
            "default_horizon_minutes": self.context_payload.get(
                "default_horizon_minutes", 1440
            ),
            "use_guided_local_search": bool(
                self.context_payload.get("use_guided_local_search", False)
            ),
            "blocking_reasons": blocking_reasons,
        }
        return (
            "Thought: Synthesized optimization preparation plan from validated orders and fleet analysis.\n"
            f"Final Answer: {json.dumps(prep_plan)}"
        )

    def _handle_route_optimization(
        self, full_prompt: str, observations: list[tuple[str, str]]
    ) -> str:
        prep = self._extract_prep_plan_from_prompt(full_prompt) or {}
        scenario = prep.get("recommended_scenario", "VRPTW")

        if not prep.get("ready_for_optimization", False):
            reasons = prep.get("blocking_reasons") or ["Pre-optimization checks failed."]
            blocked = {
                "status": "blocked",
                "problem_type": scenario,
                "solver_engine": "OR-Tools",
                "optimization_result": None,
                "error_type": "PreOptimizationBlocked",
                "error_message": "; ".join(reasons),
                "summary": f"Optimization blocked before running OR-Tools: {'; '.join(reasons)}",
            }
            return (
                "Thought: Pre-optimization checks blocked solver execution.\n"
                f"Final Answer: {json.dumps(blocked)}"
            )

        if not observations:
            order_report = self._extract_order_report_from_prompt(full_prompt) or {}
            fleet_report = self._extract_fleet_report_from_prompt(full_prompt) or {}
            depots = fleet_report.get("depots") or self.context_payload.get("depots", [])
            vehicles = fleet_report.get("available_vehicles") or self.context_payload.get(
                "vehicles", []
            )
            customers = order_report.get("valid_orders", [])
            action_input: dict[str, Any] = {
                "depots": depots,
                "vehicles": vehicles,
                "customers": customers,
                "time_limit_seconds": prep.get("time_limit_seconds", 5),
                "use_guided_local_search": prep.get("use_guided_local_search", False),
            }
            if scenario == "VRPTW":
                tool_name = "vrptw_optimization_tool"
                action_input["default_horizon_minutes"] = prep.get(
                    "default_horizon_minutes", 1440
                )
            else:
                tool_name = "cvrp_optimization_tool"
            return (
                f"Thought: Invoking {tool_name} to compute exact Google OR-Tools routes.\n"
                f"Action: {tool_name}\n"
                f"Action Input: {json.dumps(action_input)}"
            )

        parsed = _try_parse_json(observations[-1][1])
        final_payload = json.dumps(parsed) if parsed is not None else observations[-1][1]
        return f"Thought: Returning exact OR-Tools optimization output.\nFinal Answer: {final_payload}"

    def _handle_traffic_exception(
        self, full_prompt: str, observations: list[tuple[str, str]]
    ) -> str:
        route_report = self._extract_route_report_from_prompt(full_prompt) or {}
        opt_result = route_report.get("optimization_result")
        disruption_event = self.context_payload.get("simulated_disruption")

        if opt_result is None or disruption_event is None:
            report = {
                "has_disruption": False,
                "is_simulated": True,
                "simulation_notice": "SIMULATED TRAFFIC DATA — controlled simulation for Phase III validation",
                "disruption_event": disruption_event,
                "impacted_vehicles": [],
                "replanning_recommended": False,
                "recommendation_reason": (
                    "Initial optimization produced no feasible routes."
                    if opt_result is None
                    else "No simulated traffic disruption event injected."
                ),
                "replanning_report": None,
                "summary": (
                    "Traffic monitoring skipped replanning because no feasible initial routes exist."
                    if opt_result is None
                    else "Nominal simulated traffic check completed; no disruption active."
                ),
            }
            return (
                "Thought: Completing traffic monitoring status without replanning.\n"
                f"Final Answer: {json.dumps(report)}"
            )

        order_report = self._extract_order_report_from_prompt(full_prompt) or {}
        valid_orders = order_report.get("valid_orders", [])
        fleet_report = self._extract_fleet_report_from_prompt(full_prompt) or {}
        depots = fleet_report.get("depots") or self.context_payload.get("depots", [])
        vehicles = fleet_report.get("available_vehicles") or self.context_payload.get(
            "vehicles", []
        )

        if not observations:
            sim_input = {
                "optimization_result": opt_result,
                "orders": valid_orders,
                "disruption_event": disruption_event,
            }
            return (
                "Thought: Assessing simulated traffic impact using traffic_disruption_simulation_tool.\n"
                "Action: traffic_disruption_simulation_tool\n"
                f"Action Input: {json.dumps(sim_input)}"
            )

        if len(observations) == 1:
            sim_report = _try_parse_json(observations[0][1]) or {}
            if not sim_report.get("replanning_recommended", False):
                return (
                    "Thought: Simulated traffic impact does not require replanning.\n"
                    f"Final Answer: {json.dumps(sim_report)}"
                )
            replan_input = {
                "depots": depots,
                "vehicles": vehicles,
                "orders": valid_orders,
                "initial_optimization_result": opt_result,
                "disruption_event": disruption_event,
                "time_limit_seconds": self.context_payload.get("time_limit_seconds", 5),
                "default_horizon_minutes": self.context_payload.get(
                    "default_horizon_minutes", 1440
                ),
                "use_guided_local_search": bool(
                    self.context_payload.get("use_guided_local_search", False)
                ),
            }
            return (
                "Thought: Disruption warrants dynamic replanning while preserving completed deliveries.\n"
                "Action: dynamic_route_replanning_tool\n"
                f"Action Input: {json.dumps(replan_input)}"
            )

        sim_report = _try_parse_json(observations[0][1]) or {}
        replan_report = _try_parse_json(observations[1][1])
        sim_report["replanning_report"] = replan_report
        sim_report["summary"] = (
            f"{sim_report.get('summary', '')} Dynamic replanning status: "
            f"{(replan_report or {}).get('status', 'unknown')}."
        ).strip()
        return (
            "Thought: Combined simulated traffic assessment and OR-Tools dynamic replanning report.\n"
            f"Final Answer: {json.dumps(sim_report)}"
        )

    def _handle_dispatch_coordination(self, full_prompt: str) -> str:
        order_report = self._extract_order_report_from_prompt(full_prompt) or {}
        route_report = self._extract_route_report_from_prompt(full_prompt) or {}
        traffic_report = self._extract_traffic_report_from_prompt(full_prompt) or {}
        prep_plan = self._extract_prep_plan_from_prompt(full_prompt) or {}

        problem_type = route_report.get(
            "problem_type", prep_plan.get("recommended_scenario", "VRPTW")
        )
        rejected_ids = [
            r.get("customer_id", "<unknown>")
            for r in order_report.get("rejected_orders", [])
        ]

        replanning_report = traffic_report.get("replanning_report")
        has_disruption = bool(traffic_report.get("has_disruption", False))

        blocking_reasons: list[str] = []
        dispatched_routes: list[dict[str, Any]] = []
        preserved_completed: list[dict[str, Any]] = []
        unassigned_ids: list[str] = []
        solver_status = "NOT_RUN"
        route_source: Literal["initial_ortools", "replanned_ortools", "none"] = "none"
        dispatch_status: Literal["approved", "blocked"] = "blocked"

        if route_report.get("status") != "feasible" or route_report.get("optimization_result") is None:
            blocking_reasons.append(
                route_report.get("error_message")
                or route_report.get("summary")
                or "Initial OR-Tools optimization did not produce a feasible solution."
            )
        elif replanning_report is not None and replanning_report.get("replanning_triggered"):
            replan_status = replanning_report.get("status")
            preserved_completed = replanning_report.get(
                "preserved_completed_deliveries", []
            )
            if replan_status == "replanned":
                replanned_opt = replanning_report.get("replanned_result")
                dispatch_status = "approved"
                route_source = "replanned_ortools"
                if replanned_opt is not None:
                    dispatched_routes = replanned_opt.get("routes", [])
                    solver_status = replanned_opt.get("solver_status", "REPLANNED")
                    unassigned_ids = replanned_opt.get("unassigned_customer_ids", [])
                    problem_type = replanned_opt.get("problem_type", problem_type)
                else:
                    solver_status = "ALL_DELIVERIES_ALREADY_COMPLETED"
            else:
                blocking_reasons.append(
                    replanning_report.get("error_message")
                    or replanning_report.get("summary")
                    or "Dynamic replanning failed to find a feasible OR-Tools solution."
                )
                solver_status = f"REPLAN_{str(replan_status).upper()}"
        else:
            initial_opt = route_report["optimization_result"]
            dispatch_status = "approved"
            route_source = "initial_ortools"
            dispatched_routes = initial_opt.get("routes", [])
            solver_status = initial_opt.get("solver_status", "ROUTING_SUCCESS")
            unassigned_ids = initial_opt.get("unassigned_customer_ids", [])
            problem_type = initial_opt.get("problem_type", problem_type)

        serviced_ids = [c["customer_id"] for c in preserved_completed]
        for route in dispatched_routes:
            serviced_ids.extend(route.get("ordered_customer_ids", []))

        plan = {
            "dispatch_status": dispatch_status,
            "route_source": route_source,
            "problem_type": problem_type,
            "is_simulated_traffic_used": has_disruption,
            "dispatched_routes": dispatched_routes,
            "preserved_completed_deliveries": preserved_completed,
            "all_serviced_customer_ids": serviced_ids,
            "rejected_customer_ids": rejected_ids,
            "unassigned_customer_ids": unassigned_ids,
            "solver_status": solver_status,
            "blocking_reasons": blocking_reasons,
            "ortools_provenance_verified": dispatch_status == "approved",
            "summary": (
                f"Dispatch approved ({route_source}) covering {len(serviced_ids)} customers across {len(dispatched_routes)} OR-Tools routes."
                if dispatch_status == "approved"
                else f"Dispatch safely blocked: {'; '.join(blocking_reasons)}"
            ),
        }
        return (
            "Thought: Verified OR-Tools route provenance and synthesized final dispatch plan.\n"
            f"Final Answer: {json.dumps(plan)}"
        )

    def _handle_performance_report(
        self, full_prompt: str, observations: list[tuple[str, str]]
    ) -> str:
        if not observations:
            dispatch_plan = self._extract_dispatch_plan_from_prompt(full_prompt) or {}
            order_report = self._extract_order_report_from_prompt(full_prompt) or {}
            fleet_report = self._extract_fleet_report_from_prompt(full_prompt) or {}
            route_report = self._extract_route_report_from_prompt(full_prompt) or {}
            traffic_report = self._extract_traffic_report_from_prompt(full_prompt) or {}

            route_source = dispatch_plan.get("route_source", "none")
            replan = traffic_report.get("replanning_report") or {}
            initial_opt_result = route_report.get("optimization_result")
            orig_vehicles = fleet_report.get("available_vehicles") or self.context_payload.get(
                "vehicles", []
            )
            is_replanned = route_source == "replanned_ortools"
            active_opt_result = None
            active_remaining_vehicles: list[dict[str, Any]] = orig_vehicles
            disabled_vehicle_ids: list[str] = []
            pre_disruption_distance_m: int | None = None

            if dispatch_plan.get("dispatch_status") == "approved":
                if is_replanned:
                    active_opt_result = replan.get("replanned_result")
                    active_remaining_vehicles = replan.get(
                        "active_remaining_vehicles", []
                    )
                    disabled_vehicle_ids = replan.get("disabled_vehicle_ids", [])
                    pre_disruption_distance_m = replan.get(
                        "distance_travelled_before_disruption_meters"
                    )
                elif route_source == "initial_ortools":
                    active_opt_result = initial_opt_result
            elif replan.get("replanning_triggered"):
                is_replanned = True
                active_remaining_vehicles = replan.get("active_remaining_vehicles", [])
                disabled_vehicle_ids = replan.get("disabled_vehicle_ids", [])
                pre_disruption_distance_m = replan.get(
                    "distance_travelled_before_disruption_meters"
                )

            action_input = {
                "optimization_result": active_opt_result,
                "original_optimization_result": initial_opt_result,
                "completed_deliveries": dispatch_plan.get(
                    "preserved_completed_deliveries", []
                ),
                "total_submitted_orders": order_report.get("total_submitted", 0),
                "dispatch_status": dispatch_plan.get("dispatch_status", "blocked"),
                "problem_type": dispatch_plan.get("problem_type", "VRPTW"),
                "vehicles": orig_vehicles,
                "original_vehicles": orig_vehicles,
                "active_remaining_vehicles": active_remaining_vehicles,
                "disabled_vehicle_ids": disabled_vehicle_ids,
                "is_replanned": is_replanned,
                "distance_travelled_before_disruption_meters": pre_disruption_distance_m,
                "fuel_efficiency_km_per_liter": self.context_payload.get(
                    "fuel_efficiency_km_per_liter", 8.0
                ),
                "fuel_price_per_liter": self.context_payload.get(
                    "fuel_price_per_liter", 1.5
                ),
            }
            return (
                "Thought: Evaluating final dispatch KPIs and fuel estimates using logistics_performance_analysis_tool.\n"
                "Action: logistics_performance_analysis_tool\n"
                f"Action Input: {json.dumps(action_input)}"
            )

        parsed = _try_parse_json(observations[-1][1])
        final_payload = json.dumps(parsed) if parsed is not None else observations[-1][1]
        return f"Thought: Performance report generated from verified metrics tool.\nFinal Answer: {final_payload}"

    @staticmethod
    def _extract_context_section(prompt: str, OrderMarker: str) -> dict[str, Any] | None:
        return extract_json_block(prompt, OrderMarker)

    @staticmethod
    def _find_json_matching_keys(
        prompt: str, required_keys: set[str]
    ) -> dict[str, Any] | None:
        """Scan embedded JSON blocks / CrewAI task contexts for a dict with required keys."""
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        idx = 0
        while True:
            pos = prompt.find("{", idx)
            if pos == -1:
                break
            try:
                obj, end = decoder.raw_decode(prompt[pos:])
                if isinstance(obj, dict) and required_keys.issubset(obj.keys()):
                    candidates.append(obj)
                idx = pos + max(end, 1)
            except json.JSONDecodeError:
                idx = pos + 1
        return candidates[-1] if candidates else None

    def _extract_order_report_from_prompt(self, prompt: str) -> dict[str, Any] | None:
        return self._find_json_matching_keys(
            prompt, {"valid_orders", "rejected_orders", "is_valid_for_optimization"}
        )

    def _extract_fleet_report_from_prompt(self, prompt: str) -> dict[str, Any] | None:
        return self._find_json_matching_keys(
            prompt, {"available_vehicles", "capacity_sufficient", "total_fleet_capacity"}
        )

    def _extract_prep_plan_from_prompt(self, prompt: str) -> dict[str, Any] | None:
        return self._find_json_matching_keys(
            prompt, {"ready_for_optimization", "recommended_scenario", "blocking_reasons"}
        )

    def _extract_route_report_from_prompt(self, prompt: str) -> dict[str, Any] | None:
        return self._find_json_matching_keys(
            prompt, {"status", "problem_type", "solver_engine", "optimization_result"}
        )

    def _extract_traffic_report_from_prompt(self, prompt: str) -> dict[str, Any] | None:
        return self._find_json_matching_keys(
            prompt, {"has_disruption", "is_simulated", "replanning_recommended"}
        )

    def _extract_dispatch_plan_from_prompt(self, prompt: str) -> dict[str, Any] | None:
        return self._find_json_matching_keys(
            prompt, {"dispatch_status", "route_source", "dispatched_routes"}
        )


def create_llm(
    settings: Settings | None = None,
    *,
    mode: Literal["auto", "gemini", "mock"] | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    context_payload: dict[str, Any] | None = None,
) -> BaseLLM:
    """Create either a Google Gemini LLM or deterministic MockLogisticsLLM.

    Args:
        settings: Optional application Settings object.
        mode: Execution mode ('auto', 'gemini', or 'mock').
        model_name: Optional Gemini model name override.
        api_key: Optional explicit Gemini API key.
        context_payload: Deterministic payload used when creating MockLogisticsLLM.

    Returns:
        Configured CrewAI BaseLLM instance.
    """
    cfg = settings or get_settings()
    selected_mode = mode or cfg.llm_mode

    if selected_mode == "gemini":
        return create_gemini_llm(cfg, model_name=model_name, api_key=api_key)
    if selected_mode == "mock":
        return MockLogisticsLLM(
            model=f"mock/{model_name or cfg.gemini_model_name}",
            context_payload=context_payload or {},
        )

    resolved_key = resolve_gemini_api_key(cfg, explicit_api_key=api_key)
    if resolved_key:
        return create_gemini_llm(cfg, model_name=model_name, api_key=resolved_key)
    return MockLogisticsLLM(
        model=f"mock/{model_name or cfg.gemini_model_name}",
        context_payload=context_payload or {},
    )
