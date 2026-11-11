"""Verification utilities guaranteeing routes originate strictly from OR-Tools."""

from __future__ import annotations

from typing import Literal

from app.schemas.route import OptimizationResult, VehicleRoute


def validate_route_constraints(routes: list[VehicleRoute]) -> list[str]:
    """Validate structural and capacity invariants on a set of vehicle routes."""
    errors: list[str] = []
    seen_customers: set[str] = set()

    for route in routes:
        if not route.starting_depot_id or not route.ending_depot_id:
            errors.append(
                f"Vehicle '{route.vehicle_id}' is missing starting or ending depot."
            )
        if route.total_demand > route.vehicle_capacity:
            errors.append(
                f"Vehicle '{route.vehicle_id}' exceeds capacity: demand "
                f"{route.total_demand} > {route.vehicle_capacity}."
            )
        if route.total_distance_meters < 0 or route.estimated_travel_time_minutes < 0:
            errors.append(
                f"Vehicle '{route.vehicle_id}' has negative distance or travel time."
            )
        if route.stops:
            if [stop.customer_id for stop in route.stops] != route.ordered_customer_ids:
                errors.append(
                    f"Vehicle '{route.vehicle_id}' stop list does not match ordered_customer_ids."
                )
            stop_demand_sum = sum(stop.demand for stop in route.stops)
            if stop_demand_sum != route.total_demand:
                errors.append(
                    f"Vehicle '{route.vehicle_id}' stop demand sum ({stop_demand_sum}) "
                    f"does not equal total_demand ({route.total_demand})."
                )
        for customer_id in route.ordered_customer_ids:
            if customer_id in seen_customers:
                errors.append(
                    f"Duplicate customer '{customer_id}' found across dispatched routes."
                )
            seen_customers.add(customer_id)

    return errors


def verify_dispatch_routes_match_ortools(
    candidate_routes: list[VehicleRoute],
    expected_result: OptimizationResult | None,
    *,
    expected_problem_type: Literal["CVRP", "VRPTW"] | None = None,
) -> tuple[bool, list[str]]:
    """Verify dispatched routes satisfy constraints and match an OR-Tools result.

    Prevents any LLM-generated or modified route sequence from being dispatched.
    """
    errors = validate_route_constraints(candidate_routes)

    if expected_result is None:
        if candidate_routes:
            errors.append(
                "Dispatched routes present without a backing OR-Tools OptimizationResult."
            )
        return (len(errors) == 0, errors)

    if (
        expected_problem_type is not None
        and expected_result.problem_type != expected_problem_type
    ):
        errors.append(
            f"OptimizationResult problem_type '{expected_result.problem_type}' "
            f"mismatches expected '{expected_problem_type}'."
        )

    expected_dump = [route.model_dump() for route in expected_result.routes]
    candidate_dump = [route.model_dump() for route in candidate_routes]
    if candidate_dump != expected_dump:
        errors.append(
            "Candidate dispatch routes do not exactly match the Google OR-Tools solver output."
        )

    return (len(errors) == 0, errors)

