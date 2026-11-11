"""Integration tests for LogisticsOrchestrationService facade."""

from app.schemas.logistics import OrchestrationRequest, TrafficDisruptionEvent
from app.services import LogisticsOrchestrationService, run_logistics_orchestration


def test_orchestration_service_run_from_csv_end_to_end() -> None:
    """LogisticsOrchestrationService executes full CSV workflow cleanly in mock mode."""
    service = LogisticsOrchestrationService(llm_mode="mock")
    response = service.run_from_csv("data")

    assert response.order_validation.valid_count == 25
    assert response.dispatch_plan.dispatch_status == "approved"
    assert response.dispatch_plan.ortools_provenance_verified is True
    assert response.performance_report.report_status == "completed"
    assert response.performance_report.total_deliveries_accounted_for == 25


def test_run_logistics_orchestration_handles_partial_invalid_orders_safely() -> None:
    """Service filters out bad orders while successfully dispatching remaining valid orders."""
    service = LogisticsOrchestrationService(llm_mode="mock")
    baseline = service.run_from_csv("data")
    first_cust = baseline.dispatch_plan.dispatched_routes[0].ordered_customer_ids[0]

    # Build request directly using partial orders including one invalid order
    from app.optimization import load_dataset_from_csv

    depots, vehicles, customers = load_dataset_from_csv("data")
    mixed_orders = [c.model_dump() for c in customers[:5]] + [
        {
            "customer_id": "CUST_INVALID",
            "latitude": 12.95,
            "longitude": 77.60,
            "demand": -10,
        }
    ]
    req = OrchestrationRequest(
        depots=depots,
        vehicles=vehicles,
        orders=mixed_orders,
        simulated_disruption=TrafficDisruptionEvent(
            event_id="SIM-SVC-01",
            is_simulated=True,
            severity="low",
            completed_customer_ids=[first_cust] if first_cust in {c.customer_id for c in customers[:5]} else [],
        ),
    )
    resp = run_logistics_orchestration(req, llm_mode="mock")
    assert resp.order_validation.valid_count == 5
    assert resp.order_validation.rejected_count == 1
    assert "CUST_INVALID" in resp.dispatch_plan.rejected_customer_ids
    assert resp.dispatch_plan.dispatch_status == "approved"
    assert len(set(resp.dispatch_plan.all_serviced_customer_ids)) == 5

