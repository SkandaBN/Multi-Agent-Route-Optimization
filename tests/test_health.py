"""Basic import verification and FastAPI health check tests."""

import importlib
import sys

from fastapi.testclient import TestClient

from app.main import app


def test_python_version_is_3_11() -> None:
    """Verify the active interpreter is Python 3.11."""
    assert (sys.version_info.major, sys.version_info.minor) == (3, 11)


def test_core_package_imports() -> None:
    """Verify all required third-party dependencies import cleanly."""
    required_modules = [
        "crewai",
        "ortools",
        "ortools.constraint_solver.pywrapcp",
        "fastapi",
        "pydantic",
        "pydantic_settings",
        "pytest",
    ]
    for module_name in required_modules:
        module = importlib.import_module(module_name)
        assert module is not None


def test_health_endpoint() -> None:
    """Verify the FastAPI /health endpoint responds with expected metadata."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app_name"] == "Smart Logistics API"
    assert payload["version"] == "0.1.0"
    assert payload["python_version"].startswith("3.11.")

