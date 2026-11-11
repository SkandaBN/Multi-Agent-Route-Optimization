# Multi-Agent Route Optimization for Smart Logistics

Modular Smart Logistics platform combining **FastAPI**, **Pydantic**, **Google OR-Tools**, and **CrewAI** on **Python 3.11**.

> **Status:** Phase I (scaffolding & FastAPI health check), Phase II (Google OR-Tools CVRP/VRPTW optimization engine, route extraction, and performance metrics), and Phase III (CrewAI 6-agent / 7-tool / 7-task multi-agent orchestration with Google Gemini support, simulated traffic disruption replanning, and OR-Tools provenance guardrails) are implemented and tested.

## Project Structure

```text
.
├── .env.example              # Environment variable template (Google Gemini + OR-Tools settings)
├── .gitignore                # Git ignore rules
├── .python-version           # Target Python version (3.11)
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Pinned Python dependencies
├── README.md
├── data/
│   ├── customers.csv         # 25 synthetic customers with coordinates, demand, and time windows
│   ├── depots.csv            # 2 synthetic distribution depots
│   └── vehicles.csv          # 5 synthetic vehicles with capacities and depot assignments
├── scripts/
│   └── run_phase3_demo.py    # End-to-end Phase III baseline + simulated disruption demo
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI application factory & entrypoint
│   ├── api/
│   │   └── routes/
│   │       └── health.py     # GET /health endpoint
│   ├── core/
│   │   ├── config.py         # Pydantic BaseSettings configuration (Gemini, solver, fuel params)
│   │   └── llm.py            # Gemini CrewAI LLM factory & deterministic offline MockLogisticsLLM
│   ├── schemas/
│   │   ├── health.py         # Health response model
│   │   ├── order.py          # GeoPoint, Depot, and CustomerOrder schemas
│   │   ├── vehicle.py        # Vehicle schema
│   │   ├── route.py          # RouteStop, VehicleRoute, PerformanceMetrics, OptimizationResult
│   │   └── logistics.py      # Phase III reports, simulated traffic events, dispatch & KPI schemas
│   ├── optimization/
│   │   ├── __init__.py       # Exceptions & public exports
│   │   ├── distance_matrix.py# Pluggable DistanceMatrixProvider & Haversine/travel-time matrices
│   │   ├── cvrp.py           # Single-depot & multi-depot CVRP solver + CSV dataset loader
│   │   ├── vrptw.py          # Single-depot & multi-depot VRPTW solver
│   │   ├── route_extractor.py# Structured route extraction from OR-Tools solutions
│   │   └── metrics.py        # Fleet & schedule performance metrics
│   ├── tools/
│   │   ├── validation_tools.py   # OrderValidationTool & FleetCapacityAnalysisTool
│   │   ├── optimization_tools.py # CvrpOptimizationTool & VrptwOptimizationTool
│   │   ├── traffic_tools.py      # TrafficDisruptionSimulationTool & DynamicRouteReplanningTool
│   │   ├── reporting_tools.py    # LogisticsPerformanceAnalysisTool
│   │   └── provenance.py         # OR-Tools route verification & anti-fabrication helpers
│   ├── agents/
│   │   └── logistics_agents.py   # 6 CrewAI agents (validation, fleet, optimizer, traffic, dispatch, reporting)
│   ├── tasks/
│   │   └── logistics_tasks.py    # 7 sequential CrewAI tasks + dispatch provenance guardrail
│   ├── crews/
│   │   └── logistics_crew.py     # SmartLogisticsCrew sequential workflow runner
│   └── services/
│       └── orchestration_service.py # LogisticsOrchestrationService facade for workflow execution
└── tests/
    ├── test_health.py            # Phase I dependency import & /health endpoint checks
    ├── test_distance_matrix.py   # Haversine distance & travel-time matrix tests
    ├── test_cvrp.py              # Single/multi-depot CVRP & error handling tests
    ├── test_vrptw.py             # Single/multi-depot VRPTW & infeasibility tests
    ├── test_metrics.py           # Performance metrics & schedule verification tests
    ├── test_llm_config.py        # Gemini env resolution, LLM factory & mock fallback tests
    ├── tools/test_tools.py       # Unit tests for all 7 custom CrewAI tools & provenance check
    ├── agents/test_agents.py     # Unit tests for all 6 CrewAI agents
    ├── tasks/test_tasks.py       # Sequential 7-task DAG & dispatch guardrail tests
    ├── crews/test_crew_orchestration.py # E2E VRPTW/CVRP/disruption/blocking/guardrail tests
    └── services/test_orchestration_service.py # Orchestration service facade tests
```

## Setup (Python 3.11)

### 1. Create and activate the virtual environment

On Windows (PowerShell):

```powershell
python3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) when running live Google Gemini orchestration:

```powershell
Copy-Item .env.example .env
```

Optional LLM configuration environment variables:
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: API key for Google Gemini (`crewai[google-genai]`)
- `GEMINI_MODEL_NAME`: Default `gemini-2.5-flash`
- `LLM_MODE`: `auto` (uses Gemini when a key is present, otherwise deterministic `MockLogisticsLLM`), `gemini`, or `mock`

## Running Health Checks, Tests & Phase III Demo

### Run pytest (offline deterministic mode, live Gemini test auto-skipped without API key)

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### Run Phase III end-to-end synthetic dataset demonstration

```powershell
.\.venv\Scripts\python.exe scripts/run_phase3_demo.py
```

### Start the FastAPI server

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Then check the health endpoint at `http://127.0.0.1:8000/health` (or interactive API docs at `http://127.0.0.1:8000/docs`).

