# Multi-Agent Route Optimization for Smart Logistics

Modular Smart Logistics platform combining **FastAPI**, **Pydantic**, **Google OR-Tools**, and **CrewAI** on **Python 3.11**.

> **Status:** Phase I (scaffolding & FastAPI health check) and Phase II (Google OR-Tools CVRP/VRPTW optimization engine, route extraction, and performance metrics) are implemented and tested. CrewAI agents, tasks, crews, and LLM orchestration are not implemented yet.

## Project Structure

```text
.
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── .python-version           # Target Python version (3.11)
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Pinned Python dependencies
├── README.md
├── data/
│   ├── customers.csv         # 25 synthetic customers with coordinates, demand, and time windows
│   ├── depots.csv            # 2 synthetic distribution depots
│   └── vehicles.csv          # 5 synthetic vehicles with capacities and depot assignments
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI application factory & entrypoint
│   ├── api/
│   │   └── routes/
│   │       └── health.py     # GET /health endpoint
│   ├── core/
│   │   └── config.py         # Pydantic BaseSettings configuration
│   ├── schemas/
│   │   ├── health.py         # Health response model
│   │   ├── order.py          # GeoPoint, Depot, and CustomerOrder schemas
│   │   ├── vehicle.py        # Vehicle schema
│   │   └── route.py          # RouteStop, VehicleRoute, PerformanceMetrics, OptimizationResult
│   ├── optimization/
│   │   ├── __init__.py       # Exceptions & public exports
│   │   ├── distance_matrix.py# Pluggable DistanceMatrixProvider & Haversine/travel-time matrices
│   │   ├── cvrp.py           # Single-depot & multi-depot CVRP solver + CSV dataset loader
│   │   ├── vrptw.py          # Single-depot & multi-depot VRPTW solver
│   │   ├── route_extractor.py# Structured route extraction from OR-Tools solutions
│   │   └── metrics.py        # Fleet & schedule performance metrics
│   ├── agents/               # Future CrewAI agent definitions
│   ├── tasks/                # Future CrewAI task definitions
│   ├── crews/                # Future CrewAI crew orchestrators
│   ├── tools/                # Future custom CrewAI tools
│   └── services/             # Future business/orchestration services
└── tests/
    ├── test_health.py        # Phase I dependency import & /health endpoint checks
    ├── test_distance_matrix.py # Haversine distance & travel-time matrix tests
    ├── test_cvrp.py          # Single/multi-depot CVRP & error handling tests
    ├── test_vrptw.py         # Single/multi-depot VRPTW & infeasibility tests
    └── test_metrics.py       # Performance metrics & schedule verification tests
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

Copy `.env.example` to `.env` and fill in your credentials before adding LLM-backed agents:

```powershell
Copy-Item .env.example .env
```

## Running the Health Check & Tests

### Run pytest

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### Start the FastAPI server

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Then check the health endpoint at `http://127.0.0.1:8000/health` (or interactive API docs at `http://127.0.0.1:8000/docs`).

