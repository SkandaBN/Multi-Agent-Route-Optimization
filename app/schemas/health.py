"""Pydantic schemas for service health check responses."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response payload returned by the health check endpoint."""

    status: str
    app_name: str
    version: str
    environment: str
    python_version: str

