"""Pydantic schema for fleet vehicle configuration."""

from pydantic import BaseModel, Field, field_validator, model_validator


class Vehicle(BaseModel):
    """Delivery vehicle with capacity, depot assignment, and optional route duration limit."""

    vehicle_id: str
    capacity: int = Field(..., gt=0)
    start_depot_id: str
    end_depot_id: str | None = None
    max_route_duration_minutes: int | None = Field(default=None, gt=0)
    speed_kmh: float = Field(default=40.0, gt=0.0)

    @field_validator("vehicle_id", "start_depot_id")
    @classmethod
    def _check_required_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Identifier must be a non-empty string.")
        return cleaned

    @model_validator(mode="after")
    def _resolve_end_depot(self) -> "Vehicle":
        if self.end_depot_id is None or not self.end_depot_id.strip():
            self.end_depot_id = self.start_depot_id
        else:
            self.end_depot_id = self.end_depot_id.strip()
        return self

