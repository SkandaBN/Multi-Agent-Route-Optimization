"""Pydantic schemas for geographical locations, depots, and customer delivery orders."""

import math

from pydantic import BaseModel, Field, field_validator, model_validator


def validate_latitude(value: float) -> float:
    """Validate that latitude is finite and within [-90, 90] degrees."""
    if not math.isfinite(value) or not -90.0 <= value <= 90.0:
        raise ValueError(f"Invalid latitude {value}: must be finite and between -90 and 90.")
    return float(value)


def validate_longitude(value: float) -> float:
    """Validate that longitude is finite and within [-180, 180] degrees."""
    if not math.isfinite(value) or not -180.0 <= value <= 180.0:
        raise ValueError(f"Invalid longitude {value}: must be finite and between -180 and 180.")
    return float(value)


def _strip_non_empty(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return cleaned


class GeoPoint(BaseModel):
    """Base model representing a validated geographic coordinate pair."""

    latitude: float
    longitude: float

    @field_validator("latitude")
    @classmethod
    def _check_latitude(cls, value: float) -> float:
        return validate_latitude(value)

    @field_validator("longitude")
    @classmethod
    def _check_longitude(cls, value: float) -> float:
        return validate_longitude(value)


class Depot(GeoPoint):
    """Distribution depot where vehicle routes begin and end."""

    depot_id: str
    name: str = ""
    time_window_start: int = Field(default=0, ge=0)
    time_window_end: int | None = Field(default=None, ge=0)

    @field_validator("depot_id")
    @classmethod
    def _check_depot_id(cls, value: str) -> str:
        return _strip_non_empty(value, "depot_id")

    @model_validator(mode="after")
    def _check_time_window(self) -> "Depot":
        if self.time_window_end is not None and self.time_window_end < self.time_window_start:
            raise ValueError(
                f"Depot {self.depot_id}: time_window_end ({self.time_window_end}) "
                f"must be >= time_window_start ({self.time_window_start})."
            )
        return self


class CustomerOrder(GeoPoint):
    """Customer delivery order with demand, service duration, and optional time window."""

    customer_id: str
    demand: int = Field(..., ge=0)
    service_duration_minutes: int = Field(default=0, ge=0)
    time_window_start: int = Field(default=0, ge=0)
    time_window_end: int | None = Field(default=None, ge=0)

    @field_validator("customer_id")
    @classmethod
    def _check_customer_id(cls, value: str) -> str:
        return _strip_non_empty(value, "customer_id")

    @model_validator(mode="after")
    def _check_time_window(self) -> "CustomerOrder":
        if self.time_window_end is not None and self.time_window_end < self.time_window_start:
            raise ValueError(
                f"Customer {self.customer_id}: time_window_end ({self.time_window_end}) "
                f"must be >= time_window_start ({self.time_window_start})."
            )
        return self


Order = CustomerOrder

