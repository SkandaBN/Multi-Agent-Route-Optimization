"""Geographical distance and travel-time matrix utilities.

Matrix generation is isolated behind the ``DistanceMatrixProvider`` protocol so
that the default Haversine implementation can be swapped out for real road
distance APIs (e.g., OSRM or Google Routes API) without changing routing solvers.
"""

import math
from collections.abc import Sequence
from typing import Protocol, Union

from app.schemas.order import GeoPoint, validate_latitude, validate_longitude

EARTH_RADIUS_METERS = 6_371_000.0

LocationInput = Union[GeoPoint, tuple[float, float]]


def _extract_coordinates(location: LocationInput) -> tuple[float, float]:
    """Extract and validate (latitude, longitude) from a GeoPoint or tuple."""
    if isinstance(location, GeoPoint):
        return location.latitude, location.longitude
    if isinstance(location, tuple) and len(location) == 2:
        lat, lon = location
        return validate_latitude(float(lat)), validate_longitude(float(lon))
    raise TypeError(
        f"Expected GeoPoint or (latitude, longitude) tuple, got {type(location).__name__}."
    )


def haversine_distance_meters(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate great-circle distance in meters using the Haversine formula."""
    lat1_v = validate_latitude(float(lat1))
    lon1_v = validate_longitude(float(lon1))
    lat2_v = validate_latitude(float(lat2))
    lon2_v = validate_longitude(float(lon2))

    if lat1_v == lat2_v and lon1_v == lon2_v:
        return 0.0

    phi1 = math.radians(lat1_v)
    phi2 = math.radians(lat2_v)
    d_phi = math.radians(lat2_v - lat1_v)
    d_lambda = math.radians(lon2_v - lon1_v)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_METERS * c


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate great-circle distance in kilometers using the Haversine formula."""
    return haversine_distance_meters(lat1, lon1, lat2, lon2) / 1000.0


class DistanceMatrixProvider(Protocol):
    """Interface for distance matrix providers returning integer meter distances."""

    def compute_distance_matrix(self, locations: Sequence[LocationInput]) -> list[list[int]]:
        """Return a square symmetric integer distance matrix in meters."""
        ...


class HaversineDistanceMatrixProvider:
    """Default distance matrix provider using the Haversine formula in meters."""

    def compute_distance_matrix(self, locations: Sequence[LocationInput]) -> list[list[int]]:
        """Build a symmetric integer distance matrix (in meters) for all locations."""
        coords = [_extract_coordinates(loc) for loc in locations]
        n = len(coords)
        matrix = [[0] * n for _ in range(n)]

        for i in range(n):
            lat1, lon1 = coords[i]
            for j in range(i + 1, n):
                lat2, lon2 = coords[j]
                dist_m = int(round(haversine_distance_meters(lat1, lon1, lat2, lon2)))
                matrix[i][j] = dist_m
                matrix[j][i] = dist_m

        return matrix


def build_distance_matrix(
    locations: Sequence[LocationInput],
    provider: DistanceMatrixProvider | None = None,
) -> list[list[int]]:
    """Generate a symmetric distance matrix in meters using the given provider."""
    active_provider = provider or HaversineDistanceMatrixProvider()
    return active_provider.compute_distance_matrix(locations)


def build_distance_matrix_km(locations: Sequence[LocationInput]) -> list[list[float]]:
    """Generate a symmetric distance matrix in kilometers using Haversine distance."""
    coords = [_extract_coordinates(loc) for loc in locations]
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        lat1, lon1 = coords[i]
        for j in range(i + 1, n):
            lat2, lon2 = coords[j]
            dist_km = round(haversine_distance_km(lat1, lon1, lat2, lon2), 4)
            matrix[i][j] = dist_km
            matrix[j][i] = dist_km

    return matrix


def build_travel_time_matrix(
    distance_matrix_meters: Sequence[Sequence[int]],
    speed_kmh: float = 40.0,
) -> list[list[int]]:
    """Convert a meter distance matrix into an integer minute travel-time matrix.

    Uses ceiling rounding for positive distances so distinct locations never
    collapse to zero travel time and the triangle inequality is preserved.
    """
    if not math.isfinite(speed_kmh) or speed_kmh <= 0.0:
        raise ValueError(f"speed_kmh must be a positive finite number, got {speed_kmh}.")

    n = len(distance_matrix_meters)
    for row in distance_matrix_meters:
        if len(row) != n:
            raise ValueError("distance_matrix_meters must be a square matrix.")

    meters_per_minute = (speed_kmh * 1000.0) / 60.0
    time_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            dist_m = distance_matrix_meters[i][j]
            if dist_m < 0:
                raise ValueError(f"Distance matrix cannot contain negative entry ({dist_m}).")
            if i == j or dist_m == 0:
                time_matrix[i][j] = 0
            else:
                time_matrix[i][j] = math.ceil(dist_m / meters_per_minute)

    return time_matrix

