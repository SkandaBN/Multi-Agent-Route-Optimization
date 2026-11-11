"""Unit tests for Haversine distance and travel-time matrix calculations."""

import math

import pytest

from app.optimization.distance_matrix import (
    DistanceMatrixProvider,
    HaversineDistanceMatrixProvider,
    build_distance_matrix,
    build_distance_matrix_km,
    build_travel_time_matrix,
    haversine_distance_km,
    haversine_distance_meters,
)
from app.schemas.order import GeoPoint


def test_haversine_identical_points_is_zero() -> None:
    """Identical coordinates must have zero distance."""
    assert haversine_distance_meters(12.9716, 77.5946, 12.9716, 77.5946) == 0.0
    assert haversine_distance_km(12.9716, 77.5946, 12.9716, 77.5946) == 0.0


def test_haversine_known_distance() -> None:
    """One degree of latitude at the equator is approximately 111.195 km."""
    dist_km = haversine_distance_km(0.0, 0.0, 1.0, 0.0)
    assert math.isclose(dist_km, 111.195, rel_tol=1e-3)


def test_distance_matrix_symmetry_and_zero_diagonal() -> None:
    """Distance matrices in meters and km must be square, symmetric, and zero on diagonal."""
    locations = [
        GeoPoint(latitude=12.9716, longitude=77.5946),
        GeoPoint(latitude=12.9352, longitude=77.6245),
        (13.0358, 77.5970),
    ]
    mat_m = build_distance_matrix(locations)
    mat_km = build_distance_matrix_km(locations)

    assert len(mat_m) == 3
    assert len(mat_km) == 3
    for i in range(3):
        assert mat_m[i][i] == 0
        assert mat_km[i][i] == 0.0
        for j in range(3):
            assert mat_m[i][j] == mat_m[j][i]
            assert mat_km[i][j] == mat_km[j][i]
            if i != j:
                assert mat_m[i][j] > 0
                assert math.isclose(mat_m[i][j] / 1000.0, mat_km[i][j], abs_tol=0.01)


def test_custom_distance_provider_pluggable() -> None:
    """A custom road-distance provider can be substituted cleanly."""

    class MockRoadProvider:
        def compute_distance_matrix(self, locations):
            n = len(locations)
            return [[0 if i == j else 5000 for j in range(n)] for i in range(n)]

    provider: DistanceMatrixProvider = MockRoadProvider()
    mat = build_distance_matrix([(12.9, 77.5), (12.95, 77.6)], provider=provider)
    assert mat == [[0, 5000], [5000, 0]]


def test_invalid_coordinates_raise_value_error() -> None:
    """Out-of-range or non-finite coordinates must be rejected."""
    with pytest.raises(ValueError):
        haversine_distance_meters(91.0, 77.5, 12.9, 77.5)

    with pytest.raises(ValueError):
        haversine_distance_meters(12.9, -181.0, 12.9, 77.5)

    with pytest.raises(ValueError):
        build_distance_matrix([(float("nan"), 77.5), (12.9, 77.5)])


def test_travel_time_matrix_calculation() -> None:
    """Travel-time matrix converts meters to ceiling minutes symmetrically."""
    dist_m = [
        [0, 20_000, 1_000],
        [20_000, 0, 15_000],
        [1_000, 15_000, 0],
    ]
    # At 60 km/h (1000 m/min): 20_000 m = 20 min, 1_000 m = 1 min, 15_000 m = 15 min
    times = build_travel_time_matrix(dist_m, speed_kmh=60.0)
    assert times == [
        [0, 20, 1],
        [20, 0, 15],
        [1, 15, 0],
    ]


def test_travel_time_matrix_rejects_invalid_inputs() -> None:
    """Invalid speeds, non-square matrices, or negative distances must raise ValueError."""
    with pytest.raises(ValueError):
        build_travel_time_matrix([[0, 1000], [1000, 0]], speed_kmh=0.0)

    with pytest.raises(ValueError):
        build_travel_time_matrix([[0, 1000]], speed_kmh=40.0)

    with pytest.raises(ValueError):
        build_travel_time_matrix([[0, -5], [-5, 0]], speed_kmh=40.0)

