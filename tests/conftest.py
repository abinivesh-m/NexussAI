"""Shared fixtures for the AstroHealth test suite."""

from __future__ import annotations

from typing import Any

import pytest

NOMINAL: dict[str, float] = {
    "heart_rate": 62.0,
    "hrv_rmssd": 58.0,
    "spo2": 98.0,
    "bp_systolic": 116.0,
    "bp_diastolic": 74.0,
    "temperature": 36.6,
    "sleep_hours": 7.4,
    "sleep_efficiency": 90.0,
    "stress_self": 2.0,
    "bone_density_pct": 98.5,
    "eva_hours": 0.4,
    "radiation_msv": 0.45,
    "fluid_intake": 2.8,
    "episode": 0.0,
}


def make_day(day: int = 1, **overrides: float) -> dict[str, Any]:
    record = {**NOMINAL, **overrides}
    record["day"] = day
    record["date"] = f"2026-03-{day:02d}"
    return record


def make_series(days: int = 7, **overrides: float) -> list[dict[str, Any]]:
    return [make_day(day=i + 1, **overrides) for i in range(days)]


@pytest.fixture
def nominal_series() -> list[dict[str, Any]]:
    return make_series()


@pytest.fixture
def nominal_window(nominal_series: list[dict[str, Any]]) -> dict[str, float]:
    from astrohealth.features import window_features

    return window_features(nominal_series, mission_day=7)
