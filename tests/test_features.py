"""Feature extraction must be identical at train time and serve time."""

from __future__ import annotations

from astrohealth.features import FEATURES, as_vector, window_features

from .conftest import make_series


def test_window_features_returns_every_documented_feature(nominal_series):
    features = window_features(nominal_series, mission_day=7)
    assert set(features) == set(FEATURES)


def test_window_features_uses_only_the_trailing_week():
    series = make_series(days=7, heart_rate=60.0)
    series.append({**series[-1], "day": 8, "heart_rate": 90.0})
    features = window_features(series, mission_day=8)
    # The 90 bpm day is inside the window, older 60 bpm days are too.
    assert 60.0 < features["hr_mean"] < 90.0
    assert features["hr_max"] == 90.0

    long_series = make_series(days=10, heart_rate=50.0)
    long_series[-1]["heart_rate"] = 100.0
    recent = window_features(long_series, mission_day=10)
    assert recent["hr_max"] == 100.0
    assert recent["hr_mean"] > 50.0


def test_radiation_and_eva_are_summed_over_the_window(nominal_series):
    features = window_features(nominal_series, mission_day=7)
    assert features["radiation_7d"] == round(0.45 * 7, 2)
    assert features["eva_hours_7d"] == round(0.4 * 7, 2)


def test_vector_projection_follows_feature_order(nominal_window):
    vector = as_vector(nominal_window)
    assert len(vector) == len(FEATURES)
    assert vector[FEATURES.index("hr_mean")] == nominal_window["hr_mean"]


def test_empty_window_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        window_features([], mission_day=1)
