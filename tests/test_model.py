"""Model contract: scoring direction, prediction shape, explanations, metrics."""

from __future__ import annotations

import json

from astrohealth.model import (
    CLASSES,
    FEATURES,
    METRICS_PATH,
    load_model,
    model_metrics,
    strain_index,
    strain_label,
)

from .conftest import make_series


def test_strain_index_is_monotonic_in_deterioration(nominal_window):
    worse = dict(nominal_window)
    worse.update(
        {
            "hr_mean": 96.0,
            "hr_max": 128.0,
            "hrv_rmssd": 18.0,
            "spo2_min": 93.0,
            "spo2_mean": 94.6,
            "sleep_hours": 4.2,
            "sleep_efficiency": 68.0,
            "stress_self": 8.0,
            "temperature": 37.9,
            "bone_density_pct": 91.0,
            "radiation_7d": 9.0,
        }
    )
    assert strain_index(worse) > strain_index(nominal_window)
    assert 0.0 <= strain_index(nominal_window) <= 100.0
    assert 0.0 <= strain_index(worse) <= 100.0
    assert strain_label(worse) in CLASSES


def test_prediction_returns_a_valid_distribution(nominal_window):
    model = load_model()
    prediction = model.predict(nominal_window)
    assert prediction.label in CLASSES
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6
    assert 0.0 <= prediction.confidence <= 1.0
    assert 0.0 <= prediction.strain_index <= 100.0


def test_explanations_are_ranked_and_bounded(nominal_window):
    model = load_model()
    items = model.explain(nominal_window, baseline=nominal_window, top_k=4)
    assert 0 < len(items) <= 4
    contributions = [item["contribution"] for item in items]
    assert contributions == sorted(contributions, reverse=True)
    for item in items:
        assert item["feature"] in FEATURES
        assert item["label"]
        assert item["direction"] in {"high", "low"}


def test_model_card_is_present_and_honest():
    metrics = model_metrics()
    assert metrics["n_samples"] > 500
    assert metrics["feature_order"] == list(FEATURES)
    assert 0.5 < metrics["stratified_5fold"]["accuracy"] <= 1.0
    assert 0.0 < metrics["leave_one_archetype_out"]["accuracy"] <= 1.0
    assert metrics["training_data"]["kind"] == "synthetic"
    assert metrics["limitations"], "a model card without limitations is a red flag"
    assert METRICS_PATH.exists()


def test_metrics_file_matches_the_runtime_model():
    on_disk = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert model_metrics()["model_version"] == on_disk["model_version"]


def test_prediction_responds_to_a_deteriorating_window(nominal_window):
    model = load_model()
    healthy = model.predict(nominal_window)
    stressed = dict(nominal_window)
    stressed.update({"hr_mean": 98.0, "hrv_rmssd": 16.0, "sleep_hours": 4.0, "stress_self": 9.0, "temperature": 38.2})
    worse = model.predict(stressed)
    assert worse.strain_index > healthy.strain_index


def test_generated_crew_series_have_the_expected_shape():
    from astrohealth.synthetic import build_cohort

    cohort = build_cohort(days=10)
    assert len(cohort) == 6
    for member in cohort:
        assert len(member.series) == 10
        assert member.series[-1]["day"] == member.mission_day
        assert 80 <= member.series[-1]["bone_density_pct"] <= 102
        assert 0 <= member.series[-1]["stress_self"] <= 10
