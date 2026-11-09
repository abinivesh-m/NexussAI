"""Detector behaviour: one test per failure mode the detectors are meant to catch."""

from __future__ import annotations

from astrohealth.detectors import detect, symptom_score
from astrohealth.features import window_features

from .conftest import make_day, make_series


def _alerts(series):
    features = window_features(series, mission_day=series[-1]["day"])
    return detect(series, features)


def test_nominal_crew_member_produces_no_alerts(nominal_series):
    assert _alerts(nominal_series) == []


def test_tachycardia_is_critical(nominal_series):
    alerts = _alerts(make_series(heart_rate=105.0))
    match = [a for a in alerts if a["key"] == "tachycardia"]
    assert match and match[0]["severity"] == "critical"
    assert match[0]["unit"] == "bpm"


def test_hypoxia_is_critical_and_flags_the_respiratory_system():
    alerts = _alerts(make_series(spo2=90.0))
    match = [a for a in alerts if a["key"] == "hypoxia"]
    assert match and match[0]["severity"] == "critical"
    assert match[0]["system"] == "resp"


def test_sleep_debt_is_detected_from_the_weekly_mean():
    alerts = _alerts(make_series(sleep_hours=4.8))
    assert any(a["key"] == "sleep_debt" for a in alerts)


def test_rising_heart_rate_is_caught_by_the_trend_detector():
    series = [make_day(day=i + 1, heart_rate=60.0 + 2.5 * i) for i in range(7)]
    alerts = _alerts(series)
    trend = [a for a in alerts if a["detector"] == "trend" and a["system"] == "cardio"]
    assert trend, "a 15 bpm rise over a week must raise a trend alert"


def test_alerts_are_ordered_by_severity(nominal_series):
    series = make_series(heart_rate=108.0, spo2=90.0, temperature=38.4)
    alerts = _alerts(series)
    severities = [a["severity"] for a in alerts]
    order = {"critical": 0, "warning": 1, "info": 2}
    assert severities == sorted(severities, key=lambda s: order[s])


def test_every_alert_carries_evidence_and_a_recommendation(nominal_series):
    alerts = _alerts(make_series(heart_rate=112.0, temperature=38.2))
    assert alerts
    for alert in alerts:
        assert alert["detail"] and alert["recommendation"]
        assert alert["unit"] and alert["detector"] in {"rule", "trend", "baseline", "novelty", "model"}


def test_symptom_score_weights_by_clinical_severity():
    score = symptom_score({"chest_tightness": 3, "headache": 1})
    assert score["symptom_load"] == round(2.2 * 3 + 0.6, 2)
    assert score["systems"]["cardio"] == round(2.2 * 3, 2)
    assert score["reported"][0]["key"] == "chest_tightness"


def test_symptom_score_ignores_zero_severities():
    assert symptom_score({"headache": 0})["symptom_load"] == 0.0
