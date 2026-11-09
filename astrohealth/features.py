"""Feature extraction shared by training and inference.

A *day record* is one dict of aggregated signals for a single mission day.
A *window* is the trailing seven day records for one crew member.

Sharing one implementation between the training script and the API is
deliberate: train/serve skew is the most common way a demo model quietly
stops matching its own metrics.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

#: Ordered feature vector. The order is part of the model contract and is
#: written into the persisted model card.
FEATURES: tuple[str, ...] = (
    "hr_mean",
    "hr_max",
    "hrv_rmssd",
    "spo2_mean",
    "spo2_min",
    "bp_systolic",
    "bp_diastolic",
    "temperature",
    "sleep_hours",
    "sleep_efficiency",
    "stress_self",
    "bone_density_pct",
    "radiation_7d",
    "eva_hours_7d",
    "fluid_intake",
    "mission_day",
)

#: Human-readable labels for the UI and the explanation panel.
FEATURE_LABELS: dict[str, str] = {
    "hr_mean": "Mean heart rate",
    "hr_max": "Peak heart rate",
    "hrv_rmssd": "Heart-rate variability",
    "spo2_mean": "Mean SpO₂",
    "spo2_min": "Minimum SpO₂",
    "bp_systolic": "Systolic BP",
    "bp_diastolic": "Diastolic BP",
    "temperature": "Core temperature",
    "sleep_hours": "Sleep duration",
    "sleep_efficiency": "Sleep efficiency",
    "stress_self": "Self-reported stress",
    "bone_density_pct": "Bone density vs baseline",
    "radiation_7d": "Radiation dose (7 d)",
    "eva_hours_7d": "EVA hours (7 d)",
    "fluid_intake": "Fluid intake",
    "mission_day": "Mission day",
}

WINDOW_DAYS = 7


def window_features(days: Sequence[dict[str, Any]], mission_day: int) -> dict[str, float]:
    """Aggregate a trailing window of day records into the model feature dict."""
    if not days:
        raise ValueError("window_features requires at least one day record")

    window = list(days)[-WINDOW_DAYS:]

    def mean(key: str) -> float:
        return sum(float(d[key]) for d in window) / len(window)

    def total(key: str) -> float:
        return sum(float(d[key]) for d in window)

    return {
        "hr_mean": round(mean("heart_rate"), 2),
        "hr_max": round(max(float(d["heart_rate"]) for d in window), 2),
        "hrv_rmssd": round(mean("hrv_rmssd"), 2),
        "spo2_mean": round(mean("spo2"), 2),
        "spo2_min": round(min(float(d["spo2"]) for d in window), 2),
        "bp_systolic": round(mean("bp_systolic"), 2),
        "bp_diastolic": round(mean("bp_diastolic"), 2),
        "temperature": round(mean("temperature"), 2),
        "sleep_hours": round(mean("sleep_hours"), 2),
        "sleep_efficiency": round(mean("sleep_efficiency"), 2),
        "stress_self": round(mean("stress_self"), 2),
        "bone_density_pct": round(window[-1]["bone_density_pct"], 2),
        "radiation_7d": round(total("radiation_msv"), 2),
        "eva_hours_7d": round(total("eva_hours"), 2),
        "fluid_intake": round(mean("fluid_intake"), 2),
        "mission_day": float(mission_day),
    }


def as_vector(features: dict[str, float], order: Iterable[str] = FEATURES) -> list[float]:
    """Project a feature dict onto the ordered model vector."""
    return [float(features[key]) for key in order]
