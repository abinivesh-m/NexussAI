"""Alert generation: threshold rules, trends, personal baselines and novelty.

Four independent detectors run on every crew member. They are deliberately
separate because they fail in different ways:

``rule``
    Absolute bands. Cheap, interpretable, blind to gradual drift.
``trend``
    Slope over the last five days. Catches drift before a band is crossed.
``baseline``
    Deviation from that crew member's own first week. Handles the fact that a
    resting heart rate of 78 is unremarkable for one astronaut and a 24 bpm
    rise for another.
``novelty``
    Isolation-forest score. Catches patterns nobody wrote a rule for.

Every alert carries its evidence, so a crew member can see *why* it fired
rather than being asked to trust a colour.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .physiology import SYMPTOMS, VITALS

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _trend(values: Sequence[float]) -> float:
    """Least-squares slope per day."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _alert(
    key: str,
    title: str,
    severity: str,
    detail: str,
    recommendation: str,
    value: float,
    unit: str,
    detector: str,
    confidence: float = 0.9,
    system: str = "general",
) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "severity": severity,
        "detail": detail,
        "recommendation": recommendation,
        "value": round(float(value), 2),
        "unit": unit,
        "detector": detector,
        "confidence": round(float(confidence), 2),
        "system": system,
    }


def detect(
    series: Sequence[dict[str, Any]],
    features: dict[str, float],
    prediction: Any | None = None,
) -> list[dict[str, Any]]:
    """Run all detectors and return alerts ordered by severity."""
    if not series:
        return []

    latest = series[-1]
    window = list(series[-7:])
    alerts: list[dict[str, Any]] = []

    # ---------------- absolute bands ----------------
    hr = float(latest["heart_rate"])
    if hr >= 100:
        alerts.append(_alert("tachycardia", "Tachycardia", "critical",
                             f"Resting heart rate {hr:.0f} bpm is above the 90 bpm operational band.",
                             "Rest, rehydrate, repeat in 15 min. Escalate to flight surgeon if it persists.",
                             hr, "bpm", "rule", 0.95, "cardio"))
    elif hr > 90:
        alerts.append(_alert("elevated_hr", "Elevated heart rate", "warning",
                             f"Resting heart rate {hr:.0f} bpm is above the 90 bpm operational band.",
                             "Increase rest period and fluid intake; recheck after one sleep cycle.",
                             hr, "bpm", "rule", 0.8, "cardio"))
    elif hr < 45:
        alerts.append(_alert("bradycardia", "Bradycardia", "warning",
                             f"Resting heart rate {hr:.0f} bpm is unusually low.",
                             "Confirm sensor placement; check for symptoms of dizziness or presyncope.",
                             hr, "bpm", "rule", 0.75, "cardio"))

    spo2 = float(latest["spo2"])
    if spo2 < 92:
        alerts.append(_alert("hypoxia", "Hypoxaemia", "critical",
                             f"SpO₂ {spo2:.0f}% is below the 95% operational floor.",
                             "Start oxygen supplementation and notify mission control.",
                             spo2, "%", "rule", 0.97, "resp"))
    elif spo2 < 95:
        alerts.append(_alert("low_spo2", "Low oxygen saturation", "warning",
                             f"SpO₂ {spo2:.0f}% is below the 95% operational floor.",
                             "Rest, check cabin CO₂, repeat measurement.",
                             spo2, "%", "rule", 0.85, "resp"))

    sys_bp = float(latest["bp_systolic"])
    dia_bp = float(latest["bp_diastolic"])
    if sys_bp >= 140:
        alerts.append(_alert("hypertension", "Hypertensive range", "critical",
                             f"Blood pressure {sys_bp:.0f}/{dia_bp:.0f} mmHg.",
                             "Rest, reduce exertion and sodium. Notify flight surgeon.",
                             sys_bp, "mmHg", "rule", 0.9, "cardio"))
    elif sys_bp > 130:
        alerts.append(_alert("elevated_bp", "Elevated blood pressure", "warning",
                             f"Blood pressure {sys_bp:.0f}/{dia_bp:.0f} mmHg.",
                             "Reduce physical exertion for the next work block.",
                             sys_bp, "mmHg", "rule", 0.75, "cardio"))
    elif sys_bp < 95:
        alerts.append(_alert("hypotension", "Low blood pressure", "warning",
                             f"Blood pressure {sys_bp:.0f}/{dia_bp:.0f} mmHg.",
                             "Oral fluids, slow position changes, assess orthostatic tolerance.",
                             sys_bp, "mmHg", "rule", 0.8, "cardio"))

    temp = float(latest["temperature"])
    if temp >= 38.0:
        alerts.append(_alert("fever", "Fever", "critical",
                             f"Core temperature {temp:.1f} °C.",
                             "Begin infection workup, isolate if indicated, notify flight surgeon.",
                             temp, "°C", "rule", 0.95, "immune"))
    elif temp > 37.4:
        alerts.append(_alert("low_grade_fever", "Low-grade fever", "warning",
                             f"Core temperature {temp:.1f} °C.",
                             "Increase fluids; recheck in 4 h; watch for other symptoms.",
                             temp, "°C", "rule", 0.8, "immune"))

    sleep_mean = float(np.mean([d["sleep_hours"] for d in window]))
    if sleep_mean < 5.5:
        alerts.append(_alert("sleep_debt", "Significant sleep debt", "critical",
                             f"Seven-day mean sleep {sleep_mean:.1f} h against a 7 h target.",
                             "Protected sleep period; defer non-critical tasks; review workload.",
                             sleep_mean, "h", "rule", 0.9, "sleep"))
    elif sleep_mean < 6.5:
        alerts.append(_alert("poor_sleep", "Insufficient sleep", "warning",
                             f"Seven-day mean sleep {sleep_mean:.1f} h against a 7 h target.",
                             "Protect the next sleep window; reduce evening light and caffeine.",
                             sleep_mean, "h", "rule", 0.8, "sleep"))

    eff = float(np.mean([d["sleep_efficiency"] for d in window]))
    if eff < 75:
        alerts.append(_alert("sleep_quality", "Fragmented sleep", "warning",
                             f"Seven-day mean sleep efficiency {eff:.0f}%.",
                             "Sleep-hygiene review; consider actigraphy check.",
                             eff, "%", "rule", 0.7, "sleep"))

    stress = float(np.mean([d["stress_self"] for d in window]))
    if stress >= 7.5:
        alerts.append(_alert("high_stress", "High psychological load", "critical",
                             f"Seven-day mean self-rated stress {stress:.1f}/10.",
                             "Behavioural health check-in; reduce task load; notify flight surgeon.",
                             stress, "/10", "rule", 0.85, "behavioural"))
    elif stress >= 6:
        alerts.append(_alert("stress", "Elevated psychological load", "warning",
                             f"Seven-day mean self-rated stress {stress:.1f}/10.",
                             "Scheduled conference with crew support; protect recovery time.",
                             stress, "/10", "rule", 0.75, "behavioural"))

    bone = float(latest["bone_density_pct"])
    if bone < 90:
        alerts.append(_alert("bone_loss_severe", "Accelerated bone loss", "critical",
                             f"Bone mineral density {bone:.1f}% of preflight baseline.",
                             "Escalate loading-countermeasure prescription; review resistance protocol.",
                             bone, "% baseline", "rule", 0.85, "musculoskeletal"))
    elif bone < 94:
        alerts.append(_alert("bone_loss", "Bone density declining", "warning",
                             f"Bone mineral density {bone:.1f}% of preflight baseline.",
                             "Increase resistance loading; confirm calcium and vitamin D intake.",
                             bone, "% baseline", "rule", 0.8, "musculoskeletal"))

    rad_7d = float(features["radiation_7d"])
    if rad_7d > 8:
        alerts.append(_alert("radiation_high", "Radiation dose watch", "critical",
                             f"Trailing seven-day effective dose {rad_7d:.1f} mSv.",
                             "Review EVA schedule and shielding; log dose against career limit.",
                             rad_7d, "mSv/7d", "rule", 0.85, "radiation"))
    elif rad_7d > 5:
        alerts.append(_alert("radiation", "Elevated radiation dose", "warning",
                             f"Trailing seven-day effective dose {rad_7d:.1f} mSv.",
                             "Minimise non-essential EVA hours; monitor cumulative dose.",
                             rad_7d, "mSv/7d", "rule", 0.75, "radiation"))

    fluid = float(np.mean([d["fluid_intake"] for d in window]))
    if fluid < 1.8:
        alerts.append(_alert("dehydration", "Low fluid intake", "warning",
                             f"Seven-day mean fluid intake {fluid:.1f} L/day.",
                             "Structured drinking schedule; check for orthostatic symptoms.",
                             fluid, "L/day", "rule", 0.7, "hydration"))

    hrv_mean = float(np.mean([d["hrv_rmssd"] for d in window]))
    if hrv_mean < 25:
        alerts.append(_alert("hrv_suppressed", "Suppressed heart-rate variability", "warning",
                             f"Seven-day mean RMSSD {hrv_mean:.0f} ms.",
                             "Treat as a recovery deficit: sleep, fuelling and workload review.",
                             hrv_mean, "ms", "rule", 0.65, "autonomic"))

    # ---------------- trends ----------------
    hr_slope = _trend([d["heart_rate"] for d in window])
    if hr_slope > 1.2:
        alerts.append(_alert("hr_upward_trend", "Heart rate trending up", "warning",
                             f"Heart rate rising {hr_slope:.1f} bpm/day across the last seven days.",
                             "Look for a developing cause rather than treating the number.",
                             hr_slope, "bpm/day", "trend", 0.7, "cardio"))

    spo2_slope = _trend([d["spo2"] for d in window])
    if spo2_slope < -0.25:
        alerts.append(_alert("spo2_downward_trend", "Oxygen saturation drifting down", "warning",
                             f"SpO₂ falling {abs(spo2_slope):.2f} %/day.",
                             "Check cabin atmosphere, then respiratory status.",
                             spo2_slope, "%/day", "trend", 0.7, "resp"))

    sleep_slope = _trend([d["sleep_hours"] for d in window])
    if sleep_slope < -0.15:
        alerts.append(_alert("sleep_downward_trend", "Sleep duration falling", "info",
                             f"Sleep falling {abs(sleep_slope):.2f} h/day.",
                             "Act on sleep hygiene before debt accumulates.",
                             sleep_slope, "h/day", "trend", 0.6, "sleep"))

    bone_slope = _trend([d["bone_density_pct"] for d in list(series)[-14:]])
    if bone_slope < -0.06:
        alerts.append(_alert("bone_downward_trend", "Bone loss faster than expected", "warning",
                             f"Bone density falling {abs(bone_slope):.2f} %/day.",
                             "Review loading countermeasure adherence with the crew.",
                             bone_slope, "%/day", "trend", 0.65, "musculoskeletal"))

    # ---------------- personal baseline ----------------
    baseline = list(series)[:7]
    if len(baseline) >= 5:
        base_hr = float(np.mean([d["heart_rate"] for d in baseline]))
        delta = hr - base_hr
        if delta > 12:
            alerts.append(_alert("hr_vs_baseline", "Heart rate well above personal baseline", "warning",
                                 f"{hr:.0f} bpm is {delta:+.0f} bpm against this crew member's first-week mean.",
                                 "Personal-baseline drift is more informative than the population band here.",
                                 delta, "bpm", "baseline", 0.8, "cardio"))
        base_sleep = float(np.mean([d["sleep_hours"] for d in baseline]))
        d_sleep = sleep_mean - base_sleep
        if d_sleep < -1.2:
            alerts.append(_alert("sleep_vs_baseline", "Sleep below personal baseline", "info",
                                 f"{abs(d_sleep):.1f} h/night less than this crew member's own first week.",
                                 "Recovery is relative: this is a meaningful drop for them specifically.",
                                 d_sleep, "h", "baseline", 0.65, "sleep"))

    # ---------------- model ----------------
    if prediction is not None:
        if getattr(prediction, "is_novel", False):
            alerts.append(_alert("novel_pattern", "Unclassified deviation from normal patterns", "warning",
                                 "The novelty model found a combination of signals outside the "
                                 "distribution of normal crew windows.",
                                 "Review the contributing signals; this is the pattern no rule anticipated.",
                                 float(getattr(prediction, "anomaly_score", 0.0)), "score", "novelty", 0.6, "general"))
        if getattr(prediction, "label", "") == "high" and float(getattr(prediction, "confidence", 0)) >= 0.55:
            alerts.append(_alert("composite_strain", "High composite physiological strain", "warning",
                                 f"Strain model predicts high strain with {float(prediction.confidence) * 100:.0f}% confidence.",
                                 "Treat as a whole-crew signal: sleep, fuelling, workload and symptoms together.",
                                 float(prediction.strain_index), "/100", "model", float(prediction.confidence), "general"))

    # ---------------- rank ----------------
    for alert in alerts:
        alert["severity_rank"] = SEVERITY_ORDER.get(alert["severity"], 9)
    alerts.sort(key=lambda a: (a["severity_rank"], -a["confidence"]))
    for alert in alerts:
        alert.pop("severity_rank", None)
    return alerts


def symptom_score(symptoms: dict[str, int]) -> dict[str, Any]:
    """Turn a self-reported symptom checklist into a triage signal.

    ``symptoms`` maps a key from :data:`astrohealth.physiology.SYMPTOMS` to a
    severity in 0..3.
    """
    total = 0.0
    systems: dict[str, float] = {}
    reported: list[dict[str, Any]] = []
    for key, severity in symptoms.items():
        spec = SYMPTOMS.get(key)
        if not spec or severity <= 0:
            continue
        weight = float(spec["weight"])  # type: ignore[arg-type]
        contribution = weight * float(severity)
        total += contribution
        for system in spec["systems"]:  # type: ignore[union-attr]
            systems[system] = systems.get(system, 0.0) + contribution
        reported.append(
            {
                "key": key,
                "label": spec["label"],
                "severity": severity,
                "contribution": round(contribution, 2),
                "systems": spec["systems"],
            }
        )
    reported.sort(key=lambda r: r["contribution"], reverse=True)
    return {
        "symptom_load": round(total, 2),
        "systems": {k: round(v, 2) for k, v in systems.items()},
        "reported": reported,
    }
