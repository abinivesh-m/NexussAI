"""Composite risk, drivers, recommendations and EVA readiness gating.

Three inputs are fused:

* **objective** — strain-model class probabilities blended with the continuous
  strain index,
* **burden** — severity-weighted count of active alerts,
* **subjective** — the crew member's own check-in.

Keeping them separate until the last step matters. When a crew member says
"something is wrong" and the sensors say "nominal", the disagreement is the
finding. Flattening everything into one number too early hides exactly the
signal a flight surgeon needs.
"""

from __future__ import annotations

from typing import Any, Sequence

SEVERITY_WEIGHT = {"critical": 22.0, "warning": 9.0, "info": 3.0}

#: Systems that block EVA clearance outright when active at warning or above.
EVA_BLOCKING_SYSTEMS = {"cardio", "resp", "neuro", "ocular"}

#: Composite-score band edges. These are operating thresholds on a 0-100 index,
#: calibrated so that the nominal synthetic cohort sits in the low band and a
#: genuine deterioration reaches the high band. The badge and the number are
#: always derived from the same value, so a crew member can never read
#: "5.9 / moderate".
MODERATE_BAND = 25.0
HIGH_BAND = 55.0

#: A critical finding is high strain by definition; floor the score into the
#: high band so a critical alert can never be sorted below a quieter crew member.
CRITICAL_FLOOR = 68.0


def assess(
    features: dict[str, float],
    prediction: Any,
    alerts: Sequence[dict[str, Any]],
    symptom_load: float = 0.0,
    symptom_systems: dict[str, float] | None = None,
    sleep_quality: int | None = None,
    workload: int | None = None,
    has_self_report: bool = False,
) -> dict[str, Any]:
    probabilities = getattr(prediction, "probabilities", {}) or {}
    high_p = float(probabilities.get("high", 0.0))
    mod_p = float(probabilities.get("moderate", 0.0))
    model_pressure = 100.0 * (high_p + 0.45 * mod_p)
    strain = float(getattr(prediction, "strain_index", 0.0))
    objective = 0.62 * model_pressure + 0.38 * strain

    burden = 0.0
    for alert in alerts:
        weight = SEVERITY_WEIGHT.get(alert["severity"], 2.0)
        burden += weight * float(alert.get("confidence", 0.7))
    burden = min(100.0, burden)

    subjective = min(100.0, symptom_load * 6.5)
    if sleep_quality is not None:
        subjective += max(0.0, (3 - sleep_quality)) * 7.0
    if workload is not None:
        subjective += max(0.0, (workload - 3)) * 4.0
    subjective = min(100.0, subjective)

    # Novelty adds a floor: an unexplained pattern should never read as zero.
    novelty_floor = 18.0 if getattr(prediction, "is_novel", False) else 0.0

    overall = 0.52 * objective + 0.26 * burden + 0.22 * subjective
    overall = round(min(100.0, max(overall, novelty_floor)), 1)

    # A critical finding means high strain by definition. Without this floor a
    # critical alert could sit under a moderate score and the board would sort
    # the sickest crew member second.
    if any(a["severity"] == "critical" for a in alerts):
        overall = max(overall, CRITICAL_FLOOR)

    # The level is a band on the score, nothing else. Alert severity already
    # reaches the score through the burden component, so deriving the level from
    # the alerts as well would double-count and let a single mild warning label
    # a 5.9 as "moderate".
    if overall >= HIGH_BAND:
        level = "high"
    elif overall >= MODERATE_BAND:
        level = "moderate"
    else:
        level = "low"

    systems = dict(symptom_systems or {})
    for alert in alerts:
        systems[alert.get("system", "general")] = systems.get(alert.get("system", "general"), 0.0) + SEVERITY_WEIGHT.get(
            alert["severity"], 2.0
        )

    drivers = _drivers(alerts, systems)
    readiness = _readiness(level, alerts, systems, objective, subjective)
    actions = _actions(level, drivers, systems, readiness)

    return {
        "overall": overall,
        "level": level,
        "components": {
            "objective": round(objective, 1),
            "alert_burden": round(burden, 1),
            "subjective": round(subjective, 1),
            "novelty_floor": novelty_floor,
        },
        "disagreement": _disagreement(objective, subjective) if has_self_report else None,
        "drivers": drivers,
        "systems": {k: round(v, 1) for k, v in sorted(systems.items(), key=lambda kv: kv[1], reverse=True)},
        "readiness": readiness,
        "actions": actions,
    }


def _drivers(alerts: Sequence[dict[str, Any]], systems: dict[str, float]) -> list[dict[str, Any]]:
    ordered = sorted(systems.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return [{"system": name, "weight": round(value, 1)} for name, value in ordered if value > 0]


def _disagreement(objective: float, subjective: float) -> dict[str, Any] | None:
    gap = subjective - objective
    if gap >= 25:
        return {
            "kind": "crew_reports_more_than_sensors",
            "magnitude": round(gap, 1),
            "note": "The crew member reports materially worse than the sensors measure. "
            "Treat the report as valid data, not as noise.",
        }
    if gap <= -25:
        return {
            "kind": "sensors_report_more_than_crew",
            "magnitude": round(gap, 1),
            "note": "Sensors show more strain than the crew member reports. "
            "Check for under-reporting, fatigue or sensor artefact.",
        }
    return None


def _readiness(
    level: str,
    alerts: Sequence[dict[str, Any]],
    systems: dict[str, float],
    objective: float,
    subjective: float,
) -> dict[str, Any]:
    blocking: list[str] = []
    if level == "high":
        blocking.append("composite risk is high")
    for alert in alerts:
        if alert["severity"] == "critical":
            blocking.append(alert["title"])
        elif alert.get("system") in EVA_BLOCKING_SYSTEMS and alert["severity"] == "warning":
            blocking.append(alert["title"])
    if objective >= 60:
        blocking.append("objective strain above 60/100")
    if subjective >= 55:
        blocking.append("self-reported symptom load above threshold")

    cleared = not blocking
    return {
        "cleared_for_eva": cleared,
        "status": "cleared" if cleared else "not cleared",
        "blocking": blocking[:4],
        "note": "EVA clearance is a crew member's own decision aid, not an authority. "
        "It surfaces what would need to be resolved before a spacewalk.",
    }


def _actions(
    level: str,
    drivers: Sequence[dict[str, Any]],
    systems: dict[str, float],
    readiness: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    top = {d["system"] for d in drivers}

    if "cardio" in top:
        actions.append("Sit or recline for 10 minutes, take a full set of vitals, and compare with your personal baseline.")
    if "resp" in top:
        actions.append("Check cabin CO₂ and suit loop, then repeat SpO₂ after five minutes of quiet breathing.")
    if "sleep" in top:
        actions.append("Protect the next sleep window and avoid scheduling demanding tasks into it.")
    if "behavioural" in top:
        actions.append("Complete a behavioural-health check-in and tell your crewmate how you are actually doing.")
    if "immune" in top:
        actions.append("Increase fluids, monitor temperature every four hours, and log any new symptoms.")
    if "musculoskeletal" in top:
        actions.append("Review resistance-loading adherence and add a targeted loading session when fit.")
    if "radiation" in top:
        actions.append("Reduce non-essential EVA hours and log cumulative dose against the career limit.")
    if "hydration" in top:
        actions.append("Follow a structured drinking schedule of at least 2.4 L across the day.")
    if "autonomic" in top:
        actions.append("Treat suppressed HRV as a recovery debt: sleep, fuel and reduce load before it compounds.")

    actions.append(
        "Log a self-check-in now so the next assessment includes how you actually feel."
        if level != "low"
        else "Nominal: log a self-check-in to keep your baseline current."
    )
    if not readiness["cleared_for_eva"]:
        actions.append("EVA is not cleared until the blocking findings above are resolved.")
    return actions
