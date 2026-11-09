"""AstroHealth API.

Run:
    python -m uvicorn app:app --port 8000

The service is a thin layer: signal generation and storage live here, while
physiology, feature extraction, detection, risk fusion and the model live in
the :mod:`astrohealth` package so they can be tested without HTTP.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from astrohealth import __version__, detectors, risk
from astrohealth.features import FEATURE_LABELS, FEATURES, window_features
from astrohealth.model import load_model, model_metrics
from astrohealth.physiology import (
    CITATIONS,
    MISSION_PROFILES,
    SYMPTOMS,
    VITALS,
    reference_table,
)
from astrohealth.schemas import CheckIn, CheckInResponse, SimulationRequest
from astrohealth.synthetic import CrewMember, apply_episode, build_cohort

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Train on first boot so the first request is not the one that pays for it."""
    load_model()
    yield


app = FastAPI(
    title="AstroHealth",
    version=__version__,
    description="Crew-facing health monitoring and self-assessment for long-duration spaceflight.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Store:
    """In-memory mission state.

    A real deployment swaps this for a time-series store. Keeping it behind a
    small interface means the detectors and model never learn where data
    comes from.
    """

    def __init__(self) -> None:
        self.history_days = 30
        self.checkins: dict[str, dict[str, Any]] = {}
        self.cohort: dict[str, CrewMember] = {}
        self.reset()

    def reset(self) -> None:
        self.checkins = {}
        self.cohort = {member.id: member for member in build_cohort(days=self.history_days)}

    def get(self, aid: str) -> CrewMember:
        member = self.cohort.get(aid)
        if member is None:
            raise HTTPException(status_code=404, detail=f"Unknown crew member '{aid}'")
        return member

    def baseline_features(self, member: CrewMember) -> dict[str, float]:
        first_week = member.series[:7]
        return window_features(first_week, mission_day=first_week[-1]["day"])

    def regenerate(self, aid: str) -> CrewMember:
        """Restore one crew member to the deterministic baseline signals."""
        fresh = {m.id: m for m in build_cohort(days=self.history_days)}[aid]
        self.cohort[aid] = fresh
        self.checkins.pop(aid, None)
        return fresh

    def assess(self, member: CrewMember) -> dict[str, Any]:
        model = load_model()
        features = window_features(member.series, mission_day=member.mission_day)
        baseline = self.baseline_features(member)
        prediction = model.predict(features)
        alerts = detectors.detect(member.series, features, prediction)
        explanation = model.explain(features, baseline)

        checkin = self.checkins.get(member.id)
        symptom = detectors.symptom_score(checkin["symptoms"]) if checkin else {"symptom_load": 0.0, "systems": {}}

        assessment = risk.assess(
            features=features,
            prediction=prediction,
            alerts=alerts,
            symptom_load=symptom["symptom_load"],
            symptom_systems=symptom["systems"],
            sleep_quality=checkin["sleep_quality"] if checkin else None,
            workload=checkin["workload"] if checkin else None,
            has_self_report=checkin is not None,
        )

        return {
            "astronaut": _public_member(member),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "features": {k: round(v, 2) for k, v in features.items()},
            "baseline": {k: round(v, 2) for k, v in baseline.items()},
            "prediction": {
                "label": prediction.label,
                "strain_index": prediction.strain_index,
                "probabilities": prediction.probabilities,
                "confidence": prediction.confidence,
                "is_novel": prediction.is_novel,
                "anomaly_score": prediction.anomaly_score,
            },
            "alerts": alerts,
            "explanation": explanation,
            "risk": assessment,
            "checkin": checkin,
            "latest": member.series[-1],
        }


def _public_member(member: CrewMember) -> dict[str, Any]:
    return {
        "id": member.id,
        "name": member.name,
        "role": member.role,
        "age": member.age,
        "archetype": member.archetype,
        "mission_day": member.mission_day,
        "mission_profile": member.mission_profile,
        "mission_profile_label": MISSION_PROFILES[member.mission_profile]["label"],
    }


store = Store()


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(str(STATIC / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(str(STATIC / "favicon.svg"))


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    metrics = model_metrics()
    return {
        "ok": True,
        "name": "AstroHealth",
        "version": __version__,
        "crew": len(store.cohort),
        "model_version": metrics["model_version"],
        "trained_at": metrics["trained_at"],
        "data_kind": "synthetic (documented)",
    }


@app.get("/api/model")
def model_card() -> dict[str, Any]:
    metrics = model_metrics()
    return {
        "metrics": metrics,
        "features": [{"key": k, "label": FEATURE_LABELS.get(k, k)} for k in FEATURES],
        "data_sources": CITATIONS,
        "data_kind": "synthetic",
        "data_statement": (
            "AstroHealth ships a deterministic, physiologically-grounded signal "
            "generator. No astronaut physiological data is redistributed here. "
            "The model is trained and validated on that generator so the whole "
            "pipeline is reproducible end to end."
        ),
    }


@app.get("/api/reference")
def reference() -> dict[str, Any]:
    return {
        "vitals": reference_table(),
        "symptoms": [
            {"key": k, "label": v["label"], "systems": v["systems"]}
            for k, v in SYMPTOMS.items()
        ],
        "mission_profiles": MISSION_PROFILES,
        "citations": CITATIONS,
    }


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------
@app.get("/api/crew")
def crew() -> dict[str, Any]:
    members = sorted(store.cohort.values(), key=lambda m: m.id)
    summaries = []
    for member in members:
        assessment = store.assess(member)
        top = assessment["alerts"][0] if assessment["alerts"] else None
        summaries.append(
            {
                **_public_member(member),
                "latest": assessment["latest"],
                "risk": {
                    "overall": assessment["risk"]["overall"],
                    "level": assessment["risk"]["level"],
                    "readiness": assessment["risk"]["readiness"]["status"],
                },
                "alerts": {
                    "total": len(assessment["alerts"]),
                    "critical": sum(1 for a in assessment["alerts"] if a["severity"] == "critical"),
                    "warning": sum(1 for a in assessment["alerts"] if a["severity"] == "warning"),
                    "top": top,
                },
                "prediction": assessment["prediction"],
                "checked_in": member.id in store.checkins,
            }
        )
    summaries.sort(key=lambda s: s["risk"]["overall"], reverse=True)
    return {
        "crew": summaries,
        "mission": {
            "day": members[0].mission_day if members else 0,
            "profile": MISSION_PROFILES[members[0].mission_profile]["label"] if members else "",
            "crew_size": len(members),
            "highest_risk": summaries[0]["name"] if summaries else None,
        },
    }


@app.get("/api/crew/{aid}")
def crew_detail(aid: str) -> dict[str, Any]:
    return store.assess(store.get(aid))


@app.get("/api/crew/{aid}/series")
def crew_series(aid: str, days: int = 14) -> dict[str, Any]:
    member = store.get(aid)
    days = max(1, min(days, len(member.series)))
    series = member.series[-days:]
    return {
        "astronaut_id": member.id,
        "days": days,
        "series": series,
        "bands": {k: {"low": b.low, "high": b.high, "unit": b.unit, "label": b.label} for k, b in VITALS.items()},
    }


@app.get("/api/alerts")
def all_alerts() -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for member in store.cohort.values():
        assessment = store.assess(member)
        for alert in assessment["alerts"]:
            items.append(
                {
                    **alert,
                    "astronaut_id": member.id,
                    "astronaut_name": member.name,
                    "role": member.role,
                }
            )
    order = {"critical": 0, "warning": 1, "info": 2}
    items.sort(key=lambda a: (order.get(a["severity"], 9), -float(a.get("confidence", 0))))
    return {
        "alerts": items,
        "count": len(items),
        "critical": sum(1 for a in items if a["severity"] == "critical"),
        "warning": sum(1 for a in items if a["severity"] == "warning"),
        "info": sum(1 for a in items if a["severity"] == "info"),
    }


@app.get("/api/trends")
def trends(days: int = 21) -> dict[str, Any]:
    """Mission-wide daily means for the overview chart."""
    days = max(5, min(days, store.history_days))
    members = list(store.cohort.values())
    window = [m.series[-days:] for m in members]
    dates = [d["date"] for d in window[0]]

    def daily(key: str, reducer: str = "mean") -> list[float]:
        values = []
        for i in range(days):
            column = [w[i][key] for w in window]
            values.append(round(float(np.mean(column) if reducer == "mean" else np.min(column)), 2))
        return values

    return {
        "days": days,
        "dates": dates,
        "series": {
            "heart_rate": daily("heart_rate"),
            "spo2": daily("spo2"),
            "sleep_hours": daily("sleep_hours"),
            "stress_self": daily("stress_self"),
            "hrv_rmssd": daily("hrv_rmssd"),
            "bone_density_pct": daily("bone_density_pct"),
        },
    }


# ---------------------------------------------------------------------------
# Crew actions
# ---------------------------------------------------------------------------
@app.post("/api/crew/{aid}/checkin", response_model=CheckInResponse)
def checkin(aid: str, payload: CheckIn) -> CheckInResponse:
    member = store.get(aid)
    record = payload.model_dump()
    record["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store.checkins[aid] = record

    if payload.hydration_l is not None:
        member.series[-1]["fluid_intake"] = round(float(payload.hydration_l), 2)

    return CheckInResponse(astronaut_id=member.id, assessment=store.assess(member))


@app.post("/api/crew/{aid}/simulate")
def simulate(aid: str, payload: SimulationRequest) -> dict[str, Any]:
    member = store.get(aid)

    if payload.scenario == "baseline":
        fresh = store.regenerate(aid)
        return {"scenario": "baseline", "assessment": store.assess(fresh)}

    apply_episode(member.series, payload.scenario, days=payload.days)

    return {"scenario": payload.scenario, "assessment": store.assess(member)}


@app.post("/api/reset")
def reset() -> dict[str, str]:
    store.reset()
    return {"status": "reset"}
