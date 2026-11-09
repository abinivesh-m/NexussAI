"""Request/response models for the AstroHealth API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .physiology import SYMPTOMS

Scenario = Literal[
    "baseline",
    "fatigue",
    "sleep",
    "cardiac",
    "immune",
    "musculoskeletal",
    "behavioural",
    "hypovolaemia",
    "radiation",
]


class CheckIn(BaseModel):
    """A crew member's own assessment of how they feel right now."""

    symptoms: dict[str, int] = Field(default_factory=dict)
    sleep_quality: int = Field(default=3, ge=1, le=5, description="1 = terrible, 5 = excellent")
    mood: int = Field(default=3, ge=1, le=5, description="1 = very low, 5 = very good")
    workload: int = Field(default=3, ge=1, le=5, description="1 = light, 5 = overwhelming")
    hydration_l: float | None = Field(default=None, ge=0.0, le=8.0)
    notes: str | None = Field(default=None, max_length=500)

    @field_validator("symptoms")
    @classmethod
    def _validate_symptoms(cls, value: dict[str, int]) -> dict[str, int]:
        cleaned: dict[str, int] = {}
        for key, severity in value.items():
            if key not in SYMPTOMS:
                raise ValueError(f"unknown symptom '{key}'")
            if not 0 <= int(severity) <= 3:
                raise ValueError(f"symptom '{key}' severity must be between 0 and 3")
            if int(severity) > 0:
                cleaned[key] = int(severity)
        return cleaned


class SimulationRequest(BaseModel):
    scenario: Scenario = "fatigue"
    days: int = Field(default=5, ge=1, le=7)


class CheckInResponse(BaseModel):
    astronaut_id: str
    assessment: dict
