"""Deterministic, physiologically-grounded signal synthesis.

Why synthetic signals at all
----------------------------
AstroHealth needs continuous, per-crew, longitudinal data to demonstrate
monitoring. Real astronaut physiology is access-controlled and cannot be
redistributed inside a public repository, so the demo layer generates signals
instead.

The generator is *not* uniform noise. Every stream is built from:

1. a per-crew baseline drawn from an archetype (resilient, sleep-deprived,
   deconditioning, immune-stressed, cardiac, musculoskeletal),
2. an AR(1) latent state so day-to-day values are autocorrelated the way real
   physiology is, and
3. occasional multi-day degradation episodes, which are the events a monitor
   is supposed to catch.

Direction and magnitude of each adaptation follow the spaceflight physiology
documented in :mod:`astrohealth.physiology`. Everything is seeded, so a given
seed always produces the same cohort and the same model metrics.

This dataset is clearly labelled as synthetic in the UI, the README and the
model card. It is a demonstration and validation harness, not a claim to have
measured astronauts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

START_DATE = date(2026, 3, 2)


@dataclass(frozen=True)
class Archetype:
    key: str
    label: str
    hr_offset: float = 0.0
    hrv_offset: float = 0.0
    spo2_offset: float = 0.0
    temp_offset: float = 0.0
    sleep_offset: float = 0.0
    efficiency_offset: float = 0.0
    stress_offset: float = 0.0
    fluid_offset: float = 0.0
    bone_rate_mult: float = 1.0
    eva_mult: float = 1.0
    episode_prob: float = 0.10
    episode_kind: str = "fatigue"
    weight: float = 1.0


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        key="resilient",
        label="Well-adapted",
        hrv_offset=6.0,
        sleep_offset=0.3,
        stress_offset=-0.8,
        episode_prob=0.05,
        weight=2.2,
    ),
    Archetype(
        key="sleep_debt",
        label="Cumulative sleep debt",
        hr_offset=3.5,
        hrv_offset=-10.0,
        sleep_offset=-1.4,
        efficiency_offset=-9.0,
        stress_offset=1.6,
        episode_prob=0.14,
        episode_kind="sleep",
        weight=1.4,
    ),
    Archetype(
        key="deconditioning",
        label="Cardiovascular deconditioning",
        hr_offset=7.0,
        hrv_offset=-6.0,
        sleep_offset=-0.4,
        stress_offset=0.6,
        bone_rate_mult=1.35,
        eva_mult=1.6,
        episode_prob=0.12,
        episode_kind="cardiac",
        weight=1.2,
    ),
    Archetype(
        key="immune",
        label="Immune / inflammatory stress",
        hr_offset=5.0,
        hrv_offset=-8.0,
        temp_offset=0.25,
        sleep_offset=-0.7,
        stress_offset=1.1,
        episode_prob=0.16,
        episode_kind="immune",
        weight=1.0,
    ),
    Archetype(
        key="musculoskeletal",
        label="Musculoskeletal unloading",
        hrv_offset=-3.0,
        sleep_offset=-0.5,
        stress_offset=0.5,
        bone_rate_mult=1.7,
        eva_mult=0.6,
        episode_prob=0.10,
        episode_kind="musculoskeletal",
        weight=0.9,
    ),
    Archetype(
        key="behavioural",
        label="Behavioural health strain",
        hr_offset=2.5,
        hrv_offset=-5.0,
        sleep_offset=-0.9,
        stress_offset=2.4,
        episode_prob=0.15,
        episode_kind="behavioural",
        weight=1.0,
    ),
)

ARCH_BY_KEY = {a.key: a for a in ARCHETYPES}

#: Multi-day episode definitions: per-day deltas applied while an episode runs.
EPISODES: dict[str, dict[str, float]] = {
    "fatigue": {"hr": 6.0, "hrv": -12.0, "sleep": -1.8, "stress": 2.0, "eff": -8.0},
    "sleep": {"hr": 4.0, "hrv": -9.0, "sleep": -2.4, "stress": 2.6, "eff": -14.0},
    "cardiac": {"hr": 14.0, "hrv": -16.0, "bps": 9.0, "spo2": -1.6, "stress": 2.2},
    "immune": {"hr": 9.0, "hrv": -14.0, "temp": 0.9, "spo2": -1.2, "stress": 1.8},
    "musculoskeletal": {"hr": 4.0, "hrv": -5.0, "stress": 1.4, "eff": -6.0},
    "behavioural": {"hr": 5.0, "hrv": -11.0, "sleep": -1.6, "stress": 4.2},
    "hypovolaemia": {"hr": 11.0, "hrv": -8.0, "bps": -9.0, "fluid": -1.0, "stress": 1.6},
    "radiation": {"hr": 3.0, "hrv": -6.0, "stress": 1.2},
}

EPISODE_KEYS = tuple(EPISODES)


def apply_episode(records: list[dict[str, Any]], kind: str, days: int = 5) -> list[dict[str, Any]]:
    """Apply an episode's per-day deltas to the trailing ``days`` of a series.

    Shared by the demo cohort (so one crew member is mid-deterioration and the
    monitors have something real to show) and by the API's what-if endpoint.
    """
    deltas = EPISODES[kind]
    for day in records[-days:]:
        day["heart_rate"] = round(float(np.clip(day["heart_rate"] + deltas.get("hr", 0.0), 42, 160)), 1)
        day["hrv_rmssd"] = round(float(np.clip(day["hrv_rmssd"] + deltas.get("hrv", 0.0), 6, 130)), 1)
        day["spo2"] = round(float(np.clip(day["spo2"] + deltas.get("spo2", 0.0), 86, 100)), 1)
        day["bp_systolic"] = round(float(np.clip(day["bp_systolic"] + deltas.get("bps", 0.0), 82, 175)), 1)
        day["temperature"] = round(float(np.clip(day["temperature"] + deltas.get("temp", 0.0), 35.2, 39.6)), 2)
        day["sleep_hours"] = round(float(np.clip(day["sleep_hours"] + deltas.get("sleep", 0.0), 2.0, 10.5)), 1)
        day["sleep_efficiency"] = round(float(np.clip(day["sleep_efficiency"] + deltas.get("eff", 0.0), 40, 99)), 1)
        day["stress_self"] = round(float(np.clip(day["stress_self"] + deltas.get("stress", 0.0), 0, 10)), 1)
        day["fluid_intake"] = round(float(np.clip(day["fluid_intake"] + deltas.get("fluid", 0.0), 0.5, 4.5)), 2)
    return records


@dataclass
class CrewMember:
    id: str
    name: str
    role: str
    age: int
    archetype: str
    mission_day: int
    mission_profile: str = "leo"
    series: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Latent:
    """Latent AR(1) states so streams are autocorrelated, not i.i.d."""

    hr: float = 0.0
    hrv: float = 0.0
    spo2: float = 0.0
    bp: float = 0.0
    temp: float = 0.0
    sleep: float = 0.0
    stress: float = 0.0
    fluid: float = 0.0

    def step(self, rng: np.random.Generator, phi: float = 0.72, sigma: float = 1.0) -> None:
        for name in ("hr", "hrv", "spo2", "bp", "temp", "sleep", "stress", "fluid"):
            prev = getattr(self, name)
            setattr(self, name, phi * prev + rng.normal(0.0, sigma))


def _episode_plan(rng: np.random.Generator, days: int, prob: float, kind: str) -> np.ndarray:
    """Return a per-day scaling array in [0, 1] describing active episodes."""
    plan = np.zeros(days, dtype=float)
    i = 0
    while i < days:
        if rng.random() < prob:
            length = int(rng.integers(3, 8))
            ramp = np.concatenate(
                [np.linspace(0.3, 1.0, min(2, length)), np.ones(max(0, length - 3)), np.linspace(0.9, 0.2, min(1, length))]
            )[:length]
            end = min(days, i + length)
            plan[i:end] = np.maximum(plan[i:end], ramp[: end - i])
            i = end + int(rng.integers(2, 9))
        else:
            i += 1
    # Always give every profile at least one genuine deviation episode so the
    # negative class is not the only thing the model ever sees.
    if plan.max(initial=0.0) == 0.0:
        start = int(rng.integers(0, max(1, days - 5)))
        plan[start : start + 4] = 1.0
    return plan


def generate_series(
    archetype: Archetype,
    days: int,
    rng: np.random.Generator,
    profile: str = "leo",
    mission_day_start: int = 1,
) -> list[dict[str, Any]]:
    """Generate ``days`` consecutive day records for one crew member."""
    from .physiology import MISSION_PROFILES

    prof = MISSION_PROFILES[profile]
    latent = _Latent()
    plan = _episode_plan(rng, days, archetype.episode_prob, archetype.episode_kind)

    # Episode mix: the archetype's signature episode plus occasional others.
    kind = archetype.episode_kind
    if kind not in EPISODES:
        kind = "fatigue"
    delta = EPISODES[kind]

    base_hr = 62.0 + archetype.hr_offset + rng.normal(0, 3)
    base_hrv = 52.0 + archetype.hrv_offset + rng.normal(0, 8)
    base_spo2 = 97.8 + archetype.spo2_offset + rng.normal(0, 0.5)
    base_bps = 117.0 + rng.normal(0, 5)
    base_bpd = 74.0 + rng.normal(0, 4)
    base_temp = 36.55 + archetype.temp_offset + rng.normal(0, 0.12)
    base_sleep = 7.1 + archetype.sleep_offset + rng.normal(0, 0.4)
    base_eff = 88.0 + archetype.efficiency_offset + rng.normal(0, 4)
    base_stress = 3.0 + archetype.stress_offset + rng.normal(0, 0.7)
    base_fluid = 2.8 + archetype.fluid_offset + rng.normal(0, 0.25)

    # Bone density is a mission-duration effect, so it must start from the
    # preflight-equivalent value implied by how long this crew member has
    # already been in flight.
    bone = 100.0 - (prof["bone_loss_pct_per_month"] * archetype.bone_rate_mult) * (mission_day_start - 1) / 30.0
    bone += rng.normal(0, 0.4)

    # Individual vulnerability: some crew members decompensate faster than
    # their archetype suggests. This widens the tail of the strain distribution.
    severity = float(rng.choice([0.7, 1.0, 1.35], p=[0.30, 0.45, 0.25]))

    # A minority of crew carry a second, concurrent stressor. Compounding
    # problems are what actually produce high strain in practice.
    secondary: str | None = None
    if rng.random() < 0.45:
        secondary = EPISODE_KEYS[int(rng.integers(0, len(EPISODE_KEYS)))]
    secondary_plan = (
        _episode_plan(rng, days, max(0.08, archetype.episode_prob * 0.8), secondary) if secondary else None
    )

    records: list[dict[str, Any]] = []
    for i in range(days):
        latent.step(rng, sigma=1.0)
        e = plan[i]
        x = {k: v * severity for k, v in delta.items()}
        if secondary and secondary_plan is not None and secondary_plan[i] > 0:
            e = max(e, secondary_plan[i])
            for k, v in EPISODES[secondary].items():
                x[k] = x.get(k, 0.0) + 0.75 * severity * v
        extra_kind = None
        if rng.random() < 0.02:
            extra_kind = EPISODE_KEYS[int(rng.integers(0, len(EPISODE_KEYS)))]
        if extra_kind:
            for k, v in EPISODES[extra_kind].items():
                x[k] = x.get(k, 0.0) + 0.6 * v

        mission_day = mission_day_start + i
        bone_rate = prof["bone_loss_pct_per_month"] * archetype.bone_rate_mult
        bone -= bone_rate / 30.0
        bone -= 0.02 * e

        eva_hours = max(
            0.0, rng.normal(0.5 * archetype.eva_mult, 0.7) + (2.5 * archetype.eva_mult if rng.random() < 0.06 else 0.0)
        )
        radiation = max(0.05, prof["radiation_mSv_per_day"] * (1.0 + 0.35 * eva_hours / 6.0) + rng.normal(0, 0.05))

        hr = base_hr + 1.8 * latent.hr + e * x.get("hr", 0.0) + 0.6 * eva_hours
        hrv = base_hrv + 2.2 * latent.hrv + e * x.get("hrv", 0.0)
        spo2 = base_spo2 + 0.25 * latent.spo2 + e * x.get("spo2", 0.0)
        bps = base_bps + 2.0 * latent.bp + e * x.get("bps", 0.0)
        bpd = base_bpd + 1.5 * latent.bp
        temp = base_temp + 0.06 * latent.temp + e * x.get("temp", 0.0)
        sleep = base_sleep + 0.35 * latent.sleep + e * x.get("sleep", 0.0)
        eff = base_eff + 2.5 * latent.sleep + e * x.get("eff", 0.0)
        stress = base_stress + 0.6 * latent.stress + e * x.get("stress", 0.0)
        fluid = base_fluid + 0.2 * latent.fluid + e * x.get("fluid", 0.0)

        records.append(
            {
                "day": mission_day,
                "date": (START_DATE + timedelta(days=mission_day - 1)).isoformat(),
                "heart_rate": round(float(np.clip(hr, 42, 155)), 1),
                "hrv_rmssd": round(float(np.clip(hrv, 6, 130)), 1),
                "spo2": round(float(np.clip(spo2, 86, 100)), 1),
                "bp_systolic": round(float(np.clip(bps, 82, 172)), 1),
                "bp_diastolic": round(float(np.clip(bpd, 46, 112)), 1),
                "temperature": round(float(np.clip(temp, 35.2, 39.4)), 2),
                "sleep_hours": round(float(np.clip(sleep, 2.5, 10.5)), 1),
                "sleep_efficiency": round(float(np.clip(eff, 42, 99)), 1),
                "stress_self": round(float(np.clip(stress, 0, 10)), 1),
                "bone_density_pct": round(float(np.clip(bone, 82, 102)), 2),
                "eva_hours": round(float(eva_hours), 1),
                "radiation_msv": round(float(radiation), 3),
                "fluid_intake": round(float(np.clip(fluid, 0.8, 4.5)), 2),
                "episode": round(float(e), 2),
            }
        )
    return records


#: Fixed demo crew — names are invented, roles mirror a real expedition roster.
DEMO_CREW: tuple[dict[str, Any], ...] = (
    {
        "id": "astro-001",
        "name": "Cmdr. Sarah Chen",
        "role": "Mission Commander",
        "age": 44,
        "archetype": "resilient",
        "mission_day": 96,
        "mission_profile": "leo",
    },
    {
        "id": "astro-002",
        "name": "Dr. James Park",
        "role": "Flight Surgeon",
        "age": 39,
        "archetype": "sleep_debt",
        "mission_day": 96,
        "mission_profile": "leo",
        "active_episode": "sleep",
    },
    {
        "id": "astro-003",
        "name": "Lt. Maria Rodriguez",
        "role": "Vehicle Pilot",
        "age": 36,
        "archetype": "deconditioning",
        "mission_day": 96,
        "mission_profile": "leo",
    },
    {
        "id": "astro-004",
        "name": "Dr. Alexei Volkov",
        "role": "Systems Engineer",
        "age": 41,
        "archetype": "immune",
        "mission_day": 96,
        "mission_profile": "leo",
        "active_episode": "immune",
    },
    {
        "id": "astro-005",
        "name": "Dr. Priya Sharma",
        "role": "Research Scientist",
        "age": 37,
        "archetype": "musculoskeletal",
        "mission_day": 96,
        "mission_profile": "leo",
    },
    {
        "id": "astro-006",
        "name": "Capt. Tom Bradley",
        "role": "EVA Specialist",
        "age": 45,
        "archetype": "behavioural",
        "mission_day": 96,
        "mission_profile": "leo",
    },
)


def build_cohort(days: int = 30, seed: int = 2026) -> list[CrewMember]:
    """Build the fixed six-person demo crew with trailing history."""
    rng = np.random.default_rng(seed)
    cohort: list[CrewMember] = []
    for spec in DEMO_CREW:
        arch = ARCH_BY_KEY[spec["archetype"]]
        series = generate_series(
            arch,
            days=days,
            rng=rng,
            profile=spec["mission_profile"],
            mission_day_start=spec["mission_day"] - days + 1,
        )
        if spec.get("active_episode"):
            # One or two crew are deliberately rendered mid-deterioration so the
            # demo has a genuine case to detect rather than an all-green wall.
            apply_episode(series, str(spec["active_episode"]), days=4)
        cohort.append(
            CrewMember(
                id=spec["id"],
                name=spec["name"],
                role=spec["role"],
                age=spec["age"],
                archetype=arch.key,
                mission_day=spec["mission_day"],
                mission_profile=spec["mission_profile"],
                series=series,
            )
        )
    return cohort


def training_windows(
    n_profiles: int = 260,
    days: int = 26,
    seed: int = 7,
) -> tuple[list[dict[str, float]], list[str]]:
    """Generate unlabelled feature windows for model training.

    Returns ``(rows, groups)`` where ``groups`` carries the archetype key of
    the crew member each row came from. The group labels enable
    leave-one-archetype-out validation, which is the only honest way to ask
    "does this generalise to a crew member unlike any it was trained on?".
    """
    from .features import window_features

    rng = np.random.default_rng(seed)
    weights = np.array([a.weight for a in ARCHETYPES], dtype=float)
    weights /= weights.sum()

    rows: list[dict[str, float]] = []
    groups: list[str] = []
    for i in range(n_profiles):
        arch = ARCHETYPES[int(rng.choice(len(ARCHETYPES), p=weights))]
        profile = ["leo", "lunar", "transit"][int(rng.choice(3, p=[0.55, 0.25, 0.20]))]
        start = int(rng.integers(1, 120))
        series = generate_series(arch, days=days, rng=rng, profile=profile, mission_day_start=start)
        # Slide several windows per profile to get more samples per simulated crew.
        for end in range(12, days + 1, 4):
            window = series[:end]
            rows.append(window_features(window, mission_day=start + end - 1))
            groups.append(arch.key)
    return rows, groups
