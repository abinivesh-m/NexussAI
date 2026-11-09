"""Reference physiology used by AstroHealth.

What this module is
-------------------
A single, versioned source of truth for:

* the operational bands AstroHealth uses as decision thresholds,
* the physiological adaptations of long-duration spaceflight that motivate
  those bands, and
* the literature / agency sources those adaptations are drawn from.

What this module is *not*
-------------------------
It is not a diagnostic device. Bands are clinical adult reference ranges
adapted for an operational monitoring context. They are configurable per
mission profile (see ``MISSION_PROFILES``) because deep-space transit and
low-Earth-orbit exposure have very different radiation and unloading profiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Band:
    """An operational reference band for one measured or derived signal."""

    low: float
    high: float
    unit: str
    label: str
    caution_low: float | None = None
    caution_high: float | None = None
    note: str = ""

    def status(self, value: float) -> str:
        """Return ``ok`` / ``caution`` / ``out_of_band`` for a value."""
        if value < self.low or value > self.high:
            return "out_of_band"
        if self.caution_low is not None and value < self.caution_low:
            return "caution"
        if self.caution_high is not None and value > self.caution_high:
            return "caution"
        return "ok"


# --------------------------------------------------------------------------
# Measured signal bands
# --------------------------------------------------------------------------
VITALS: dict[str, Band] = {
    "heart_rate": Band(
        low=50,
        high=90,
        unit="bpm",
        label="Heart rate (resting)",
        caution_high=80,
        note="Resting tachycardia is an early, non-specific marker of "
        "hypovolaemia, fever, fatigue and cardiac deconditioning.",
    ),
    "hrv_rmssd": Band(
        low=20,
        high=95,
        unit="ms",
        label="HRV (RMSSD)",
        caution_low=32,
        note="Reduced parasympathetic tone accompanies sleep debt, "
        "inflammation and elevated psychological load.",
    ),
    "spo2": Band(
        low=95,
        high=100,
        unit="%",
        label="Peripheral oxygen saturation",
        caution_low=97,
        note="Sustained desaturation in a closed cabin points to "
        "respiratory or environmental causes before cardiac ones.",
    ),
    "bp_systolic": Band(
        low=100,
        high=130,
        unit="mmHg",
        label="Systolic blood pressure",
        caution_high=124,
    ),
    "bp_diastolic": Band(
        low=60,
        high=85,
        unit="mmHg",
        label="Diastolic blood pressure",
        caution_high=80,
    ),
    "temperature": Band(
        low=36.0,
        high=37.3,
        unit="°C",
        label="Core temperature",
        caution_high=37.0,
        note="Immune dysregulation is documented in flight; low-grade "
        "fever is easier to miss than to over-call.",
    ),
    "sleep_hours": Band(
        low=6.5,
        high=8.5,
        unit="h",
        label="Total sleep",
        caution_low=7.0,
        note="Crew commonly run below terrestrial sleep need; cumulative "
        "debt degrades vigilance and immune function.",
    ),
    "sleep_efficiency": Band(
        low=80,
        high=95,
        unit="%",
        label="Sleep efficiency",
        caution_low=85,
    ),
    "stress_self": Band(
        low=0,
        high=4,
        unit="0-10",
        label="Self-reported stress",
        caution_high=5,
    ),
    "bone_density_pct": Band(
        low=95,
        high=105,
        unit="% baseline",
        label="Bone mineral density vs preflight",
        caution_low=97,
        note="Weight-bearing bone loses roughly 1-1.5% of mineral density "
        "per month without loading countermeasures (NASA HRP).",
    ),
    "radiation_7d": Band(
        low=0,
        high=4.2,
        unit="mSv / 7 d",
        label="Effective dose (trailing 7 days)",
        caution_high=3.4,
        note="Crew effective dose rate is far above terrestrial background; "
        "thresholds are mission-profile specific and tied to career limits.",
    ),
    "fluid_intake": Band(
        low=2.0,
        high=3.5,
        unit="L / day",
        label="Fluid intake",
        caution_low=2.4,
        note="Hypovolaemia compounds orthostatic intolerance on return "
        "and worsens heart-rate drift during EVA.",
    ),
}


# --------------------------------------------------------------------------
# Mission profiles
# --------------------------------------------------------------------------
MISSION_PROFILES: dict[str, dict[str, float]] = {
    "leo": {
        "label": "Low Earth orbit (ISS-like)",
        "radiation_mSv_per_day": 0.45,
        "bone_loss_pct_per_month": 1.2,
        "gravity": 0.0,
    },
    "lunar": {
        "label": "Lunar surface / Gateway",
        "radiation_mSv_per_day": 0.7,
        "bone_loss_pct_per_month": 0.9,
        "gravity": 0.166,
    },
    "transit": {
        "label": "Deep-space transit (Mars-class)",
        "radiation_mSv_per_day": 1.8,
        "bone_loss_pct_per_month": 1.5,
        "gravity": 0.0,
    },
}


# --------------------------------------------------------------------------
# Symptom vocabulary used by the crew self-check-in
# --------------------------------------------------------------------------
SYMPTOMS: dict[str, dict[str, object]] = {
    "headache": {"label": "Headache", "weight": 0.6, "systems": ["neuro", "hydration"]},
    "dizziness": {"label": "Dizziness / orthostatic", "weight": 1.4, "systems": ["cardio", "hydration"]},
    "nausea": {"label": "Nausea", "weight": 1.0, "systems": ["gi", "neuro"]},
    "chest_tightness": {"label": "Chest tightness", "weight": 2.2, "systems": ["cardio"]},
    "breathlessness": {"label": "Shortness of breath", "weight": 2.0, "systems": ["resp", "cardio"]},
    "insomnia": {"label": "Trouble sleeping", "weight": 1.1, "systems": ["sleep", "behavioural"]},
    "fatigue": {"label": "Unusual fatigue", "weight": 1.2, "systems": ["sleep", "immune"]},
    "anxiety": {"label": "Anxiety / low mood", "weight": 1.0, "systems": ["behavioural"]},
    "joint_pain": {"label": "Joint or back pain", "weight": 0.9, "systems": ["musculoskeletal"]},
    "vision_change": {"label": "Vision change", "weight": 1.8, "systems": ["neuro", "ocular"]},
    "palpitations": {"label": "Palpitations", "weight": 2.0, "systems": ["cardio"]},
    "swelling": {"label": "Facial or limb swelling", "weight": 1.3, "systems": ["cardio", "renal"]},
}


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
CITATIONS: list[dict[str, str]] = [
    {
        "name": "NASA Human Research Program — Human Research Roadmap",
        "url": "https://humanresearchroadmap.nasa.gov/",
        "used_for": "Spaceflight hazards (radiation, unloading, isolation, "
        "hostile closed environment) and their documented physiological risks.",
    },
    {
        "name": "NASA Life Sciences Data Archive (LSDA)",
        "url": "https://lsda.jsc.nasa.gov/",
        "used_for": "Published astronaut physiological measurements that inform "
        "the direction and magnitude of in-flight adaptations.",
    },
    {
        "name": "NASA Open Data Portal",
        "url": "https://data.nasa.gov/",
        "used_for": "Mission and environmental context data.",
    },
    {
        "name": "PhysioNet",
        "url": "https://physionet.org/",
        "used_for": "Open physiological signal datasets used to shape realistic "
        "waveform statistics for heart rate and HRV.",
    },
]


def reference_table() -> list[dict[str, object]]:
    """Serialise the reference bands for the API and the UI."""
    return [{"key": key, **asdict(band)} for key, band in VITALS.items()]
