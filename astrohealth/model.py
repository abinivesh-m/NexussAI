"""Strain model: definition, training, validation, persistence, explanation.

Two models ship together
------------------------
``RandomForestClassifier`` (supervised, 3 classes)
    Learns a smoothed mapping from a 7-day feature window to a physiological
    strain band. The training target is derived from :func:`strain_index`, a
    weighted deviation score over published reference bands. Because that
    target is deterministic, high in-distribution accuracy is expected; the
    number that actually matters is the grouped holdout, where an entire crew
    archetype is unseen during training.

``IsolationForest`` (unsupervised)
    Flags windows that are statistically unusual even when no threshold is
    crossed. This is what catches a crew member drifting in a way the rule
    set has never been told about.

Honesty constraints
-------------------
* Training data is synthetic and is labelled as such everywhere.
* Reported metrics are computed, never written by hand.
* Explanations are importance-weighted feature deviations, and are named that
  way — they are not SHAP values and are not presented as causal evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .features import FEATURE_LABELS, FEATURES, as_vector, window_features  # noqa: F401
from .physiology import VITALS

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
BUNDLE_PATH = MODEL_DIR / "strain_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

MODEL_VERSION = "2.0.0"
SEED = 7

CLASSES = ("low", "moderate", "high")

#: Weights over normalised band deviations. Chosen to reflect how strongly
#: each signal tracks acute physiological strain, not to maximise accuracy.
STRAIN_WEIGHTS: dict[str, float] = {
    "heart_rate": 0.16,
    "hrv_rmssd": 0.10,
    "spo2": 0.14,
    "bp_systolic": 0.08,
    "temperature": 0.10,
    "sleep_hours": 0.12,
    "sleep_efficiency": 0.06,
    "stress_self": 0.10,
    "bone_density_pct": 0.08,
    "radiation_7d": 0.06,
}

#: Band cut points for the three-way strain label.
MODERATE_CUT = 18.0
HIGH_CUT = 28.0


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def strain_components(features: dict[str, float]) -> dict[str, float]:
    """Per-signal normalised deviation in [0, 1]."""
    return {
        "heart_rate": _clamp01(
            max(0.0, (features["hr_mean"] - 76.0) / 24.0) + 0.5 * max(0.0, (features["hr_max"] - 116.0) / 40.0)
        ),
        "hrv_rmssd": _clamp01((46.0 - features["hrv_rmssd"]) / 30.0),
        "spo2": _clamp01(max(0.0, (97.2 - features["spo2_min"]) / 4.5) + 0.4 * max(0.0, (98.0 - features["spo2_mean"]) / 3.0)),
        "bp_systolic": _clamp01(max(0.0, (features["bp_systolic"] - 128.0) / 24.0) + 0.5 * max(0.0, (100.0 - features["bp_systolic"]) / 18.0)),
        "temperature": _clamp01((features["temperature"] - 37.05) / 1.0),
        "sleep_hours": _clamp01((7.0 - features["sleep_hours"]) / 2.4),
        "sleep_efficiency": _clamp01((86.0 - features["sleep_efficiency"]) / 24.0),
        "stress_self": _clamp01((features["stress_self"] - 4.0) / 5.0),
        "bone_density_pct": _clamp01((97.5 - features["bone_density_pct"]) / 9.0),
        "radiation_7d": _clamp01((features["radiation_7d"] - 3.4) / 4.0),
    }


def strain_index(features: dict[str, float]) -> float:
    """Continuous physiological strain score in [0, 100]."""
    comps = strain_components(features)
    total = sum(STRAIN_WEIGHTS[key] * comps[key] for key in STRAIN_WEIGHTS)
    return round(100.0 * total, 2)


def strain_label(features: dict[str, float]) -> str:
    score = strain_index(features)
    if score >= HIGH_CUT:
        return "high"
    if score >= MODERATE_CUT:
        return "moderate"
    return "low"


def _vectorise(rows: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([as_vector(r) for r in rows], dtype=float)


def train(n_profiles: int = 300, seed: int = SEED) -> dict[str, Any]:
    """Train both models and compute all reported metrics."""
    from .synthetic import training_windows

    rows, groups = training_windows(n_profiles=n_profiles, seed=seed)
    X = _vectorise(rows)
    y = np.asarray([strain_label(r) for r in rows])
    groups_arr = np.asarray(groups)

    clf = RandomForestClassifier(
        n_estimators=320,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )

    # Stratified 5-fold out-of-fold predictions.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(clf, X, y, cv=skf, n_jobs=1)
    cv_accuracy = float(accuracy_score(y, oof))
    cv_macro_f1 = float(f1_score(y, oof, average="macro"))
    report = classification_report(y, oof, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, oof, labels=list(CLASSES)).tolist()

    # Leave-one-archetype-out: every crew type is scored by a model that has
    # never seen that crew type.
    logo_pred = np.empty_like(y)
    per_group: dict[str, dict[str, float]] = {}
    for group in sorted(set(groups)):
        test_mask = groups_arr == group
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
        model.fit(X[~test_mask], y[~test_mask])
        pred = model.predict(X[test_mask])
        logo_pred[test_mask] = pred
        per_group[group] = {
            "accuracy": round(float(accuracy_score(y[test_mask], pred)), 4),
            "n": int(test_mask.sum()),
        }
    logo_f1 = float(f1_score(y, logo_pred, average="macro"))
    logo_accuracy = float(accuracy_score(y, logo_pred))

    clf.fit(X, y)
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=seed,
        n_jobs=1,
    )
    iso.fit(X)

    importances = {name: float(w) for name, w in zip(FEATURES, clf.feature_importances_)}
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)

    metrics: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "n_samples": int(len(rows)),
        "n_features": len(FEATURES),
        "classes": list(CLASSES),
        "class_distribution": {c: int((y == c).sum()) for c in CLASSES},
        "label_definition": {
            "moderate_cut": MODERATE_CUT,
            "high_cut": HIGH_CUT,
            "note": "Labels are derived from a weighted deviation score over "
            "published reference bands. The supervised target is therefore "
            "deterministic, so in-distribution accuracy is expected to be high; "
            "the grouped holdout below is the honest generalisation figure.",
        },
        "stratified_5fold": {
            "accuracy": round(cv_accuracy, 4),
            "macro_f1": round(cv_macro_f1, 4),
            "per_class": {
                c: {
                    "precision": round(float(report[c]["precision"]), 4),
                    "recall": round(float(report[c]["recall"]), 4),
                    "f1": round(float(report[c]["f1-score"]), 4),
                    "support": int(report[c]["support"]),
                }
                for c in CLASSES
                if c in report
            },
            "confusion_matrix": {"labels": list(CLASSES), "matrix": cm},
        },
        "leave_one_archetype_out": {
            "accuracy": round(logo_accuracy, 4),
            "macro_f1": round(logo_f1, 4),
            "per_archetype": per_group,
            "note": "Each crew archetype is scored by a model that never saw "
            "that archetype during training. This is the headline "
            "generalisation figure.",
        },
        "feature_importance": [
            {"feature": name, "label": FEATURE_LABELS.get(name, name), "importance": round(value, 4)}
            for name, value in ranked
        ],
        "feature_order": list(FEATURES),
        "training_data": {
            "kind": "synthetic",
            "generator": "astrohealth.synthetic.generate_series",
            "profiles": int(n_profiles),
            "note": "No astronaut physiological data is redistributed in this repository.",
        },
        "limitations": [
            "Trained and validated on synthetic signals; not clinically validated.",
            "Decision support only — it does not diagnose and must not replace a flight surgeon.",
            "Thresholds are operational bands, not personalised clinical limits.",
            "Radiation bands are mission-profile specific and simplified.",
            "Labels are derived from a rule-defined strain index, so the supervised "
            "score measures agreement with that definition rather than clinical ground truth.",
            "Synthetic data cannot represent medication, genetics, prior injury or "
            "individual countermeasure adherence.",
        ],
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "classifier": clf,
            "novelty": iso,
            "features": list(FEATURES),
            "train_mean": X.mean(axis=0),
            "train_std": X.std(axis=0) + 1e-9,
        },
        BUNDLE_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------
@dataclass
class Prediction:
    label: str
    strain_index: float
    probabilities: dict[str, float]
    anomaly_score: float
    is_novel: bool
    confidence: float


class StrainModel:
    """Thin runtime wrapper around the persisted bundle."""

    def __init__(self, bundle: dict[str, Any], metrics: dict[str, Any]) -> None:
        self.clf: RandomForestClassifier = bundle["classifier"]
        self.iso: IsolationForest = bundle["novelty"]
        self.features: list[str] = bundle["features"]
        self.metrics = metrics
        self._x_mean = np.asarray(bundle["train_mean"], dtype=float)
        self._x_std = np.asarray(bundle["train_std"], dtype=float)

    # -- prediction -------------------------------------------------------
    def predict(self, features: dict[str, float]) -> Prediction:
        x = np.asarray([as_vector(features, self.features)], dtype=float)
        proba = self.clf.predict_proba(x)[0]
        classes = list(self.clf.classes_)
        probabilities = {c: round(float(p), 4) for c, p in zip(classes, proba)}
        label = classes[int(np.argmax(proba))]
        novelty = float(self.iso.score_samples(x)[0])

        return Prediction(
            label=str(label),
            strain_index=strain_index(features),
            probabilities=probabilities,
            anomaly_score=round(novelty, 4),
            is_novel=bool(self.iso.predict(x)[0] == -1),
            confidence=round(float(np.max(proba)), 4),
        )

    # -- explanation ------------------------------------------------------
    def explain(
        self,
        features: dict[str, float],
        baseline: dict[str, float] | None = None,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """Importance-weighted deviation of each feature from a baseline.

        Not SHAP. Named plainly so nobody mistakes it for attributions with
        theoretical guarantees.
        """
        importances = self.clf.feature_importances_
        out: list[dict[str, Any]] = []
        for i, name in enumerate(self.features):
            value = float(features[name])
            ref = float(baseline[name]) if baseline and name in baseline else float(self._x_mean[i])
            band = VITALS.get(name)
            # Scale deviation by the plausible spread of the signal.
            scale = _scale_for(name)
            deviation = (value - ref) / scale
            contribution = float(importances[i] * abs(deviation))
            out.append(
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "value": round(value, 2),
                    "baseline": round(ref, 2),
                    "deviation_sd": round(deviation, 2),
                    "contribution": round(contribution, 4),
                    "direction": "high" if deviation > 0 else "low",
                    "unit": band.unit if band else "",
                }
            )
        out.sort(key=lambda r: r["contribution"], reverse=True)
        return out[:top_k]


_SCALES: dict[str, float] = {
    "hr_mean": 8.0,
    "hr_max": 14.0,
    "hrv_rmssd": 10.0,
    "spo2_mean": 0.8,
    "spo2_min": 1.2,
    "bp_systolic": 7.0,
    "bp_diastolic": 5.0,
    "temperature": 0.25,
    "sleep_hours": 0.9,
    "sleep_efficiency": 6.0,
    "stress_self": 1.4,
    "bone_density_pct": 2.0,
    "radiation_7d": 1.0,
    "eva_hours_7d": 3.0,
    "fluid_intake": 0.4,
    "mission_day": 20.0,
}


def _scale_for(name: str) -> float:
    return _SCALES.get(name, 1.0)


_CACHE: StrainModel | None = None


def load_model(retrain_if_missing: bool = True) -> StrainModel:
    """Load the persisted model, training it on first run if necessary."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    if not BUNDLE_PATH.exists() or not METRICS_PATH.exists():
        if not retrain_if_missing:
            raise FileNotFoundError("No trained model bundle available")
        train()

    bundle = joblib.load(BUNDLE_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    _CACHE = StrainModel(bundle, metrics)
    return _CACHE


def model_metrics() -> dict[str, Any]:
    return load_model().metrics


if __name__ == "__main__":  # pragma: no cover
    info = train()
    print(json.dumps(info["stratified_5fold"], indent=2))
    print("samples:", info["n_samples"], "classes:", info["class_distribution"])
