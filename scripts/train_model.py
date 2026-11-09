"""Train and validate the strain models, then write artifacts.

Usage:
    python scripts/train_model.py

Writes ``models/strain_model.joblib`` and ``models/metrics.json``. The metrics
file is committed on purpose: the model card shown in the UI reads it, so the
numbers a visitor sees are the numbers this script actually produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astrohealth.model import METRICS_PATH, train  # noqa: E402


def main() -> None:
    metrics = train()
    cv = metrics["stratified_5fold"]
    logo = metrics["leave_one_archetype_out"]

    print(f"training windows : {metrics['n_samples']}")
    print(f"class balance    : {metrics['class_distribution']}")
    print(f"5-fold accuracy  : {cv['accuracy']:.4f}")
    print(f"5-fold macro F1  : {cv['macro_f1']:.4f}")
    print(f"unseen archetype : {logo['accuracy']:.4f} accuracy / {logo['macro_f1']:.4f} macro F1")
    print("worst archetype  :", min(logo["per_archetype"].items(), key=lambda kv: kv[1]["accuracy"]))
    print(f"metrics written  : {METRICS_PATH}")
    print(json.dumps(cv["per_class"], indent=2))


if __name__ == "__main__":
    main()
