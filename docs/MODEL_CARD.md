# Model card — AstroHealth strain model

**Version** 2.0.0 · **Generated from** `models/metrics.json` by `scripts/train_model.py`

This card describes what the strain model is, what it was trained on, how it was validated, and where it fails. It is written to be read by someone who will use the output to make a decision.

---

## 1. What the model does

Given a seven-day window of measured physiology (plus mission-day context), the model outputs:

1. **A class** — `low`, `moderate` or `high` strain.
2. **A confidence** — the class-probability mass for the predicted class.
3. **A continuous strain index** — 0–100, from a weighted deviation score, so the interface can show *how far* into a band a crew member sits rather than only which band.
4. **A novelty flag** — from a separate Isolation Forest, raised when the window does not resemble the training cohort.

It is a screening and prioritisation instrument. It is **not** a diagnostic device.

## 2. Model details

| | |
|---|---|
| Estimator | `RandomForestClassifier` (balanced subsample class weights) |
| Novelty detector | `IsolationForest`, contamination 0.08 |
| Inputs | 16 numeric features |
| Outputs | 3 classes + strain index + novelty |
| Training windows | 1,200 |
| Random seed | 7 |
| Artifact | `models/strain_model.joblib` |

### Features

`hr_mean`, `hr_max`, `hrv_rmssd`, `spo2_mean`, `spo2_min`, `bp_systolic`, `bp_diastolic`, `temperature`, `sleep_hours`, `sleep_efficiency`, `stress_self`, `bone_density_pct`, `radiation_7d`, `eva_hours_7d`, `fluid_intake`, `mission_day`.

Feature extraction is a single function (`astrohealth.features.window_features`) called by both training and serving, which is the only reliable way to keep the two from drifting apart. A test asserts that the extracted feature set matches the documented list exactly.

### How the labels were made

There is no clinical ground-truth label for "strain". Labels are derived from a rule-defined **strain index**: a weighted sum of standardised deviations from operational reference bands across cardiac, respiratory, sleep, thermal, musculoskeletal and exposure dimensions. Class cuts are applied at a moderate cut of 18.0 and a high cut of 28.0, chosen so the classes do not collapse.

**Consequence, stated plainly:** the supervised accuracy measures *agreement with that definition*, not agreement with clinical reality. A high score means "the model reproduces the index well", not "the model is medically correct". The index itself is defensible as an operational triage heuristic; it is not a validated clinical score.

### Feature importance

| Feature | Gini importance |
|---|---:|
| Sleep duration | 0.172 |
| Self-reported stress | 0.154 |
| HRV (RMSSD) | 0.150 |
| Sleep efficiency | 0.077 |
| Mean heart rate | 0.068 |
| Minimum SpO₂ | 0.055 |

Importances are in-sample Gini values. They are not SHAP values, are not causal, and are not additive contributions to any individual prediction. Per-crew-member explanations in the application are computed separately, as importance-weighted deviation from that person's own baseline.

## 3. Training data

Synthetic, produced by `astrohealth/synthetic.py` from a deterministic generator. Six crew archetypes (resilient, behavioural, sleep debt, immune, musculoskeletal, deconditioning) across three mission profiles (LEO, lunar, deep-space transit). Class balance in the training set: `low` 838, `moderate` 260, `high` 102.

No astronaut physiological data is redistributed in this repository. See [DATA_AND_LIMITATIONS.md](DATA_AND_LIMITATIONS.md).

## 4. Evaluation

Two evaluations, because they answer different questions.

**Stratified 5-fold, out-of-fold** — can the model separate classes drawn from the same population it trained on?

| Metric | Value |
|---|---|
| Accuracy | **0.8992** |
| Macro F1 | **0.8269** |

| Class | Precision | Recall | Support |
|---|---:|---:|---:|
| low | 0.942 | 0.967 | 838 |
| moderate | 0.762 | 0.777 | 260 |
| high | 0.893 | 0.657 | 102 |

**Leave-one-archetype-out** — can it recognise a *kind of person* it has never seen? Each archetype is held out entirely, the model is retrained, and tested on it.

| Metric | Value |
|---|---|
| Accuracy | **0.8408** |
| Macro F1 | **0.6866** |

| Held-out archetype | Accuracy | n |
|---|---:|---:|
| Resilient | 0.984 | 320 |
| Musculoskeletal | 0.943 | 124 |
| Behavioural | 0.818 | 132 |
| Deconditioning | 0.809 | 236 |
| Immune | 0.807 | 192 |
| Sleep debt | **0.628** | 196 |

### Reading the confusion matrix

|  | low | moderate | high |
|---|---:|---:|---:|
| **low** | 810 | 28 | 0 |
| **moderate** | 50 | 202 | 8 |
| **high** | 0 | 35 | 67 |

Errors are one-sided and predictable: moderate and high strain are confused with each other, never with low. In an operational setting that is the safer failure direction — the model over-triages rather than under-triages. The cost is a higher false-alarm rate for the top two bands, which is why every alert carries its own evidence and a named detector, and why high risk never rests on the model alone.

## 5. Intended use

- Screening and prioritisation during long-duration missions.
- Supporting a crew member's own judgement, and surfacing disagreement with it.
- Demonstrating an end-to-end, inspectable monitoring pipeline.

## 6. Out-of-scope use

- Diagnosis, treatment selection, or fitness-for-duty certification.
- Any real clinical or operational decision without a flight surgeon in the loop.
- Individual medical monitoring of any actual person.
- Inferring anything about a population from these synthetic results.

## 7. Limitations

1. Trained and validated on synthetic signals; not clinically validated.
2. Decision support only — it does not diagnose and must not replace a flight surgeon.
3. Thresholds are operational bands, not personalised clinical limits.
4. Radiation bands are mission-profile specific and simplified.
5. Labels are rule-derived, so supervised metrics measure agreement with the rule.
6. Synthetic data cannot represent medication, genetics, prior injury, or countermeasure adherence.
7. Sleep debt is the weakest archetype at 0.628 accuracy, so sleep-led deterioration should be trusted less than its confidence suggests.
8. The high-strain class has the lowest recall (0.657). Roughly one in three genuine high-strain windows is called moderate.

## 8. Reproducing this card

```bash
python scripts/train_model.py   # retrains, revalidates, rewrites models/metrics.json
```

The application reads `models/metrics.json` at runtime and renders these numbers directly, so the model page cannot drift from this document without the file changing.
