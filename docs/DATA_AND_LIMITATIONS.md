# Data and limitations

AstroHealth's honest disclosure of what the numbers in it are, and what they are not.

## 1. No astronaut data is redistributed

This repository contains **no physiological records of any real person**. Nothing here was exported from a mission, a flight surgeon's file, the NASA Life Sciences Data Archive, or any clinical dataset. No PhysioNet record is bundled, transformed, or used for training.

Every number the application displays is generated locally by `astrohealth/synthetic.py`.

## 2. What the generator is

`astrohealth/synthetic.py` is a deterministic, seedable signal generator. It is not a random noise generator; its structure encodes published physiology:

- **Reference bands** (`astrohealth/physiology.py`) for twelve signals — resting heart rate, HRV RMSSD, SpO₂, systolic and diastolic pressure, core temperature, total sleep, sleep efficiency, self-reported stress, bone mineral density, seven-day radiation dose and fluid intake — with an operational low/high band, a caution threshold, and a note explaining the spaceflight mechanism behind it.
- **Mission profiles** with different exposure physics: LEO (ISS-like) at 0.45 mSv/day and 1.2% bone loss/month; lunar surface / Gateway at 0.70 mSv/day and 0.9%/month; deep-space transit at 1.8 mSv/day and 1.5%/month.
- **Six crew archetypes** — resilient, behavioural, sleep debt, immune, musculoskeletal, deconditioning — each with its own baseline offsets, recovery behaviour and variance.
- **Circadian and weekly structure**, so sleep, heart rate and HRV co-vary the way they do in a real person across a week rather than independently.
- **Episodes**, which apply a documented set of per-day deltas to a trailing window. These exist so detectors have a genuine deterioration to find. They are used both for the demo crew and for the what-if harness in the API.

The perturbations are chosen to be *directionally* realistic — the sign and rough order of magnitude of in-flight adaptation match the literature. They are **not** statistically calibrated to it. No synthetic record here should be treated as a measurement.

## 3. Sources and what they are used for

| Source | Used for |
|---|---|
| [NASA Human Research Program — Human Research Roadmap](https://humanresearchroadmap.nasa.gov/) | Spaceflight hazards (radiation, unloading, isolation, hostile closed environment) and their documented physiological risks |
| [NASA Life Sciences Data Archive](https://lsda.jsc.nasa.gov/) | Published astronaut physiological findings that inform the *direction* and rough magnitude of in-flight adaptations |
| [NASA Open Data Portal](https://data.nasa.gov/) | Mission and environmental context |
| [PhysioNet](https://physionet.org/) | Open signal corpora consulted for realistic day-to-day dispersion in heart rate and HRV. No records redistributed or used for training |

These are prior-art references that motivate the bands and the generator's structure. They are not training data, and no result in this repository should be attributed to them.

## 4. What the model card means by "89.9% accuracy"

Labels for training are derived from a rule-defined strain index, not from clinicians. The model therefore learns to reproduce that index from the 16 features. High agreement means the features carry the same information the index encodes — which is a useful sanity property and a real engineering check — but it is **not** evidence of clinical validity.

Two properties make the reported numbers more useful than a single accuracy figure:

- **Leave-one-archetype-out validation** holds out an entire crew type and retrains. It answers "does this generalise to a person unlike anyone in the training set?" — and it drops to 84.1% accuracy / 0.687 macro F1, with sleep debt at 62.8%. That drop is the honest measure of transfer.
- **Per-class reporting** shows the high-strain class at 0.657 recall. The model under-calls high strain roughly one time in three. This is disclosed on the model page, not buried.

## 5. Limitations, in the order they are likely to matter

1. **Synthetic, not clinical.** No part of this has been validated against real astronauts or real patients.
2. **Rule-derived labels.** Supervised metrics measure agreement with the rule, not with clinical ground truth.
3. **Small cohort.** Six archetypes is a design choice for clarity, not a sampling of a population. Nothing generalises from six archetypes to a crew of four.
4. **No countermeasures, medication, genetics, or prior injury.** These dominate real crew health and are absent.
5. **Operational bands are not personal limits.** A crew member can be outside a band and entirely fine, and inside every band and deteriorating. That is precisely why the check-in loop exists.
6. **Radiation is simplified.** A trailing seven-day effective dose against a mission-profile band is a coarse stand-in for tissue-specific, cumulative career exposure.
7. **The what-if harness is a demonstration.** Scenario perturbations are not forecasts, and the interface labels them as such.
8. **In-memory state.** Check-ins and scenarios live in process memory and reset on restart. There is no database, no authentication, and no audit trail — deliberately, because this is a single-deployment demonstration.

## 6. What would be required to make this real

- Physiological data from an analogue mission or a validated flight dataset, with clinical adjudication of outcomes.
- Labels defined by a flight surgeon, not by a formula, and inter-rater agreement reported.
- Prospective validation: does the score precede events, and does acting on it change them?
- Per-crew-member personalisation with enough history to establish an individual baseline, rather than cohort archetypes.
- Sensor provenance and artefact handling, which no synthetic pipeline can imitate.
- Regulatory review appropriate to a medical decision-support device.

Until then: this is a working, inspectable system that shows how such a thing would be built, and it says so on its own model page.
