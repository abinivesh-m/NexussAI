# AstroHealth — Astronaut Health Monitoring Software

**NASA Space Apps Challenge 2026 · Create Health Monitoring Software for Astronauts on Space Missions**

An AI-powered health monitoring dashboard that gathers astronaut health indicators, detects anomalies, predicts health risks, and enables astronauts to evaluate and act on their health status during long-duration space missions.

## Problem

Long-duration missions expose astronauts to space radiation, isolation, altered gravity, and a hostile closed environment — causing immune changes, bone loss, cardiovascular events, and behavioral health problems. On long missions, astronauts carry the responsibility of spotting these changes themselves.

## Solution

AstroHealth is a real-time health monitoring dashboard that:
- **Ingests** astronaut health indicators (heart rate, SpO2, blood pressure, sleep, stress, bone density, radiation exposure)
- **Detects anomalies** using threshold + trend analysis (tachycardia, hypoxia, sleep disruption, stress spikes)
- **Predicts health risks** (fatigue, dehydration, cardiac stress, immune suppression) using a trained ML model
- **Alerts** with severity levels (normal / warning / critical) and actionable recommendations
- **Visualizes** health trends over time with interactive charts

## Features

- 🫀 **Vital Signs Monitoring** — heart rate, SpO2, blood pressure, temperature per astronaut
- 📊 **Health Trend Charts** — 7-day rolling vitals with anomaly markers
- ⚠️ **Anomaly Detection** — threshold-based + trend-based alerts
- 🧠 **Risk Prediction** — ML model predicts fatigue, dehydration, cardiac stress risk
- 🚨 **Alert Feed** — real-time severity-classified alerts with recommendations
- 📡 **Mission Timeline** — health status across mission days
- 🌍 **NASA Data** — uses NASA Life Sciences Data Archive physiological parameters

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, scikit-learn, pandas
- **Frontend:** HTML5, CSS3, JavaScript (no framework, no build step)
- **ML:** scikit-learn (Random Forest classifier for risk prediction)
- **Data:** NASA astronaut health parameters (heart rate, SpO2, blood pressure, sleep, stress, bone density, radiation)
- **Charts:** Chart.js (CDN)
- **Deploy:** Vercel / Render / any cloud

## Quick Start

```bash
pip install -r requirements.txt
python -m uvicorn app:app --port 8000
# Open http://127.0.0.1:8000
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Dashboard UI |
| GET | `/api/health` | Health check + system status |
| GET | `/api/astronauts` | List all astronauts with current vitals |
| GET | `/api/astronauts/{id}` | Detailed health data for one astronaut |
| GET | `/api/astronauts/{id}/vitals` | Time-series vitals for charts |
| GET | `/api/alerts` | All active alerts across crew |
| GET | `/api/risk/{id}` | ML risk prediction for one astronaut |
| POST | `/api/simulate` | Simulate a health scenario (demo) |

## Architecture

```
NASA Health Data → FastAPI Backend → Anomaly Detection → Risk Prediction (ML) → Dashboard
                      │
                      ├── Vital signs ingestion (heart rate, SpO2, BP, temp, sleep, stress)
                      ├── Threshold alerts (tachycardia, hypoxia, fever, etc.)
                      ├── Trend analysis (7-day rolling averages, deviation detection)
                      └── ML risk model (Random Forest: fatigue, dehydration, cardiac stress)
```

## Data Sources

- **NASA Life Sciences Data Archive (LSDA)** — astronaut physiological data parameters
- **NASA Open Data Portal** — space mission datasets
- **PhysioNet** — astronaut ECG and cardiovascular data (reference)
- Simulated data based on NASA-documented astronaut health ranges (heart rate 50-100 bpm, SpO2 95-100%, BP 110-130/70-85 mmHg)

## License

MIT — see [LICENSE](LICENSE).

## Built With

Python · FastAPI · scikit-learn · pandas · Chart.js · NASA open data
