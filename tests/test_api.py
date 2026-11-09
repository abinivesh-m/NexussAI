"""HTTP contract tests. These run the real app, not a mock."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        test_client.post("/api/reset")
        yield test_client


def test_health_reports_a_trained_model(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["crew"] == 6
    assert body["data_kind"].startswith("synthetic")
    assert body["model_version"]


def test_index_and_static_assets_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_crew_list_is_sorted_by_risk(client):
    body = client.get("/api/crew").json()
    crew = body["crew"]
    assert len(crew) == 6
    risks = [member["risk"]["overall"] for member in crew]
    assert risks == sorted(risks, reverse=True)
    for member in crew:
        assert member["risk"]["level"] in {"low", "moderate", "high"}
        assert "latest" in member and "heart_rate" in member["latest"]


def test_crew_detail_carries_a_full_assessment(client):
    body = client.get("/api/crew/astro-001").json()
    for key in ("astronaut", "features", "prediction", "alerts", "explanation", "risk", "latest", "baseline"):
        assert key in body
    assert body["prediction"]["label"] in {"low", "moderate", "high"}
    assert body["risk"]["readiness"]["status"] in {"cleared", "not cleared"}
    assert body["risk"]["actions"]


def test_unknown_crew_member_returns_404(client):
    assert client.get("/api/crew/does-not-exist").status_code == 404


def test_series_endpoint_respects_the_day_limit(client):
    body = client.get("/api/crew/astro-002/series?days=5").json()
    assert body["days"] == 5
    assert len(body["series"]) == 5


def test_alerts_aggregate_across_the_crew(client):
    body = client.get("/api/alerts").json()
    assert body["count"] == len(body["alerts"])
    for alert in body["alerts"]:
        assert alert["astronaut_id"] and alert["astronaut_name"]


def test_checkin_changes_the_assessment_and_can_block_eva(client):
    before = client.get("/api/crew/astro-001").json()
    response = client.post(
        "/api/crew/astro-001/checkin",
        json={
            "symptoms": {"chest_tightness": 3, "breathlessness": 2},
            "sleep_quality": 1,
            "mood": 2,
            "workload": 5,
            "hydration_l": 1.0,
            "notes": "Tight chest after EVA.",
        },
    )
    assert response.status_code == 200
    after = response.json()["assessment"]
    assert after["checkin"]["notes"] == "Tight chest after EVA."
    assert after["risk"]["overall"] >= before["risk"]["overall"]
    assert after["risk"]["readiness"]["cleared_for_eva"] is False
    assert after["risk"]["readiness"]["blocking"]


def test_checkin_rejects_unknown_symptoms(client):
    response = client.post("/api/crew/astro-001/checkin", json={"symptoms": {"wing_pain": 2}})
    assert response.status_code == 422


def test_checkin_rejects_out_of_range_severity(client):
    response = client.post("/api/crew/astro-001/checkin", json={"symptoms": {"headache": 9}})
    assert response.status_code == 422


def test_scenario_runs_and_can_be_reset(client):
    baseline = client.get("/api/crew/astro-003").json()
    stressed = client.post("/api/crew/astro-003/simulate", json={"scenario": "cardiac", "days": 5}).json()
    assert stressed["scenario"] == "cardiac"
    assert stressed["assessment"]["prediction"]["strain_index"] >= baseline["prediction"]["strain_index"]

    reset = client.post("/api/crew/astro-003/simulate", json={"scenario": "baseline"}).json()
    assert reset["scenario"] == "baseline"


def test_model_card_exposes_computed_metrics(client):
    body = client.get("/api/model").json()
    metrics = body["metrics"]
    assert metrics["stratified_5fold"]["accuracy"] > 0.5
    assert metrics["stratified_5fold"]["confusion_matrix"]["labels"] == ["low", "moderate", "high"]
    assert len(body["features"]) == metrics["n_features"]
    assert body["data_sources"]


def test_reference_endpoint_lists_bands_and_citations(client):
    body = client.get("/api/reference").json()
    assert len(body["vitals"]) >= 10
    assert body["citations"]
    assert any(b["key"] == "heart_rate" for b in body["vitals"])
    assert body["symptoms"]


def test_trends_return_matching_lengths(client):
    body = client.get("/api/trends?days=10").json()
    assert len(body["dates"]) == 10
    for series in body["series"].values():
        assert len(series) == 10
