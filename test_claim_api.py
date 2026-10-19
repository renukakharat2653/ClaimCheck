"""
Tests for ClaimCheck REST API endpoints.
"""

import json
import pytest
from backend.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_dashboard_api(client):
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.get_json()
    assert "application_metrics" in data
    assert "historical_dataset_baseline" in data
    assert "model_status_overview" in data
    assert data["historical_dataset_baseline"]["total_records"] == 15420


def test_models_status_api(client):
    response = client.get("/api/models/status")
    assert response.status_code == 200
    data = response.get_json()
    assert "mlp_model" in data
    assert "distilbert_model" in data
    assert "flan_t5_model" in data


def test_models_metrics_api(client):
    response = client.get("/api/models/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "metrics" in data
    assert "roc_auc" in data["metrics"]
    assert "confusion_matrix" in data


def test_submit_and_query_claim(client):
    # Fetch sample claim for payload
    samples_resp = client.get("/api/claims/samples")
    assert samples_resp.status_code == 200
    samples = samples_resp.get_json()
    assert len(samples) > 0

    claim_payload = {
        "claim_data": samples[0],
        "narrative_text": "Driver reported collision on highway off-ramp during heavy rain. Front bumper and hood damaged.",
    }

    # Submit claim
    post_resp = client.post(
        "/api/claims/analyze",
        data=json.dumps(claim_payload),
        content_type="application/json",
    )
    assert post_resp.status_code == 201
    res_data = post_resp.get_json()
    assert res_data["status"] == "success"
    assert "claim_id" in res_data
    assert "assessment" in res_data
    assert "report" in res_data

    claim_id = res_data["claim_id"]

    # Query single claim
    get_resp = client.get(f"/api/claims/{claim_id}")
    assert get_resp.status_code == 200
    claim_detail = get_resp.get_json()
    assert claim_detail["id"] == claim_id

    # Query report
    rep_resp = client.get(f"/api/claims/{claim_id}/report")
    assert rep_resp.status_code == 200
    report_detail = rep_resp.get_json()
    assert "human_review_disclaimer" in report_detail
