"""
Tests for ClaimCheck MLP model inference and missing-model handling.
"""

import os
import json
import pytest
from backend.ml.predict_mlp import ClaimRiskPredictor


def test_mlp_model_inference():
    predictor = ClaimRiskPredictor()
    assert predictor.is_loaded, "Trained MLP model and preprocessor should be loaded."

    # Load a benchmark sample claim
    with open("data/sample_claims.json", "r") as f:
        samples = json.load(f)

    test_claim = samples[0]
    result = predictor.assess_claim(test_claim)

    assert result["status"] == "success"
    assert "raw_score" in result
    assert 0.0 <= result["raw_score"] <= 1.0
    assert result["risk_category"] in ["Lower Review Priority", "Needs Review", "Higher Review Priority"]
    assert result["priority_code"] in ["LOW", "MEDIUM", "HIGH"]
    assert "threshold_config" in result


def test_missing_model_handling():
    predictor = ClaimRiskPredictor(model_path="backend/models/non_existent.keras")
    result = predictor.assess_claim({"Make": "Toyota"})

    assert result["status"] == "model_not_trained"
    assert result["risk_category"] == "Model not trained"
    assert result["raw_score"] is None
