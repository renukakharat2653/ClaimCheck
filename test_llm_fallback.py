"""
Tests for Generative AI explanation reporting and deterministic fallback.
"""

from backend.services.report_service import generate_claim_report, HUMAN_REVIEW_DISCLAIMER


def test_claim_report_complete_sections():
    sample_claim = {
        "Make": "Ford",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "30000 to 39000",
        "PolicyType": "Sedan - Collision",
        "BasePolicy": "Collision",
        "Deductible": 500,
        "AccidentArea": "Urban",
        "Month": "Feb",
        "DayOfWeek": "Wednesday",
        "Year": 1995,
        "Fault": "Policy Holder",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
    }

    mock_mlp = {
        "raw_score": 0.6842,
        "risk_category": "Higher Review Priority",
        "priority_code": "HIGH",
        "observed_indicators": [
            {"feature": "PoliceReportFiled", "observation": "No formal police report"}
        ],
    }

    mock_text_analysis = {
        "provided": True,
        "mode": "MODE_2_ANALYSIS",
        "word_count": 14,
        "char_count": 82,
        "detected_keywords": ["cash"],
        "embedding_dimension": 768,
        "is_validated_fraud_probability": False,
    }

    report = generate_claim_report(sample_claim, mock_mlp, mock_text_analysis)

    # Verify all 8 core elements
    assert "claim_summary" in report
    assert "mlp_risk_assessment" in report
    assert report["mlp_risk_assessment"]["raw_score"] == 0.6842
    assert "observed_indicators" in report
    assert "text_analysis_findings" in report
    assert report["text_analysis_findings"]["is_validated_fraud_probability"] is False
    assert "suggested_examiner_checks" in report
    assert len(report["suggested_examiner_checks"]) >= 3
    assert "limitations_and_uncertainty" in report
    assert "human_review_disclaimer" in report
    assert report["human_review_disclaimer"] == HUMAN_REVIEW_DISCLAIMER
    assert "generation_metadata" in report
    assert report["generation_metadata"]["generation_method"] in ["flan_t5_base", "deterministic_template_fallback"]
