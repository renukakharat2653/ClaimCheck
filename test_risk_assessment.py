"""
Tests for ClaimCheck risk assessment thresholds and risk indicator extraction.
"""

from backend.services.risk_assessment_service import extract_risk_indicators, assess_claim_risk


def test_risk_indicator_extraction():
    claim_with_risks = {
        "Days_Policy_Accident": "1 to 7",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "VehiclePrice": "more than 69000",
        "PastNumberOfClaims": "more than 4",
        "Fault": "Policy Holder",
    }
    indicators = extract_risk_indicators(claim_with_risks)
    assert len(indicators) >= 4
    features_flagged = [ind["feature"] for ind in indicators]
    assert "PoliceReportFiled" in features_flagged
    assert "Days_Policy_Accident" in features_flagged


def test_demonstration_threshold_categorization():
    # Test low threshold mapping
    claim_data = {
        "Make": "Honda",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "PolicyType": "Sedan - Liability",
        "BasePolicy": "Liability",
        "Deductible": 400,
        "AccidentArea": "Urban",
        "Month": "Jan",
        "DayOfWeek": "Monday",
        "Year": 1994,
        "Age": 45,
        "Fault": "Third Party",
        "PoliceReportFiled": "Yes",
        "WitnessPresent": "Yes",
        "AgentType": "External",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "no change",
        "NumberOfCars": "1 vehicle",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "none",
        "AgeOfVehicle": "3 years",
        "AgeOfPolicyHolder": "41 to 50",
        "RepNumber": 5,
        "DriverRating": 1,
        "WeekOfMonth": 2,
        "WeekOfMonthClaimed": 2,
        "DayOfWeekClaimed": "Tuesday",
        "MonthClaimed": "Jan",
        "Sex": "Male",
        "MaritalStatus": "Married",
    }

    result = assess_claim_risk(claim_data)
    assert result["status"] == "success"
    assert "priority_code" in result
    assert result["threshold_config"]["is_demonstration_only"] is True
