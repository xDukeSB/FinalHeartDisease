"""Unit tests for routine full-body checkup input validation and prediction pipeline."""

import joblib
import pytest

from src.prediction import (
    identify_clinical_risk_factors,
    predict_heart_disease,
    validate_patient_input,
)


@pytest.fixture
def sample_valid_checkup():
    return {
        "age": 52,
        "sex": "Male",
        "systolic_bp": 128,
        "diastolic_bp": 82,
        "total_cholesterol": 195,
        "ldl": 120,
        "hdl": 48,
        "triglycerides": 140,
        "fasting_blood_sugar": 95.0,
        "hba1c": 5.4,
        "height_cm": 172.0,
        "weight_kg": 76.0,
        "bmi": 25.7,
        "smoking_status": "Never",
        "diabetes_history": "false",
        "family_history_heart_disease": "false",
        "physical_activity": "Moderate"
    }


def test_validate_patient_input_valid(sample_valid_checkup):
    is_valid, errors = validate_patient_input(sample_valid_checkup)
    assert is_valid
    assert len(errors) == 0


def test_validate_patient_input_out_of_range(sample_valid_checkup):
    bad_data = sample_valid_checkup.copy()
    bad_data["age"] = 150  # Over limit
    bad_data["systolic_bp"] = 300  # Over limit

    is_valid, errors = validate_patient_input(bad_data)
    assert not is_valid
    assert any("Age" in e for e in errors)
    assert any("Systolic BP" in e for e in errors)


def test_validate_patient_input_invalid_category(sample_valid_checkup):
    bad_data = sample_valid_checkup.copy()
    bad_data["sex"] = "unknown_category"

    is_valid, errors = validate_patient_input(bad_data)
    assert not is_valid
    assert any("sex" in e for e in errors)


def test_predict_heart_disease_end_to_end(sample_valid_checkup):
    pipeline = joblib.load("models/best_model.joblib")
    result = predict_heart_disease(pipeline, sample_valid_checkup)

    assert result["prediction_class"] in [0, 1]
    assert result["risk_level"] in ["Low", "Moderate", "High"]
    assert 0.0 <= result["probability_disease"] <= 100.0
    assert 0.0 <= result["probability_healthy"] <= 100.0
    assert pytest.approx(result["probability_disease"] + result["probability_healthy"], abs=0.5) == 100.0
    assert "narrative" in result
    assert "disclaimer" in result


def test_identify_clinical_risk_factors(sample_valid_checkup):
    high_risk = sample_valid_checkup.copy()
    high_risk["systolic_bp"] = 155
    high_risk["diastolic_bp"] = 95
    high_risk["total_cholesterol"] = 260
    high_risk["smoking_status"] = "Current"
    high_risk["diabetes_history"] = "true"

    factors = identify_clinical_risk_factors(high_risk)
    assert any("Hypertension" in f for f in factors)
    assert any("Cholesterol" in f for f in factors)
    assert any("Smoker" in f for f in factors)
    assert any("Diabetic" in f for f in factors)
