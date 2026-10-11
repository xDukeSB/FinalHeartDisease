"""Unit tests for metric calculation, confusion matrix, and feature importance."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.evaluation import (
    calculate_metrics,
    compute_confusion_matrix_data,
    extract_aggregated_feature_importance,
)
from src.preprocessing import build_preprocessor


def test_calculate_metrics():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.7])

    metrics = calculate_metrics(y_true, y_pred, y_prob=y_prob)
    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "Specificity" in metrics
    assert "F1 Score" in metrics
    assert "ROC-AUC" in metrics
    assert "PR-AUC" in metrics
    assert 0.0 <= metrics["Accuracy"] <= 100.0
    assert 0.0 <= metrics["Specificity"] <= 100.0
    assert 0.0 <= metrics["ROC-AUC"] <= 100.0



def test_compute_confusion_matrix_data():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    cm_data = compute_confusion_matrix_data(y_true, y_pred)
    assert cm_data["tn"] == 1
    assert cm_data["fp"] == 1
    assert cm_data["fn"] == 1
    assert cm_data["tp"] == 1
    assert cm_data["total"] == 4
    assert cm_data["tn_pct"] == 25.0


def test_extract_aggregated_feature_importance():
    # Build minimal mock pipeline
    X = pd.DataFrame({
        "age": [50, 60, 45, 65],
        "systolic_bp": [120, 140, 110, 130],
        "diastolic_bp": [80, 90, 70, 85],
        "total_cholesterol": [200, 240, 190, 260],
        "ldl": [120, 150, 110, 160],
        "hdl": [50, 45, 55, 40],
        "triglycerides": [150, 180, 130, 200],
        "fasting_blood_sugar": [90.0, 120.0, 85.0, 140.0],
        "hba1c": [5.2, 6.1, 5.0, 7.2],
        "height_cm": [170.0, 175.0, 160.0, 165.0],
        "weight_kg": [70.0, 85.0, 55.0, 80.0],
        "bmi": [24.2, 27.8, 21.5, 29.4],
        "sex": ["Male", "Female", "Male", "Female"],
        "smoking_status": ["Never", "Current", "Never", "Former"],
        "diabetes_history": ["false", "true", "false", "true"],
        "family_history_heart_disease": ["false", "true", "false", "true"],
        "physical_activity": ["Moderate", "Sedentary", "Active", "Sedentary"]
    })
    y = np.array([0, 1, 0, 1])

    preprocessor = build_preprocessor()
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    pipe.fit(X, y)

    res = extract_aggregated_feature_importance(pipe)
    assert "aggregated" in res
    assert "raw" in res
    # Sum of aggregated weights should be approximately 1.0
    agg_sum = sum(res["aggregated"].values())
    assert pytest.approx(agg_sum, abs=1e-2) == 1.0
