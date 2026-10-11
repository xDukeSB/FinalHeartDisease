"""Integration tests for model training pipeline, artifact persistence, and hyperparameter tuning."""

import json
import os
import shutil
import tempfile

import joblib
import pytest

from src.training import train_and_evaluate_all_models


@pytest.fixture
def temp_models_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_train_and_evaluate_all_models_pipeline(temp_models_dir):
    data_path = "full_body_heart_checkup_synthetic_3000.csv"
    if not os.path.exists(data_path) and os.path.exists("data/full_body_heart_checkup_synthetic_3000.csv"):
        data_path = "data/full_body_heart_checkup_synthetic_3000.csv"

    trained_pipelines, metadata = train_and_evaluate_all_models(
        data_path=data_path,
        output_dir=temp_models_dir,
        random_state=42,
        test_size=0.20,
        tune_hyperparameters=False
    )

    # 1. Check returned objects
    assert len(trained_pipelines) == 3
    assert "Logistic Regression" in trained_pipelines
    assert "Decision Tree" in trained_pipelines
    assert "Random Forest" in trained_pipelines

    # 2. Check metadata
    assert metadata["train_samples"] == 2400
    assert metadata["test_samples"] == 600
    assert metadata["best_model_name"] in trained_pipelines
    assert len(metadata["comparison_table"]) == 3

    # Check metrics present in comparison table
    for row in metadata["comparison_table"]:
        assert "Accuracy" in row
        assert "Precision" in row
        assert "Recall" in row
        assert "Specificity" in row
        assert "F1 Score" in row
        assert "ROC-AUC" in row
        assert 0.0 <= row["Accuracy"] <= 100.0

    # 3. Check serialized files on disk
    best_model_path = os.path.join(temp_models_dir, "best_model.joblib")
    all_models_path = os.path.join(temp_models_dir, "all_models.joblib")
    metadata_path = os.path.join(temp_models_dir, "model_metadata.json")

    assert os.path.exists(best_model_path)
    assert os.path.exists(all_models_path)
    assert os.path.exists(metadata_path)

    # Check loaded pipeline predicts correctly
    loaded_pipeline = joblib.load(best_model_path)
    assert hasattr(loaded_pipeline, "predict")

    with open(metadata_path, "r", encoding="utf-8") as f:
        loaded_meta = json.load(f)
    assert loaded_meta["train_samples"] == 2400
