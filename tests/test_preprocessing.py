"""Unit tests for checkup dataset loading, schema validation, and preprocessing pipeline."""

import os

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    EXPECTED_COLUMNS,
    build_preprocessor,
    load_and_preprocess_data,
    normalize_boolean_series,
    validate_columns,
)

DATA_PATH = "full_body_heart_checkup_synthetic_3000.csv"
if not os.path.exists(DATA_PATH) and os.path.exists("data/full_body_heart_checkup_synthetic_3000.csv"):
    DATA_PATH = "data/full_body_heart_checkup_synthetic_3000.csv"


def test_validate_columns_success():
    df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    validate_columns(df)  # Should not raise


def test_validate_columns_failure():
    df = pd.DataFrame(columns=["age", "sex", "heart_disease_label"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(df)


def test_normalize_boolean_series():
    s = pd.Series(["TRUE", "false", "1", "0", True, False, np.nan])
    res = normalize_boolean_series(s)
    assert res.iloc[0] == "true"
    assert res.iloc[1] == "false"
    assert res.iloc[2] == "true"
    assert res.iloc[3] == "false"
    assert res.iloc[4] == "true"
    assert res.iloc[5] == "false"
    assert pd.isna(res.iloc[6])


def test_load_and_preprocess_data_exclusions():
    X, y, _meta = load_and_preprocess_data(DATA_PATH, binary_target=True)
    assert "id" not in X.columns
    assert "heart_disease_label" not in X.columns
    assert len(X) == 3000
    assert len(y) == 3000
    assert set(y.unique()).issubset({0, 1})
    assert list(X.columns) == ALL_FEATURE_COLUMNS


def test_build_preprocessor_fit_transform():
    X, _y, _ = load_and_preprocess_data(DATA_PATH, binary_target=True)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    # Check that transformed matrix has no remaining NaNs
    assert not np.isnan(transformed).any()


def test_detect_dataset_schema():
    from src.preprocessing import (
        CHECKUP_EXPECTED_COLUMNS,
        CLEVELAND_EXPECTED_COLUMNS,
        detect_dataset_schema,
    )
    df_checkup = pd.DataFrame(columns=CHECKUP_EXPECTED_COLUMNS)
    df_cleveland = pd.DataFrame(columns=CLEVELAND_EXPECTED_COLUMNS)
    assert detect_dataset_schema(df_checkup) == "checkup"
    assert detect_dataset_schema(df_cleveland) == "cleveland"



def test_load_cleveland_dataset_if_present():
    cleveland_path = "data/heart_old_cleveland.csv"
    if os.path.exists(cleveland_path):
        X, y, info = load_and_preprocess_data(cleveland_path, binary_target=True)
        assert info["schema"] == "cleveland"
        assert "id" not in X.columns
        assert "dataset" not in X.columns
        assert "num" not in X.columns
        assert set(y.unique()).issubset({0, 1})
        assert len(X) > 0


def test_load_empty_or_nonexistent_dataset():
    with pytest.raises((FileNotFoundError, ValueError)):
        load_and_preprocess_data("non_existent_file.csv")

