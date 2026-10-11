"""Data loading, validation, and preprocessing pipeline supporting dual schema architectures.

Supported Schemas:
1. Routine Full-Body Health Checkup Schema (19 columns, 17 predictors)
2. Historical Cleveland / Multicenter Clinical Schema (16 columns, 13 predictors)
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Schema 1: Routine Full-Body Health Checkup
CHECKUP_EXPECTED_COLUMNS: list[str] = [
    "id",
    "age",
    "sex",
    "systolic_bp",
    "diastolic_bp",
    "total_cholesterol",
    "ldl",
    "hdl",
    "triglycerides",
    "fasting_blood_sugar",
    "hba1c",
    "height_cm",
    "weight_kg",
    "bmi",
    "smoking_status",
    "diabetes_history",
    "family_history_heart_disease",
    "physical_activity",
    "heart_disease_label"
]

CHECKUP_NUMERICAL_FEATURES: list[str] = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "total_cholesterol",
    "ldl",
    "hdl",
    "triglycerides",
    "fasting_blood_sugar",
    "hba1c",
    "height_cm",
    "weight_kg",
    "bmi"
]

CHECKUP_CATEGORICAL_FEATURES: list[str] = [
    "sex",
    "smoking_status",
    "diabetes_history",
    "family_history_heart_disease",
    "physical_activity"
]

# Schema 2: Historical Cleveland Multicenter Clinical Dataset
CLEVELAND_EXPECTED_COLUMNS: list[str] = [
    "id",
    "age",
    "sex",
    "dataset",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num"
]

CLEVELAND_NUMERICAL_FEATURES: list[str] = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak"
]

CLEVELAND_CATEGORICAL_FEATURES: list[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

# Defaults for backward compatibility
EXPECTED_COLUMNS = CHECKUP_EXPECTED_COLUMNS
NUMERICAL_FEATURES = CHECKUP_NUMERICAL_FEATURES
CATEGORICAL_FEATURES = CHECKUP_CATEGORICAL_FEATURES
ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def detect_dataset_schema(df: pd.DataFrame) -> str:
    """Detect whether DataFrame matches Checkup schema or Cleveland schema."""
    if "heart_disease_label" in df.columns and "systolic_bp" in df.columns:
        return "checkup"
    if "num" in df.columns and "cp" in df.columns:
        return "cleveland"
    raise ValueError(
        f"Unrecognized dataset schema. Columns present: {list(df.columns)}. "
        f"Must match either Checkup schema ({len(CHECKUP_EXPECTED_COLUMNS)} cols) "
        f"or Cleveland schema ({len(CLEVELAND_EXPECTED_COLUMNS)} cols)."
    )


def validate_columns(df: pd.DataFrame, schema: str = "checkup") -> None:
    """Validate that required columns exist for the detected schema."""
    expected = CHECKUP_EXPECTED_COLUMNS if schema == "checkup" else CLEVELAND_EXPECTED_COLUMNS
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns for {schema} schema: {missing}")


def normalize_boolean_series(series: pd.Series) -> pd.Series:
    """Normalize boolean values in a series to 'true', 'false', or np.nan."""
    def _map_val(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (bool, np.bool_)):
            return "true" if val else "false"
        val_str = str(val).strip().lower()
        if val_str in ["true", "1", "1.0", "yes", "t", "y"]:
            return "true"
        if val_str in ["false", "0", "0.0", "no", "f", "n"]:
            return "false"
        return val_str
    return series.apply(_map_val)


def normalize_dataframe(df: pd.DataFrame, schema: str = "checkup") -> pd.DataFrame:
    """Normalize categorical text, boolean fields, and numeric values for given schema."""
    cleaned = df.copy()

    # Drop identifiers and target labels if present
    drop_cols = [c for c in ["id", "dataset", "num", "heart_disease_label"] if c in cleaned.columns]
    if drop_cols:
        cleaned = cleaned.drop(columns=drop_cols)

    if schema == "checkup":
        # Normalize boolean columns
        for col in ["diabetes_history", "family_history_heart_disease"]:
            if col in cleaned.columns:
                cleaned[col] = normalize_boolean_series(cleaned[col])

        # Normalize string categoricals
        for col in ["sex", "smoking_status", "physical_activity"]:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].apply(
                    lambda x: str(x).strip().title() if pd.notna(x) else np.nan
                )

        # Ensure numeric columns are floats
        for col in CHECKUP_NUMERICAL_FEATURES:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

        # Auto-calculate missing BMI if height and weight present
        if "bmi" in cleaned.columns and "height_cm" in cleaned.columns and "weight_kg" in cleaned.columns:
            mask_missing_bmi = cleaned["bmi"].isna() & cleaned["height_cm"].notna() & cleaned["weight_kg"].notna()
            if mask_missing_bmi.any():
                h_m = cleaned.loc[mask_missing_bmi, "height_cm"] / 100.0
                cleaned.loc[mask_missing_bmi, "bmi"] = np.round(
                    cleaned.loc[mask_missing_bmi, "weight_kg"] / (h_m ** 2), 1
                )

    elif schema == "cleveland":
        # Normalize boolean columns
        for col in ["fbs", "exang"]:
            if col in cleaned.columns:
                cleaned[col] = normalize_boolean_series(cleaned[col])

        # Normalize ca
        if "ca" in cleaned.columns:
            def _norm_ca(val):
                if pd.isna(val):
                    return np.nan
                try:
                    return str(int(float(val)))
                except (ValueError, TypeError):
                    return str(val).strip()
            cleaned["ca"] = cleaned["ca"].apply(_norm_ca)

        # Normalize strings
        for col in ["sex", "cp", "restecg", "slope", "thal"]:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].apply(
                    lambda x: str(x).strip().lower() if pd.notna(x) else np.nan
                )

        # Ensure numeric columns are floats
        for col in CLEVELAND_NUMERICAL_FEATURES:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned


def load_and_preprocess_data(
    filepath: str = "full_body_heart_checkup_synthetic_3000.csv",
    binary_target: bool = True
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """
    Load raw CSV, auto-detect schema, validate columns, normalize values,
    extract feature matrix (excluding identifiers and target), and construct target.
    """
    df = pd.read_csv(filepath)
    schema = detect_dataset_schema(df)
    validate_columns(df, schema=schema)

    if schema == "checkup":
        target_col = "heart_disease_label"
        y = df[target_col].astype(int)
        num_features = CHECKUP_NUMERICAL_FEATURES
        cat_features = CHECKUP_CATEGORICAL_FEATURES
    else:
        target_col = "num"
        y = (df["num"] > 0).astype(int) if binary_target else df["num"].astype(int)
        num_features = CLEVELAND_NUMERICAL_FEATURES
        cat_features = CLEVELAND_CATEGORICAL_FEATURES

    feature_cols = num_features + cat_features

    metadata = {
        "schema": schema,
        "raw_shape": list(df.shape),
        "raw_columns": list(df.columns),
        "feature_columns": feature_cols,
        "numerical_features": num_features,
        "categorical_features": cat_features,
        "missing_counts": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum()),
        "target_distribution_raw": df[target_col].value_counts().to_dict()
    }

    feature_df = df.drop(columns=[c for c in ["id", "dataset", target_col] if c in df.columns])
    X = normalize_dataframe(feature_df, schema=schema)
    X = X[feature_cols]

    return X, y, metadata


def build_preprocessor(
    numerical_features: list[str] = None,
    categorical_features: list[str] = None
) -> ColumnTransformer:
    """
    Create a scikit-learn ColumnTransformer for numerical and categorical features.
    Uses median imputation + StandardScaler for numerical features.
    Uses most_frequent imputation + OneHotEncoder for categorical features.
    """
    if numerical_features is None:
        numerical_features = CHECKUP_NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CHECKUP_CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor
