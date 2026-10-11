"""Model training pipeline, cross-validation, hyperparameter tuning, evaluation, and artifact persistence."""

import datetime
import json
import os
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import (
    calculate_metrics,
    compute_confusion_matrix_data,
    extract_aggregated_feature_importance,
)
from src.preprocessing import (
    build_preprocessor,
    load_and_preprocess_data,
)


def get_model_definitions(random_state: int = 42) -> dict[str, Any]:
    """Define candidate classification algorithms."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
    }


def get_tuning_param_grids() -> dict[str, dict[str, list]]:
    """Hyperparameter search grids for training partition cross-validation."""
    return {
        "Logistic Regression": {
            "classifier__C": [0.1, 1.0, 5.0]
        },
        "Decision Tree": {
            "classifier__max_depth": [3, 5, 7],
            "classifier__min_samples_split": [5, 10]
        },
        "Random Forest": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [5, 8]
        }
    }


def train_and_evaluate_all_models(
    data_path: str = "data/full_body_heart_checkup_synthetic_3000.csv",
    output_dir: str = "models",
    random_state: int = 42,
    test_size: float = 0.20,
    tune_hyperparameters: bool = False
) -> tuple[dict[str, Pipeline], dict[str, Any]]:
    """
    Execute full reproducible training workflow:
    1. Loads dataset and auto-detects schema without data leakage.
    2. Performs stratified 80/20 train/test split.
    3. Fits scikit-learn Pipelines with preprocessing and classifiers.
    4. Optional hyperparameter tuning via GridSearchCV on training set only.
    5. Evaluates on held-out test data and via 5-fold cross-validation.
    6. Computes Accuracy, Precision, Recall, Specificity, F1 Score, and ROC-AUC.
    7. Selects best model according to held-out F1 Score.
    8. Serializes artifacts to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    X, y, dataset_info = load_and_preprocess_data(data_path, binary_target=True)
    num_features = dataset_info.get("numerical_features")
    cat_features = dataset_info.get("categorical_features")
    feature_cols = dataset_info.get("feature_columns")

    # Stratified Train/Test Split (leakage-free)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    models_to_train = get_model_definitions(random_state=random_state)
    param_grids = get_tuning_param_grids()

    trained_pipelines: dict[str, Pipeline] = {}
    comparison_table = []
    model_details: dict[str, Any] = {}

    best_model_name = None
    best_f1 = -1.0

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, clf in models_to_train.items():
        preprocessor = build_preprocessor(
            numerical_features=num_features,
            categorical_features=cat_features
        )
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        best_params = {}
        if tune_hyperparameters and name in param_grids:
            # Hyperparameter tuning strictly within training partition
            grid_search = GridSearchCV(
                pipeline,
                param_grid=param_grids[name],
                cv=cv_strategy,
                scoring="f1",
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Baseline cross-validation on training partition only
            pipeline.fit(X_train, y_train)

        # Cross-validation metrics on training set
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_strategy,
            scoring=["accuracy", "precision", "recall", "f1"],
            n_jobs=-1
        )

        trained_pipelines[name] = pipeline

        # Evaluate on strictly held-out test partition
        y_test_pred = pipeline.predict(X_test)
        y_test_prob = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_test_prob = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                pass

        test_metrics = calculate_metrics(
            y_test.values,
            y_test_pred,
            y_prob=y_test_prob
        )
        cm_data = compute_confusion_matrix_data(y_test.values, y_test_pred)
        feat_imp = extract_aggregated_feature_importance(pipeline, clinical_features=feature_cols)

        # Record metrics table
        comparison_table.append({
            "Model": name,
            "Accuracy": test_metrics["Accuracy"],
            "Precision": test_metrics["Precision"],
            "Recall": test_metrics["Recall"],
            "Specificity": test_metrics.get("Specificity", 0.0),
            "F1 Score": test_metrics["F1 Score"],
            "ROC-AUC": test_metrics.get("ROC-AUC", 0.0),
            "CV_Accuracy_Mean": round(float(np.mean(cv_results["test_accuracy"]) * 100), 2),
            "CV_F1_Mean": round(float(np.mean(cv_results["test_f1"]) * 100), 2),
            "Tuned_Parameters": best_params
        })

        model_details[name] = {
            "test_metrics": test_metrics,
            "confusion_matrix": cm_data,
            "feature_importance": feat_imp,
            "cv_metrics": {
                "accuracy_mean": round(float(np.mean(cv_results["test_accuracy"]) * 100), 2),
                "accuracy_std": round(float(np.std(cv_results["test_accuracy"]) * 100), 2),
                "f1_mean": round(float(np.mean(cv_results["test_f1"]) * 100), 2),
                "f1_std": round(float(np.std(cv_results["test_f1"]) * 100), 2)
            },
            "best_params": best_params
        }

        # Track best model based on held-out test F1 score
        if test_metrics["F1 Score"] > best_f1:
            best_f1 = test_metrics["F1 Score"]
            best_model_name = name

    comparison_table = sorted(comparison_table, key=lambda x: x["F1 Score"], reverse=True)

    # Save artifacts
    best_pipeline = trained_pipelines[best_model_name]
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    all_models_path = os.path.join(output_dir, "all_models.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    joblib.dump(best_pipeline, best_model_path)
    joblib.dump(trained_pipelines, all_models_path)

    test_data_path = os.path.join(output_dir, "test_data.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_data_path)

    metadata: dict[str, Any] = {
        "training_timestamp": datetime.datetime.now().isoformat(),
        "random_state": random_state,
        "dataset_info": dataset_info,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_schema": {
            "all_features": feature_cols,
            "numerical_features": num_features,
            "categorical_features": cat_features
        },
        "best_model_name": best_model_name,
        "hyperparameter_tuning_enabled": tune_hyperparameters,
        "comparison_table": comparison_table,
        "model_details": model_details
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return trained_pipelines, metadata
