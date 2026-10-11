"""Model evaluation metrics, confusion matrix computation, and feature importance aggregation."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessing import CHECKUP_CATEGORICAL_FEATURES, CHECKUP_NUMERICAL_FEATURES


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None
) -> dict[str, float]:
    """
    Compute comprehensive binary classification metrics formatted as percentages (0.00 - 100.00).
    Includes Accuracy, Precision, Recall (Sensitivity), Specificity, F1 Score, ROC-AUC, and PR-AUC.
    """
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0

    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, _fn, _tp = cm.ravel()
    specificity = (float(tn) / float(tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

    metrics = {
        "Accuracy": round(float(acc), 2),
        "Precision": round(float(prec), 2),
        "Recall": round(float(rec), 2),
        "Specificity": round(float(specificity), 2),
        "F1 Score": round(float(f1), 2)
    }

    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob) * 100.0
            metrics["ROC-AUC"] = round(float(roc_auc), 2)
        except (ValueError, TypeError):
            pass
        try:
            pr_auc = average_precision_score(y_true, y_prob) * 100.0
            metrics["PR-AUC"] = round(float(pr_auc), 2)
        except (ValueError, TypeError):
            pass

    return metrics


def compute_confusion_matrix_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None
) -> dict[str, Any]:
    """Compute confusion matrix breakdown, raw cell counts, and percentages."""
    if labels is None:
        labels = ["No Disease", "Disease Indicated"]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = int(np.sum(cm))

    return {
        "matrix": cm.tolist(),
        "labels": labels,
        "total": total,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn_pct": round(float(tn / total * 100), 1) if total > 0 else 0.0,
        "fp_pct": round(float(fp / total * 100), 1) if total > 0 else 0.0,
        "fn_pct": round(float(fn / total * 100), 1) if total > 0 else 0.0,
        "tp_pct": round(float(tp / total * 100), 1) if total > 0 else 0.0
    }


def get_feature_names_from_preprocessor(preprocessor) -> list[str]:
    """Extract transformed feature names from ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend([f"num__{c}" for c in cols])
            elif name == "cat":
                try:
                    ohe = trans.named_steps["onehot"]
                    cat_names = ohe.get_feature_names_out(cols)
                    feature_names.extend([f"cat__{c}" for c in cat_names])
                except Exception:
                    feature_names.extend([f"cat__{c}" for c in cols])
        return feature_names


def extract_aggregated_feature_importance(
    pipeline,
    clinical_features: list[str] | None = None
) -> dict[str, Any]:
    """
    Extract feature importances and aggregate one-hot encoded dummy variables
    back to the original clinical features.
    Supports RandomForest, DecisionTree, and LogisticRegression.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_names = get_feature_names_from_preprocessor(preprocessor)

    raw_importances = []
    is_coef = False

    if hasattr(classifier, "feature_importances_"):
        raw_importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        raw_importances = np.abs(classifier.coef_[0])
        is_coef = True
    else:
        return {"aggregated": {}, "raw": {}, "is_coef": False}

    # Normalize raw importances
    total_raw = float(np.sum(raw_importances))
    normalized_raw = (raw_importances / total_raw) if total_raw > 0 else raw_importances

    raw_dict = {
        name: round(float(val), 4)
        for name, val in zip(transformed_names, normalized_raw)
    }

    if clinical_features is None:
        clinical_features = CHECKUP_NUMERICAL_FEATURES + CHECKUP_CATEGORICAL_FEATURES

    aggregated = {feat: 0.0 for feat in clinical_features}

    for full_name, val in zip(transformed_names, normalized_raw):
        matched = False
        for feat in clinical_features:
            if f"__{feat}" in full_name or f"__{feat}_" in full_name or full_name.endswith(f"__{feat}"):
                aggregated[feat] += float(val)
                matched = True
                break
        if not matched:
            for feat in clinical_features:
                if feat in full_name:
                    aggregated[feat] += float(val)
                    break

    agg_total = sum(aggregated.values())
    if agg_total > 0:
        aggregated = {
            k: round(float(v / agg_total), 4)
            for k, v in aggregated.items()
        }

    sorted_agg = dict(sorted(aggregated.items(), key=lambda item: item[1], reverse=True))

    return {
        "aggregated": sorted_agg,
        "raw": raw_dict,
        "is_coef": is_coef
    }
