# Remediation Log — Heart Disease Predictor Technical Audit

This log details all engineering fixes applied to address defects identified during the technical audit of the Heart Disease Predictor system.

---

## Remediation 01: Dynamic Dual-Schema Detection & Historical Cleveland Support

- **Defect Addressed:** `FIND-ML-01` (Severity: P1 — High)
- **Files Modified:**
  - [`src/preprocessing.py`](file:///c:/sagar/FinalHeartDisease/src/preprocessing.py)
  - [`src/training.py`](file:///c:/sagar/FinalHeartDisease/src/training.py)
- **Nature of Change:**
  - Implemented `detect_dataset_schema()` to dynamically inspect input DataFrames and detect whether they follow the 19-column Routine Checkup schema (`full_body_heart_checkup_synthetic_3000.csv`) or the 16-column historical Multicenter Cleveland schema (`heart_old_cleveland.csv`).
  - Added dedicated numerical and categorical column definitions for both schemas (`CHECKUP_NUMERICAL_FEATURES`, `CHECKUP_CATEGORICAL_FEATURES`, `CLEVELAND_NUMERICAL_FEATURES`, `CLEVELAND_CATEGORICAL_FEATURES`).
  - Updated `build_preprocessor()` and `load_and_preprocess_data()` to dynamically configure imputer strategies and one-hot encoders without assuming a single hardcoded column list.
  - Eliminated data leakage by ensuring all imputers and encoders are fitted strictly within scikit-learn `Pipeline` objects on training splits.
- **Regression Tests Executed:**
  - `tests/test_preprocessing.py::test_detect_dataset_schema`
  - `tests/test_preprocessing.py::test_load_cleveland_dataset_if_present`
  - `tests/test_preprocessing.py::test_build_preprocessor_fit_transform`
- **Verification Result:** PASS (Both schemas load, validate, and transform without errors).
- **Remaining Risk:** Custom user datasets that diverge from both known schemas will raise an informative `ValueError` with missing/unexpected column lists.

---

## Remediation 02: Clinical Evaluation Metrics Expansion (Specificity, ROC-AUC, PR-AUC)

- **Defect Addressed:** `FIND-ML-02` (Severity: P2 — Medium)
- **Files Modified:**
  - [`src/evaluation.py`](file:///c:/sagar/FinalHeartDisease/src/evaluation.py)
  - [`src/training.py`](file:///c:/sagar/FinalHeartDisease/src/training.py)
  - [`app.py`](file:///c:/sagar/FinalHeartDisease/app.py)
  - [`train_model.py`](file:///c:/sagar/FinalHeartDisease/train_model.py)
- **Nature of Change:**
  - Expanded `calculate_metrics()` to compute Specificity ($TN / (TN + FP)$) to measure the system's ability to rule out healthy individuals (crucial for screening tools to avoid over-investigation).
  - Integrated `roc_auc_score` and `average_precision_score` (PR-AUC) when model output probabilities are available.
  - Updated the model comparison tables in both CLI (`train_model.py`) and Streamlit web UI (`app.py`) to display Specificity and ROC-AUC.
- **Regression Tests Executed:**
  - `tests/test_evaluation.py::test_calculate_metrics`
  - `tests/test_evaluation.py::test_compute_confusion_matrix_data`
- **Verification Result:** PASS (Logistic Regression: 76.16% Specificity, 75.26% ROC-AUC; Random Forest: 77.91% Specificity, 72.52% ROC-AUC; Decision Tree: 75.00% Specificity, 66.80% ROC-AUC).
- **Remaining Risk:** Models without calibrated probability distributions may present overly confident predictions in extreme edge cases; isotonic or Platt calibration can be explored for clinical deployment.

---

## Remediation 03: Leakage-Free Hyperparameter Tuning via Stratified Cross-Validation

- **Defect Addressed:** `FIND-ML-03` (Severity: P2 — Medium)
- **Files Modified:**
  - [`src/training.py`](file:///c:/sagar/FinalHeartDisease/src/training.py)
  - [`train_model.py`](file:///c:/sagar/FinalHeartDisease/train_model.py)
- **Nature of Change:**
  - Implemented `get_tuning_param_grids()` and integrated `GridSearchCV` inside `train_and_evaluate_all_models()`.
  - Enforced strict partition boundaries: hyperparameter search runs solely on `X_train` using 5-fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), scoring on F1 score.
  - Ensured the held-out test partition (`X_test`, N=600) is evaluated strictly once after model selection to prevent data leakage.
  - Added `--tune` flag to `train_model.py` and persisted the selected best parameters in `models/model_metadata.json`.
- **Regression Tests Executed:**
  - `tests/test_training_pipeline.py::test_train_and_evaluate_all_models_pipeline`
- **Verification Result:** PASS (Optimal parameters identified: Logistic Regression `C=5.0`; Random Forest `max_depth=8, n_estimators=50`; Decision Tree `max_depth=5, min_samples_split=5`).
- **Remaining Risk:** Grid search is constrained to lightweight candidate parameter ranges for rapid reproducibility; extensive bayesian optimization can be scheduled for large datasets.

---

## Remediation 04: Selectbox State Crash on Form Resubmission

- **Defect Addressed:** `FIND-UI-01` (Severity: P1 — High)
- **Files Modified:**
  - [`app.py`](file:///c:/sagar/FinalHeartDisease/app.py)
- **Nature of Change:**
  - Identified root cause: `st.session_state["patient_inputs"]` was being populated with model-encoded boolean strings (`"true"`, `"false"`). When the user submitted the form a second time, the selectbox widget tried to locate `"false"` in its options list `["No", "Yes"]`, throwing `ValueError: 'false' is not in list`.
  - Decoupled the form UI presentation state from the internal model input representation.
  - Created `safe_get_option_index()` with case-insensitive matching and boolean-to-Yes/No normalization fallback.
- **Regression Tests Executed:**
  - `tests/test_ui_app.py::test_multiple_submissions_and_reruns`
  - `tests/test_ui_app.py::test_app_reset_flow`
- **Verification Result:** PASS (Form sustains arbitrary consecutive submissions and state resets without crashing).
- **Remaining Risk:** None identified in tested environments.

---

## Remediation 05: Comparison Table Markdown Code Block Rendering Bug

- **Defect Addressed:** `FIND-UI-02` (Severity: P2 — Medium)
- **Files Modified:**
  - [`app.py`](file:///c:/sagar/FinalHeartDisease/app.py)
- **Nature of Change:**
  - Python multiline f-strings for `table_html` previously contained 4-space indentations. Under Markdown rendering rules in Streamlit, lines indented by 4 spaces are parsed as code blocks (`<pre><code>`), causing the raw HTML tags to be rendered as plain text rather than a rendered table.
  - Formatted the HTML as a flat string without leading indentation, wrapped inside `.comparison-table-wrapper`.
- **Regression Tests Executed:**
  - `tests/test_ui_app.py::test_app_initial_render`
- **Verification Result:** PASS (Model comparison renders as a sleek dark-themed HTML table).
- **Remaining Risk:** None.

---

## Remediation 06: Test Suite Environment Configuration

- **Defect Addressed:** `FIND-TEST-01` (Severity: P2 — Medium)
- **Files Modified:**
  - [`pytest.ini`](file:///c:/sagar/FinalHeartDisease/pytest.ini)
  - [`tests/test_preprocessing.py`](file:///c:/sagar/FinalHeartDisease/tests/test_preprocessing.py)
  - [`tests/test_training_pipeline.py`](file:///c:/sagar/FinalHeartDisease/tests/test_training_pipeline.py)
- **Nature of Change:**
  - Created root `pytest.ini` with `pythonpath = .` and `testpaths = tests`.
  - Resolved `ModuleNotFoundError: No module named 'src'` when invoking `pytest` directly without python `-m`.
  - Added full test coverage for dual-schema loading and end-to-end model training.
- **Regression Tests Executed:**
  - `pytest -v` across all 4 test suites (21 test cases).
- **Verification Result:** PASS (All 21 tests pass in 9.8s).
- **Remaining Risk:** None.
