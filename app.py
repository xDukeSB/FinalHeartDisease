"""Heart Disease Predictor — Cardiovascular Risk Assessment from Routine Full-Body Checkup Reports."""

import json
import os
from typing import Any

import joblib
import streamlit as st

from src.prediction import (
    predict_heart_disease,
    validate_patient_input,
)
from src.training import train_and_evaluate_all_models
from src.visualization import (
    create_confusion_matrix_figure,
    create_feature_importance_figure,
    create_probability_gauge_figure,
)

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor — Full-Body Checkup",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark mode recreation
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global resets & typography */
html, body, [class*="css"], .stApp {
    background-color: #0B0E14 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #F5F7FA !important;
}

/* Force centered narrow container (680px) */
.main .block-container {
    max-width: 680px !important;
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    margin: 0 auto !important;
}

/* Header typography */
.app-header-title {
    font-size: 26px;
    font-weight: 700;
    color: #F5F7FA;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.app-header-sub {
    font-size: 13px;
    color: #AAB3C2;
    margin-bottom: 12px;
}

/* Custom divider */
.custom-divider {
    border: 0;
    border-top: 1px solid #303540;
    margin: 16px 0;
}

/* Section header */
.section-heading {
    font-size: 16px;
    font-weight: 700;
    color: #F5F7FA;
    margin-bottom: 14px;
    letter-spacing: -0.01em;
}

/* Form inputs styling */
label[data-testid="stWidgetLabel"] p {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #AAB3C2 !important;
    margin-bottom: 2px !important;
}

/* Input boxes & dropdowns */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #252833 !important;
    border: 1px solid #303540 !important;
    border-radius: 6px !important;
    color: #F5F7FA !important;
    font-size: 13px !important;
    min-height: 38px !important;
}

div[data-baseweb="input"] input {
    color: #F5F7FA !important;
    background-color: transparent !important;
    font-size: 13px !important;
}

div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div {
    border-color: #2E86DE !important;
    box-shadow: 0 0 0 1px #2E86DE !important;
}

/* Dropdown popover menu */
ul[data-baseweb="menu"] {
    background-color: #1A1D26 !important;
    border: 1px solid #303540 !important;
}
li[data-baseweb="menu-item"] {
    color: #F5F7FA !important;
    font-size: 13px !important;
}
li[data-baseweb="menu-item"]:hover {
    background-color: #252833 !important;
    color: #2E86DE !important;
}

/* Custom styled buttons */
div.stButton > button {
    background-color: #252833 !important;
    color: #F5F7FA !important;
    border: 1px solid #303540 !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.15s ease-in-out !important;
}
div.stButton > button:hover {
    background-color: #2E3342 !important;
    border-color: #2E86DE !important;
    color: #FFFFFF !important;
}
div.stButton > button:active {
    background-color: #2E86DE !important;
    border-color: #2E86DE !important;
}

/* Tabs styling */
div[data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid #303540 !important;
    gap: 8px !important;
}
button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #AAB3C2 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 4px 4px 0 0 !important;
    padding: 8px 14px !important;
    border: none !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #F5F7FA !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #2E86DE !important;
}

/* Comparison Table */
.comparison-table-wrapper {
    width: 100%;
    margin-top: 10px;
    margin-bottom: 14px;
    border: 1px solid #303540;
    border-radius: 6px;
    overflow: hidden;
    background-color: #12151E;
}
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    text-align: left;
}
.comparison-table th {
    background-color: #1A1E29;
    color: #AAB3C2;
    font-weight: 600;
    padding: 9px 12px;
    border-bottom: 1px solid #303540;
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.comparison-table td {
    padding: 9px 12px;
    color: #F5F7FA;
    border-bottom: 1px solid #222632;
}
.comparison-table tr:last-child td {
    border-bottom: none;
}
.comparison-table tr:hover {
    background-color: #1E2330;
}
.best-tag {
    background-color: #1A365D;
    color: #63B3ED;
    border: 1px solid #2B6CB0;
    font-size: 10.5px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
    margin-left: 6px;
}

/* Prediction Result Card */
.result-card {
    background-color: #12151E;
    border: 1px solid #303540;
    border-radius: 8px;
    padding: 18px 20px;
    margin: 16px 0;
}
.result-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 5px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.02em;
    margin-bottom: 10px;
}
.badge-danger {
    background-color: #381515;
    color: #FC8181;
    border: 1px solid #E53E3E;
}
.badge-success {
    background-color: #11301F;
    color: #68D391;
    border: 1px solid #38A169;
}
.badge-warning {
    background-color: #362810;
    color: #F6AD55;
    border: 1px solid #DD6B20;
}

/* Disclaimer box */
.disclaimer-box {
    background-color: #141720;
    border: 1px solid #272C38;
    border-left: 3px solid #2E86DE;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 11.5px;
    color: #8E99AB;
    line-height: 1.45;
    margin-top: 14px;
}

#MainMenu, header, footer {
    visibility: hidden;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load trained models and evaluation metadata, retraining if necessary."""
    best_model_path = "models/best_model.joblib"
    all_models_path = "models/all_models.joblib"
    metadata_path = "models/model_metadata.json"

    dataset_path = "data/full_body_heart_checkup_synthetic_3000.csv"
    if not os.path.exists(dataset_path) and os.path.exists("full_body_heart_checkup_synthetic_3000.csv"):
        dataset_path = "full_body_heart_checkup_synthetic_3000.csv"

    if not (os.path.exists(best_model_path) and os.path.exists(metadata_path)):
        train_and_evaluate_all_models(data_path=dataset_path, output_dir="models")

    best_pipeline = joblib.load(best_model_path)
    all_pipelines = joblib.load(all_models_path) if os.path.exists(all_models_path) else {"Best Model": best_pipeline}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return best_pipeline, all_pipelines, metadata


# Load cached artifacts safely
try:
    best_pipeline, all_pipelines, metadata = load_model_artifacts()
    best_model_name = metadata.get("best_model_name", "Logistic Regression")
except Exception as e:
    st.error(f"Error initializing machine learning pipeline: {e}")
    st.stop()


# Default form baseline values (representative routine checkup values)
DEFAULT_VALUES = {
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
    "diabetes_history": "No",
    "family_history_heart_disease": "No",
    "physical_activity": "Moderate"
}

# Session state initialization
if "patient_inputs" not in st.session_state:
    st.session_state["patient_inputs"] = DEFAULT_VALUES.copy()
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "selected_model_name" not in st.session_state:
    st.session_state["selected_model_name"] = best_model_name


def reset_form_callback():
    """Reset form fields and clear prior predictions."""
    st.session_state["patient_inputs"] = DEFAULT_VALUES.copy()
    st.session_state["prediction_result"] = None
    for k, v in DEFAULT_VALUES.items():
        st.session_state[f"field_{k}"] = v


# ==========================================
# 1. HEADER SECTION
# ==========================================
st.markdown('<div class="app-header-title">❤️ Heart Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="app-header-sub">Machine learning cardiovascular risk prediction from routine full-body checkup reports</div>', unsafe_allow_html=True)
st.markdown('<hr class="custom-divider"/>', unsafe_allow_html=True)


# ==========================================
# 2. PATIENT INFORMATION FORM SECTION
# ==========================================
st.markdown('<div class="section-heading">Routine Health Checkup Information</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)

# Options mapping
SEX_OPTIONS = ["Male", "Female"]
SMOKING_OPTIONS = ["Never", "Former", "Current"]
ACTIVITY_OPTIONS = ["Sedentary", "Moderate", "Active"]
YES_NO_OPTIONS = ["No", "Yes"]


def safe_get_option_index(options: list, value: Any, default_idx: int = 0) -> int:
    """Safely return option index without raising ValueError on case or format mismatch."""
    if value in options:
        return options.index(value)

    val_str = str(value).strip().lower()

    if options == YES_NO_OPTIONS:
        if val_str in ["yes", "true", "1"]:
            return options.index("Yes")
        return options.index("No")

    if options == SEX_OPTIONS:
        if val_str in ["female", "f"]:
            return options.index("Female")
        return options.index("Male")

    for i, opt in enumerate(options):
        if str(opt).strip().lower() == val_str:
            return i

    return default_idx


with col_left:
    age_input = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=110,
        value=int(st.session_state["patient_inputs"].get("age", 52)),
        step=1,
        key="field_age"
    )

    systolic_bp_input = st.number_input(
        "Systolic Blood Pressure (mmHg)",
        min_value=70,
        max_value=240,
        value=int(st.session_state["patient_inputs"].get("systolic_bp", 128)),
        step=1,
        key="field_systolic_bp"
    )

    total_chol_input = st.number_input(
        "Total Cholesterol (mg/dL)",
        min_value=80,
        max_value=450,
        value=int(st.session_state["patient_inputs"].get("total_cholesterol", 195)),
        step=1,
        key="field_total_cholesterol"
    )

    hdl_input = st.number_input(
        "HDL Cholesterol (mg/dL)",
        min_value=15,
        max_value=120,
        value=int(st.session_state["patient_inputs"].get("hdl", 48)),
        step=1,
        key="field_hdl"
    )

    fbs_input = st.number_input(
        "Fasting Blood Sugar (mg/dL)",
        min_value=50.0,
        max_value=350.0,
        value=float(st.session_state["patient_inputs"].get("fasting_blood_sugar", 95.0)),
        step=1.0,
        format="%.1f",
        key="field_fasting_blood_sugar"
    )

    height_input = st.number_input(
        "Height (cm)",
        min_value=120.0,
        max_value=220.0,
        value=float(st.session_state["patient_inputs"].get("height_cm", 172.0)),
        step=0.5,
        format="%.1f",
        key="field_height_cm"
    )

    diabetes_input = st.selectbox(
        "Diagnosed History of Diabetes",
        options=YES_NO_OPTIONS,
        index=safe_get_option_index(YES_NO_OPTIONS, st.session_state["patient_inputs"].get("diabetes_history", "No")),
        key="field_diabetes_history"
    )

    activity_input = st.selectbox(
        "Physical Activity Tier",
        options=ACTIVITY_OPTIONS,
        index=safe_get_option_index(ACTIVITY_OPTIONS, st.session_state["patient_inputs"].get("physical_activity", "Moderate")),
        key="field_physical_activity"
    )


with col_right:
    sex_input = st.selectbox(
        "Sex",
        options=SEX_OPTIONS,
        index=safe_get_option_index(SEX_OPTIONS, st.session_state["patient_inputs"].get("sex", "Male")),
        key="field_sex"
    )

    diastolic_bp_input = st.number_input(
        "Diastolic Blood Pressure (mmHg)",
        min_value=40,
        max_value=140,
        value=int(st.session_state["patient_inputs"].get("diastolic_bp", 82)),
        step=1,
        key="field_diastolic_bp"
    )

    ldl_input = st.number_input(
        "LDL Cholesterol (mg/dL)",
        min_value=30,
        max_value=300,
        value=int(st.session_state["patient_inputs"].get("ldl", 120)),
        step=1,
        key="field_ldl"
    )

    trig_input = st.number_input(
        "Triglycerides (mg/dL)",
        min_value=30,
        max_value=600,
        value=int(st.session_state["patient_inputs"].get("triglycerides", 140)),
        step=1,
        key="field_triglycerides"
    )

    hba1c_input = st.number_input(
        "HbA1c (%)",
        min_value=3.5,
        max_value=15.0,
        value=float(st.session_state["patient_inputs"].get("hba1c", 5.4)),
        step=0.1,
        format="%.1f",
        key="field_hba1c"
    )

    weight_input = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=250.0,
        value=float(st.session_state["patient_inputs"].get("weight_kg", 76.0)),
        step=0.5,
        format="%.1f",
        key="field_weight_kg"
    )

    smoking_input = st.selectbox(
        "Smoking Status",
        options=SMOKING_OPTIONS,
        index=safe_get_option_index(SMOKING_OPTIONS, st.session_state["patient_inputs"].get("smoking_status", "Never")),
        key="field_smoking_status"
    )

    fam_hist_input = st.selectbox(
        "Family History of Heart Disease",
        options=YES_NO_OPTIONS,
        index=safe_get_option_index(YES_NO_OPTIONS, st.session_state["patient_inputs"].get("family_history_heart_disease", "No")),
        key="field_family_history_heart_disease"
    )

# Real-time BMI calculation from height and weight
computed_bmi = round(weight_input / ((height_input / 100.0) ** 2), 1)
st.markdown(
    f'<div style="background-color: #161A24; border: 1px solid #303540; border-radius: 6px; padding: 7px 12px; margin-top: 8px; font-size: 12px; color: #AAB3C2;">'
    f'Calculated BMI: <b style="color: #F5F7FA;">{computed_bmi:.1f} kg/m²</b> '
    f'({"Normal weight" if computed_bmi < 25 else "Overweight" if computed_bmi < 30 else "Obese"})'
    f'</div>',
    unsafe_allow_html=True
)


# ==========================================
# 3. ACTION BUTTONS SECTION
# ==========================================
st.markdown('<hr class="custom-divider"/>', unsafe_allow_html=True)

btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    analyze_clicked = st.button("🔎 Analyze Health Checkup Report", width="stretch")

with btn_col2:
    reset_clicked = st.button("🔄 Reset Form", width="stretch", on_click=reset_form_callback)

# Process form submission
if analyze_clicked:
    ui_inputs = {
        "age": age_input,
        "sex": sex_input,
        "systolic_bp": systolic_bp_input,
        "diastolic_bp": diastolic_bp_input,
        "total_cholesterol": total_chol_input,
        "ldl": ldl_input,
        "hdl": hdl_input,
        "triglycerides": trig_input,
        "fasting_blood_sugar": fbs_input,
        "hba1c": hba1c_input,
        "height_cm": height_input,
        "weight_kg": weight_input,
        "bmi": computed_bmi,
        "smoking_status": smoking_input,
        "diabetes_history": diabetes_input,
        "family_history_heart_disease": fam_hist_input,
        "physical_activity": activity_input
    }
    st.session_state["patient_inputs"] = ui_inputs

    model_inputs = {
        "age": age_input,
        "sex": sex_input,
        "systolic_bp": systolic_bp_input,
        "diastolic_bp": diastolic_bp_input,
        "total_cholesterol": total_chol_input,
        "ldl": ldl_input,
        "hdl": hdl_input,
        "triglycerides": trig_input,
        "fasting_blood_sugar": fbs_input,
        "hba1c": hba1c_input,
        "height_cm": height_input,
        "weight_kg": weight_input,
        "bmi": computed_bmi,
        "smoking_status": smoking_input,
        "diabetes_history": "true" if diabetes_input == "Yes" else "false",
        "family_history_heart_disease": "true" if fam_hist_input == "Yes" else "false",
        "physical_activity": activity_input
    }

    # Validate inputs
    is_valid, validation_errors = validate_patient_input(model_inputs)
    if not is_valid:
        st.error("Input validation failed:\n\n" + "\n".join(f"- {e}" for e in validation_errors))
    else:
        with st.spinner("Analyzing checkup report parameters through trained pipeline..."):
            active_model = all_pipelines.get(st.session_state["selected_model_name"], best_pipeline)
            try:
                prediction_res = predict_heart_disease(active_model, model_inputs)
                st.session_state["prediction_result"] = prediction_res
            except Exception as e:
                st.error(f"Inference error: {e!s}")


# ==========================================
# 4. PREDICTION RESULTS DISPLAY SECTION
# ==========================================
if st.session_state.get("prediction_result"):
    res = st.session_state["prediction_result"]
    is_positive = (res["prediction_class"] == 1)

    badge_class = "badge-danger" if is_positive else "badge-success"
    status_icon = "⚠️" if is_positive else "✅"

    st.markdown('<hr class="custom-divider"/>', unsafe_allow_html=True)
    card_header = (
        f'<div class="result-card">'
        f'<div class="result-badge {badge_class}">'
        f'{status_icon} {res["status_label"]} — {res["risk_level"]} Risk ({res["probability_disease"]}% Cardiovascular Likelihood)'
        f'</div>'
        f'<div style="font-size: 13.5px; color: #F5F7FA; line-height: 1.5; margin-bottom: 12px;">'
        f'{res["narrative"]}'
        f'</div></div>'
    )
    st.markdown(card_header, unsafe_allow_html=True)

    # Risk gauge
    gauge_fig = create_probability_gauge_figure(res["probability_disease"])
    st.plotly_chart(gauge_fig, width="stretch", config={"displayModeBar": False})

    # Contributing clinical risk factors
    if res["risk_factors"]:
        st.markdown("<div style='font-size: 12.5px; font-weight: 600; color: #AAB3C2; margin-top: 10px; margin-bottom: 6px;'>Key Checkup Report Risk Observations:</div>", unsafe_allow_html=True)
        for rf in res["risk_factors"]:
            st.markdown(f"<div style='font-size: 12px; color: #E2E8F0; margin-bottom: 3px;'>• {rf}</div>", unsafe_allow_html=True)

    # Disclaimer
    disclaimer_html = (
        f'<div class="disclaimer-box">'
        f'<b>Notice:</b> {res["disclaimer"]}'
        f'</div>'
    )
    st.markdown(disclaimer_html, unsafe_allow_html=True)


# ==========================================
# 5. MODEL INSIGHTS SECTION
# ==========================================
st.markdown('<hr class="custom-divider"/>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Model Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size: 12.5px; color: #AAB3C2; margin-top: -8px; margin-bottom: 14px;">'
    'Evaluation metrics across routine health checkup parameters and model feature importance weights.'
    '</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🧠 Feature Importance", "📈 Confusion Matrix"])

active_model_name = st.session_state["selected_model_name"]

with tab1:
    st.markdown(
        '<div style="font-size: 13.5px; font-weight: 600; color: #F5F7FA; margin-bottom: 6px;">'
        'Machine Learning Model Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    comparison_rows = metadata.get("comparison_table", [])
    rows_html = []
    for row in comparison_rows:
        is_best = (row["Model"] == best_model_name)
        tag = '<span class="best-tag">Selected Best</span>' if is_best else ""
        spec = row.get("Specificity", 0.0)
        roc = row.get("ROC-AUC", 0.0)
        rows_html.append(
            f'<tr><td style="font-weight: 600;">{row["Model"]}{tag}</td>'
            f'<td style="text-align: right;">{row["Accuracy"]:.2f}%</td>'
            f'<td style="text-align: right;">{row["Precision"]:.2f}%</td>'
            f'<td style="text-align: right;">{row["Recall"]:.2f}%</td>'
            f'<td style="text-align: right;">{spec:.2f}%</td>'
            f'<td style="text-align: right;">{row["F1 Score"]:.2f}%</td>'
            f'<td style="text-align: right;">{roc:.2f}%</td></tr>'
        )
    rows_body = "".join(rows_html)
    table_html = (
        '<div class="comparison-table-wrapper">'
        '<table class="comparison-table">'
        '<thead><tr>'
        '<th>Model</th>'
        '<th style="text-align: right;">Accuracy</th>'
        '<th style="text-align: right;">Precision</th>'
        '<th style="text-align: right;">Recall</th>'
        '<th style="text-align: right;">Specificity</th>'
        '<th style="text-align: right;">F1 Score</th>'
        '<th style="text-align: right;">ROC-AUC</th>'
        '</tr></thead>'
        f'<tbody>{rows_body}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size: 11.5px; color: #8E99AB; line-height: 1.4; margin-top: 6px;">'
        '• Metrics calculated on a strictly held-out 20% test partition (N = 600) with stratified class distribution.<br/>'
        '• Binary metrics with positive class = 1 (Cardiovascular risk indicated). Evaluated on full-body checkup features.'
        '</div>',
        unsafe_allow_html=True
    )


with tab2:
    model_opts = list(all_pipelines.keys())
    sel_model_feat = st.selectbox(
        "Select Model for Feature Importance Analysis",
        options=model_opts,
        index=model_opts.index(best_model_name) if best_model_name in model_opts else 0,
        key="sel_model_feat"
    )

    model_detail = metadata.get("model_details", {}).get(sel_model_feat, {})
    feat_data = model_detail.get("feature_importance", {})
    agg_feat = feat_data.get("aggregated", {})
    is_coef = feat_data.get("is_coef", False)

    if agg_feat:
        fig_feat = create_feature_importance_figure(agg_feat, model_name=sel_model_feat, is_coef=is_coef)
        st.plotly_chart(fig_feat, width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div style="font-size: 11.5px; color: #8E99AB; line-height: 1.4; margin-top: 4px;">'
            '• Values represent aggregated predictive weights of routine checkup parameters across one-hot encoded categories.<br/>'
            '• Relative weights indicate statistical contribution to the model and do not constitute independent diagnoses.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Feature importance data is unavailable for the selected classifier.")


with tab3:
    model_opts_cm = list(all_pipelines.keys())
    sel_model_cm = st.selectbox(
        "Select Model for Confusion Matrix Inspection",
        options=model_opts_cm,
        index=model_opts_cm.index(best_model_name) if best_model_name in model_opts_cm else 0,
        key="sel_model_cm"
    )

    model_detail_cm = metadata.get("model_details", {}).get(sel_model_cm, {})
    cm_data = model_detail_cm.get("confusion_matrix", {})

    if cm_data:
        fig_cm = create_confusion_matrix_figure(cm_data, model_name=sel_model_cm)
        st.plotly_chart(fig_cm, width="stretch", config={"displayModeBar": False})
        st.markdown(
            f'<div style="font-size: 11.5px; color: #8E99AB; line-height: 1.4; margin-top: 4px;">'
            f'• Evaluated on {cm_data["total"]} independent test cases.<br/>'
            f'• Correct Classifications: {cm_data["tn"] + cm_data["tp"]} ({((cm_data["tn"] + cm_data["tp"]) / cm_data["total"] * 100):.1f}%) | '
            f'Misclassifications: {cm_data["fp"] + cm_data["fn"]} ({((cm_data["fp"] + cm_data["fn"]) / cm_data["total"] * 100):.1f}%)'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Confusion matrix data is unavailable for the selected classifier.")
