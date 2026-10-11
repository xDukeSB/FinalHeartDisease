"""Patient full-body checkup validation, risk prediction, probability calibration, and clinical narrative."""

from typing import Any

import pandas as pd

from src.preprocessing import ALL_FEATURE_COLUMNS, normalize_dataframe

# Clinical parameter bounds for routine checkup report validation
VALIDATION_RANGES = {
    "age": (18, 110, "Age must be between 18 and 110 years."),
    "systolic_bp": (70, 240, "Systolic BP must be between 70 and 240 mmHg."),
    "diastolic_bp": (40, 140, "Diastolic BP must be between 40 and 140 mmHg."),
    "total_cholesterol": (80, 450, "Total Cholesterol must be between 80 and 450 mg/dL."),
    "ldl": (30, 300, "LDL Cholesterol must be between 30 and 300 mg/dL."),
    "hdl": (15, 120, "HDL Cholesterol must be between 15 and 120 mg/dL."),
    "triglycerides": (30, 600, "Triglycerides must be between 30 and 600 mg/dL."),
    "fasting_blood_sugar": (50, 350, "Fasting Blood Sugar must be between 50 and 350 mg/dL."),
    "hba1c": (3.5, 15.0, "HbA1c must be between 3.5% and 15.0%."),
    "height_cm": (120.0, 220.0, "Height must be between 120 and 220 cm."),
    "weight_kg": (30.0, 250.0, "Weight must be between 30 and 250 kg."),
    "bmi": (14.0, 65.0, "BMI must be between 14.0 and 65.0 kg/m².")
}

VALID_CATEGORIES = {
    "sex": ["male", "female"],
    "smoking_status": ["never", "former", "current"],
    "diabetes_history": ["true", "false", "yes", "no"],
    "family_history_heart_disease": ["true", "false", "yes", "no"],
    "physical_activity": ["sedentary", "moderate", "active"]
}


def validate_patient_input(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate numerical limits and categorical options of patient checkup data."""
    errors = []

    # Check required keys
    for feat in ALL_FEATURE_COLUMNS:
        if feat not in data or data[feat] is None or str(data[feat]).strip() == "":
            errors.append(f"Missing required clinical field: {feat}")

    if errors:
        return False, errors

    # Numerical range checks
    for field, (low, high, msg) in VALIDATION_RANGES.items():
        try:
            val = float(data[field])
            if val < low or val > high:
                errors.append(f"{msg} Received: {val}")
        except (ValueError, TypeError):
            errors.append(f"Field '{field}' must be a valid number.")

    # Categorical option checks
    for field, allowed in VALID_CATEGORIES.items():
        val_str = str(data[field]).strip().lower()
        if val_str not in allowed:
            errors.append(f"Invalid option for '{field}': '{data[field]}'. Expected one of {allowed}")

    # Check blood pressure plausibility
    try:
        sbp = float(data["systolic_bp"])
        dbp = float(data["diastolic_bp"])
        if sbp <= dbp:
            errors.append(f"Systolic BP ({sbp:.0f}) must be higher than Diastolic BP ({dbp:.0f}).")
    except Exception:
        pass

    return len(errors) == 0, errors


def prepare_input_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """Transform user checkup inputs into preprocessor-compatible DataFrame."""
    row = {}
    for feat in ALL_FEATURE_COLUMNS:
        val = data[feat]
        if feat in ["diabetes_history", "family_history_heart_disease"]:
            val_str = str(val).strip().lower()
            val = "true" if val_str in ["yes", "true", "1"] else "false"
        elif feat in ["sex", "smoking_status", "physical_activity"]:
            val = str(val).strip().title()
        row[feat] = val

    # Ensure BMI matches height and weight
    try:
        h_m = float(row["height_cm"]) / 100.0
        w_kg = float(row["weight_kg"])
        row["bmi"] = round(w_kg / (h_m ** 2), 1)
    except Exception:
        pass

    df = pd.DataFrame([row])
    return normalize_dataframe(df)


def identify_clinical_risk_factors(data: dict[str, Any]) -> list[str]:
    """Identify key physiological and lifestyle flags based on ACC/AHA clinical guidelines."""
    flags = []
    try:
        age = float(data.get("age", 0))
        sbp = float(data.get("systolic_bp", 0))
        dbp = float(data.get("diastolic_bp", 0))
        tc = float(data.get("total_cholesterol", 0))
        ldl = float(data.get("ldl", 0))
        hdl = float(data.get("hdl", 0))
        trig = float(data.get("triglycerides", 0))
        fbs = float(data.get("fasting_blood_sugar", 0))
        hba1c = float(data.get("hba1c", 0))
        bmi = float(data.get("bmi", 0))
        sex = str(data.get("sex", "")).lower()

        smoking = str(data.get("smoking_status", "")).lower()
        diabetes = str(data.get("diabetes_history", "")).lower() in ["yes", "true", "1"]
        fam_hist = str(data.get("family_history_heart_disease", "")).lower() in ["yes", "true", "1"]
        activity = str(data.get("physical_activity", "")).lower()

        # Blood pressure
        if sbp >= 140 or dbp >= 90:
            flags.append(f"Stage 2 Hypertension ({sbp:.0f}/{dbp:.0f} mmHg)")
        elif sbp >= 130 or dbp >= 80:
            flags.append(f"Stage 1 Hypertension ({sbp:.0f}/{dbp:.0f} mmHg)")

        # Lipids
        if tc >= 240:
            flags.append(f"High Serum Cholesterol ({tc:.0f} mg/dL >= 240)")
        elif tc >= 200:
            flags.append(f"Borderline High Cholesterol ({tc:.0f} mg/dL)")

        if ldl >= 160:
            flags.append(f"Elevated Atherogenic LDL ({ldl:.0f} mg/dL >= 160)")
        elif ldl >= 130:
            flags.append(f"Borderline High LDL ({ldl:.0f} mg/dL)")

        if (sex == "male" and hdl < 40) or (sex == "female" and hdl < 50):
            flags.append(f"Suboptimal Protective HDL ({hdl:.0f} mg/dL < 40/50 threshold)")

        if trig >= 200:
            flags.append(f"Hypertriglyceridemia ({trig:.0f} mg/dL >= 200)")
        elif trig >= 150:
            flags.append(f"Borderline High Triglycerides ({trig:.0f} mg/dL)")

        # Glucose & HbA1c
        if diabetes or fbs >= 126 or hba1c >= 6.5:
            flags.append(f"Diabetic Glycemic Range (FBS: {fbs:.0f} mg/dL, HbA1c: {hba1c:.1f}%)")
        elif fbs >= 100 or hba1c >= 5.7:
            flags.append(f"Prediabetic Range (FBS: {fbs:.0f} mg/dL, HbA1c: {hba1c:.1f}%)")

        # Adiposity
        if bmi >= 30.0:
            flags.append(f"Class I+ Obesity (BMI: {bmi:.1f} kg/m² >= 30.0)")
        elif bmi >= 25.0:
            flags.append(f"Overweight BMI (BMI: {bmi:.1f} kg/m²)")

        # Lifestyle & History
        if smoking == "current":
            flags.append("Active Tobacco Smoker (substantially increases endothelial plaque risk)")
        elif smoking == "former":
            flags.append("Former Smoker history")

        if fam_hist:
            flags.append("Documented Family History of Premature Cardiovascular Disease")

        if activity == "sedentary":
            flags.append("Sedentary Lifestyle (insufficient regular aerobic exercise)")

        if age >= 65:
            flags.append(f"Senior Age Bracket ({age:.0f} years old)")

    except Exception:
        pass

    return flags


def predict_heart_disease(
    pipeline: Any,
    input_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Execute end-to-end inference on routine health checkup inputs.
    Returns prediction class, probability, risk category, clinical flags,
    and narrative assessment.
    """
    is_valid, errors = validate_patient_input(input_data)
    if not is_valid:
        raise ValueError("Invalid checkup inputs:\n" + "\n".join(errors))

    df_input = prepare_input_dataframe(input_data)

    pred_class = int(pipeline.predict(df_input)[0])

    probabilities = None
    prob_disease = None
    prob_healthy = None

    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(df_input)[0]
            prob_healthy = round(float(proba[0]) * 100, 1)
            prob_disease = round(float(proba[1]) * 100, 1)
            probabilities = {
                "no_disease": prob_healthy,
                "disease": prob_disease
            }
        except Exception:
            pass

    # Risk categorization
    if prob_disease is not None:
        if prob_disease >= 65.0:
            risk_level = "High"
            risk_color = "#E74C3C"
        elif prob_disease >= 35.0:
            risk_level = "Moderate"
            risk_color = "#F39C12"
        else:
            risk_level = "Low"
            risk_color = "#2ECC71"
    else:
        risk_level = "High" if pred_class == 1 else "Low"
        risk_color = "#E74C3C" if pred_class == 1 else "#2ECC71"

    status_label = "Elevated Cardiovascular Risk Indicated" if pred_class == 1 else "Normal Cardiovascular Baseline"
    risk_factors = identify_clinical_risk_factors(input_data)

    if pred_class == 1:
        narrative = (
            f"Based on your routine health checkup parameters, the trained machine learning model indicates "
            f"an elevated cardiovascular risk profile ({prob_disease}% estimated likelihood). "
            f"Combinations of blood pressure, lipid balance (LDL/HDL/Triglycerides), glycemic status, and lifestyle factors "
            f"contribute to this assessment. We recommend discussing these findings with your physician."
        )
    else:
        narrative = (
            f"Based on your routine health checkup parameters, the trained model indicates a favorable "
            f"cardiovascular risk baseline ({prob_healthy}% healthy likelihood). "
            f"Key metrics including blood pressure, lipid fractions, and glycemic markers currently align "
            f"with lower statistical likelihood of cardiovascular events. Continue regular checkups."
        )

    return {
        "prediction_class": pred_class,
        "status_label": status_label,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "probability_disease": prob_disease,
        "probability_healthy": prob_healthy,
        "probabilities": probabilities,
        "narrative": narrative,
        "risk_factors": risk_factors,
        "disclaimer": (
            "Medical Disclaimer: This application is an educational cardiovascular risk screening prototype "
            "designed for health-checkup report analysis. It does NOT diagnose heart disease, exclude cardiovascular conditions, "
            "or substitute for clinical evaluation by a qualified medical professional. Always consult your doctor for medical advice."
        )
    }
