"""Generate medically plausible synthetic routine full-body checkup dataset.

Dataset: full_body_heart_checkup_synthetic_3000.csv
Based on Framingham / ACC/AHA ASCVD epidemiological cardiovascular risk parameters.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd

COLUMNS_ORDER = [
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


def generate_checkup_dataset(
    n_samples: int = 3000,
    seed: int = 42,
    introduce_missing: bool = True
) -> pd.DataFrame:
    """Generate realistic, diverse records reflecting routine health checkup reports."""
    np.random.seed(seed)

    ids = np.arange(1, n_samples + 1)

    # 1. Age: 22 to 82 years (mean ~52, realistic checkup demographic)
    age = np.clip(np.random.normal(loc=52.0, scale=12.5, size=n_samples).astype(int), 22, 84)

    # 2. Sex: Male (52%), Female (48%)
    sex = np.random.choice(["Male", "Female"], size=n_samples, p=[0.52, 0.48])
    is_male = (sex == "Male").astype(float)

    # 3. Height & Weight -> BMI
    # Height (cm): Male ~ N(174, 7), Female ~ N(162, 6)
    height_cm = np.where(
        is_male == 1,
        np.random.normal(174.0, 7.0, size=n_samples),
        np.random.normal(162.0, 6.0, size=n_samples)
    )
    height_cm = np.clip(np.round(height_cm, 1), 145.0, 202.0)

    # Base BMI ~ N(26.8, 4.8), slightly higher with age
    base_bmi = np.random.normal(26.5, 4.5, size=n_samples) + (age - 50) * 0.04
    bmi = np.clip(np.round(base_bmi, 1), 17.5, 46.0)

    # Calculate weight from height and target BMI: weight = BMI * (height/100)^2
    weight_kg = np.round(bmi * ((height_cm / 100.0) ** 2), 1)
    # Re-calculate exact BMI to prevent any floating point divergence
    bmi = np.round(weight_kg / ((height_cm / 100.0) ** 2), 1)

    # 4. Lifestyle & Medical History
    # Smoking status
    smoking_status = np.random.choice(
        ["Never", "Former", "Current"],
        size=n_samples,
        p=[0.55, 0.26, 0.19]
    )
    is_current_smoker = (smoking_status == "Current").astype(float)
    is_former_smoker = (smoking_status == "Former").astype(float)

    # Physical Activity
    physical_activity = np.random.choice(
        ["Sedentary", "Moderate", "Active"],
        size=n_samples,
        p=[0.42, 0.44, 0.14]
    )
    is_sedentary = (physical_activity == "Sedentary").astype(float)

    # Family history of heart disease
    family_history_heart_disease = np.random.choice(
        [True, False],
        size=n_samples,
        p=[0.27, 0.73]
    )

    # Diabetes history: correlated with BMI and age
    diabetes_prob = 1.0 / (1.0 + np.exp(-(
        -4.2 + 0.03 * age + 0.08 * (bmi - 25) + 0.5 * is_sedentary
    )))
    diabetes_history = np.random.binomial(1, diabetes_prob).astype(bool)

    # 5. Glycemic Profile: Fasting Glucose (mg/dL) & HbA1c (%)
    # Non-diabetic glucose ~ N(92, 11), diabetic ~ N(142, 32)
    fasting_blood_sugar = np.where(
        diabetes_history,
        np.random.normal(142.0, 28.0, size=n_samples),
        np.random.normal(92.0, 11.0, size=n_samples) + (bmi - 25) * 0.4
    )
    fasting_blood_sugar = np.clip(np.round(fasting_blood_sugar, 1), 65.0, 310.0)

    # HbA1c (%) tightly aligned with glucose: HbA1c = (glucose + 46.7) / 28.7 + noise
    hba1c_calc = (fasting_blood_sugar + 46.7) / 28.7 + np.random.normal(0, 0.25, size=n_samples)
    hba1c = np.clip(np.round(hba1c_calc, 1), 4.3, 12.8)

    # 6. Blood Pressure: Systolic & Diastolic (mmHg)
    # Influenced by age, BMI, sex, and smoking
    sys_base = 112.0 + 0.38 * (age - 30) + 0.75 * (bmi - 24) + 4.0 * is_male + 3.0 * is_current_smoker
    systolic_bp = np.clip(np.round(np.random.normal(sys_base, 11.0)), 90, 210).astype(int)

    dia_base = 72.0 + 0.18 * (age - 30) + 0.45 * (bmi - 24) + 0.22 * (systolic_bp - 120)
    diastolic_bp = np.clip(np.round(np.random.normal(dia_base, 7.5)), 55, 125).astype(int)
    # Ensure pulse pressure >= 20 mmHg
    systolic_bp = np.maximum(systolic_bp, diastolic_bp + 20)

    # 7. Lipid Profile (mg/dL)
    # Triglycerides: log-normal distribution, higher with BMI, sedentary lifestyle, diabetes
    trig_log_mean = np.log(130.0) + 0.02 * (bmi - 25) + 0.15 * diabetes_history.astype(float) + 0.1 * is_current_smoker
    triglycerides = np.clip(np.round(np.random.lognormal(trig_log_mean, 0.38)), 45, 480).astype(int)

    # HDL: higher in females, lower in smokers/sedentary/high BMI
    hdl_mean = 52.0 + 8.0 * (1.0 - is_male) - 0.35 * (bmi - 25) - 4.5 * is_current_smoker - 3.0 * is_sedentary
    hdl = np.clip(np.round(np.random.normal(hdl_mean, 9.5)), 24, 98).astype(int)

    # LDL: correlated with age, BMI
    ldl_mean = 118.0 + 0.3 * (age - 45) + 0.7 * (bmi - 25)
    ldl = np.clip(np.round(np.random.normal(ldl_mean, 26.0)), 45, 235).astype(int)

    # Total Cholesterol: approximately Friedewald relationship (Total = LDL + HDL + Trig/5) + biological variation
    total_cholesterol = ldl + hdl + (triglycerides / 5.0) + np.random.normal(0, 10.0, size=n_samples)
    total_cholesterol = np.clip(np.round(total_cholesterol), 110, 370).astype(int)

    # 8. Medically Grounded Stochastic Cardiovascular Risk Score (Outcome Label)
    # Using clinical risk weights derived from Framingham / ASCVD Risk Equations
    # Note: Includes significant unexplained stochastic variance so it is non-deterministic
    risk_logit = (
        -1.65
        + 0.055 * (age - 50)
        + 0.45 * is_male
        + 0.025 * (systolic_bp - 120)
        + 0.010 * (total_cholesterol - 190)
        - 0.032 * (hdl - 50)
        + 0.003 * (triglycerides - 130)
        + 0.050 * (bmi - 25)
        + 0.65 * is_current_smoker
        + 0.20 * is_former_smoker
        + 0.65 * diabetes_history.astype(float)
        + 0.60 * family_history_heart_disease.astype(float)
        + 0.40 * is_sedentary
        + np.random.normal(0, 0.75, size=n_samples)  # Clinically realistic unobserved variance
    )

    disease_probability = 1.0 / (1.0 + np.exp(-risk_logit))
    heart_disease_label = np.random.binomial(1, disease_probability).astype(int)

    # Build DataFrame
    df = pd.DataFrame({
        "id": ids,
        "age": age,
        "sex": sex,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "total_cholesterol": total_cholesterol,
        "ldl": ldl,
        "hdl": hdl,
        "triglycerides": triglycerides,
        "fasting_blood_sugar": fasting_blood_sugar,
        "hba1c": hba1c,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "smoking_status": smoking_status,
        "diabetes_history": diabetes_history,
        "family_history_heart_disease": family_history_heart_disease,
        "physical_activity": physical_activity,
        "heart_disease_label": heart_disease_label
    })

    # Ensure exact column order
    df = df[COLUMNS_ORDER]

    # 9. Realistic occasional missing values (simulating omitted routine lab tests)
    if introduce_missing:
        # HbA1c occasionally omitted if fasting glucose is standard (~6% missing)
        mask_hba1c = np.random.rand(n_samples) < 0.06
        df.loc[mask_hba1c, "hba1c"] = np.nan

        # Triglycerides or LDL occasionally not separately reported (~4% missing)
        mask_trig = np.random.rand(n_samples) < 0.04
        df.loc[mask_trig, "triglycerides"] = np.nan

        mask_ldl = np.random.rand(n_samples) < 0.03
        df.loc[mask_ldl, "ldl"] = np.nan

        mask_fbs = np.random.rand(n_samples) < 0.02
        df.loc[mask_fbs, "fasting_blood_sugar"] = np.nan

    return df


def validate_checkup_dataset(df: pd.DataFrame) -> Tuple[bool, list]:
    """Validate dataset meets all design requirements."""
    issues = []

    # 1. Row count
    if len(df) < 3000:
        issues.append(f"Insufficient row count: {len(df)} < 3000")

    # 2. Columns & exact order
    if list(df.columns) != COLUMNS_ORDER:
        issues.append(f"Columns do not match required order:\nExpected: {COLUMNS_ORDER}\nGot: {list(df.columns)}")

    # 3. Unique IDs
    if df["id"].nunique() != len(df):
        issues.append("Record IDs are not unique.")

    # 4. BMI consistency: weight / (height/100)^2
    expected_bmi = np.round(df["weight_kg"] / ((df["height_cm"] / 100.0) ** 2), 1)
    bmi_diff = np.abs(df["bmi"] - expected_bmi)
    if (bmi_diff > 0.15).any():
        issues.append("BMI is inconsistent with height and weight.")

    # 5. Sex categories
    if not set(df["sex"].unique()).issubset({"Male", "Female"}):
        issues.append(f"Unexpected sex categories: {df['sex'].unique()}")

    # 6. Smoking categories
    if not set(df["smoking_status"].unique()).issubset({"Never", "Former", "Current"}):
        issues.append(f"Unexpected smoking status values: {df['smoking_status'].unique()}")

    # 7. Physical activity categories
    if not set(df["physical_activity"].unique()).issubset({"Sedentary", "Moderate", "Active"}):
        issues.append(f"Unexpected physical activity values: {df['physical_activity'].unique()}")

    # 8. Target outcome
    if not set(df["heart_disease_label"].unique()).issubset({0, 1}):
        issues.append(f"Unexpected target labels: {df['heart_disease_label'].unique()}")

    # 9. Plausible ranges
    if (df["systolic_bp"] <= df["diastolic_bp"]).any():
        issues.append("Systolic BP must strictly exceed Diastolic BP.")

    return len(issues) == 0, issues


def main():
    print("=====================================================================")
    print(" 🏥 Synthetic Full-Body Health Checkup Dataset Generator ")
    print("=====================================================================")
    print("Generating 3,000 diverse, medically plausible patient records...")

    df = generate_checkup_dataset(n_samples=3000, seed=42, introduce_missing=True)

    is_valid, issues = validate_checkup_dataset(df)
    if not is_valid:
        print("Validation errors detected:")
        for issue in issues:
            print(f"  ❌ {issue}")
        raise ValueError("Generated dataset failed validation criteria.")

    print("✅ All validation checks passed!")
    print(f"   Shape: {df.shape}")
    print(f"   Disease prevalence: {df['heart_disease_label'].mean() * 100:.1f}%")
    print(f"   Missing counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Save to both data/ directory and root for direct accessibility
    os.makedirs("data", exist_ok=True)
    csv_filename = "full_body_heart_checkup_synthetic_3000.csv"
    data_path = os.path.join("data", csv_filename)

    df.to_csv(csv_filename, index=False)
    df.to_csv(data_path, index=False)
    print(f"\nSaved CSV files:")
    print(f"  -> ./{csv_filename}")
    print(f"  -> ./{data_path}")


if __name__ == "__main__":
    main()
