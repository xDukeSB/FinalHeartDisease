# 🏥 Heart Disease Prediction Using Routine Full-Body Checkup Reports

An educational, machine learning-powered cardiovascular risk screening system designed around parameters commonly obtained from routine full-body health checkup reports or standard patient questionnaires.

Unlike clinical hospital datasets that mandate invasive or specialized cardiac investigations (such as cardiac fluoroscopy vessel counts, thallium myocardial perfusion scintigraphy, or exercise stress treadmill ECG testing), this system operates entirely on **routine clinical metrics**: blood pressure, complete lipid profile, glycemic markers, anthropometric body composition, and medical history.

---

## 1. Primary Objectives & Clinical Rationale

- **Accessible Health Screening:** Ordinary individuals can take routine health checkup lab reports (complete blood counts, lipid panels, diabetic profiles, and vitals) and obtain an educational cardiovascular risk assessment.
- **Educational Prototype:** Designed strictly for educational risk awareness. **It does NOT replace a doctor's diagnosis, emergency cardiac evaluation, or clinical care.**
- **Non-Invasive Feature Set:** 17 input attributes commonly recorded during annual or routine preventive physical examinations.

---

## 2. Dataset Overview: `full_body_heart_checkup_synthetic_3000.csv`

- **Record Count:** 3,000 diverse, non-duplicate patient observations.
- **Dataset Nature:** Medically plausible synthetic dataset generated for software engineering, reproducible pipeline testing, and educational cardiovascular risk demonstration.
- **Prevalence:** Approximately $42.7\%$ positive cardiovascular risk indication (1,281 positive, 1,719 negative).
- **Generator Script:** [`generate_dataset.py`](file:///c:/sagar/FinalHeartDisease/generate_dataset.py) (uses fixed seed `42` for reproducibility).

### Why a Synthetic Checkup Dataset was Generated
While historical open-access cohorts exist (such as the UCI Cleveland dataset or Framingham risk study), existing public datasets either:
1. Mandate specialized invasive/treadmill cardiac testing (`ca`, `thal`, `oldpeak`), rendering them unusable for ordinary people with routine blood test reports; or
2. Lack standard modern routine checkup parameters like complete lipid subfractions (`ldl`, `hdl`, `triglycerides`), height/weight/BMI, and `hba1c`.

Therefore, a medically plausible synthetic cohort was generated based on the **ACC/AHA ASCVD Risk Estimator** and **Framingham Heart Study** epidemiological equations, preserving genuine physiological relationships (such as the Friedewald lipid relationship, pulse pressure dynamics, and HbA1c/glucose alignment).

> ⚠️ **Limitation Notice:** Any generated target labels reflect simulated epidemiological probability functions and are **not clinically validated patient diagnoses**. A model trained on synthetic data cannot be used in actual clinical practice without prospective validation on real-world clinical cohort records.

---

## 3. Dataset Schema & Column Definitions

The dataset contains **exactly 19 columns** in the following specified order:

| # | Column Name | Data Type | Units | Description | Expected Range |
|---|---|---|---|---|---|
| 1 | `id` | Integer | None | Unique record identifier (excluded from predictive features) | 1 – 3,000 |
| 2 | `age` | Integer | Years | Patient age in completed years | 22 – 84 |
| 3 | `sex` | Categorical | None | Biological sex (`Male`, `Female`) | Male / Female |
| 4 | `systolic_bp` | Integer | mmHg | Systolic blood pressure | 90 – 210 |
| 5 | `diastolic_bp` | Integer | mmHg | Diastolic blood pressure | 55 – 125 |
| 6 | `total_cholesterol` | Integer | mg/dL | Serum total cholesterol | 110 – 370 |
| 7 | `ldl` | Integer | mg/dL | Low-density lipoprotein (atherogenic cholesterol) | 45 – 235 |
| 8 | `hdl` | Integer | mg/dL | High-density lipoprotein (protective cholesterol) | 24 – 98 |
| 9 | `triglycerides` | Integer | mg/dL | Fasting serum triglycerides | 45 – 480 |
| 10 | `fasting_blood_sugar` | Float | mg/dL | Fasting plasma glucose | 65.0 – 310.0 |
| 11 | `hba1c` | Float | % | Glycated hemoglobin (3-month glycemic indicator) | 4.3 – 12.8 |
| 12 | `height_cm` | Float | cm | Standing height | 145.0 – 202.0 |
| 13 | `weight_kg` | Float | kg | Body weight | 40.0 – 145.0 |
| 14 | `bmi` | Float | kg/m² | Body Mass Index ($weight / (height/100)^2$) | 17.5 – 46.0 |
| 15 | `smoking_status` | Categorical | None | Tobacco smoking tier (`Never`, `Former`, `Current`) | Never / Former / Current |
| 16 | `diabetes_history` | Boolean | None | Clinical diagnosis history of diabetes (`True`, `False`) | True / False |
| 17 | `family_history_heart_disease` | Boolean | None | First-degree premature CAD family history (`True`, `False`) | True / False |
| 18 | `physical_activity` | Categorical | None | Weekly aerobic exercise level (`Sedentary`, `Moderate`, `Active`) | Sedentary / Moderate / Active |
| 19 | `heart_disease_label` | Integer | Binary | Educational target label ($0 = \text{No indication}$, $1 = \text{Indication present}$) | 0 / 1 |

The full data dictionary is documented in [`data_dictionary.csv`](file:///c:/sagar/FinalHeartDisease/data_dictionary.csv).

---

## 4. Loading the Dataset in Python

```python
import pandas as pd

# Load the generated checkup dataset
df = pd.read_csv("full_body_heart_checkup_synthetic_3000.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 3 Records:")
print(df.head(3))
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values Count:")
print(df.isnull().sum()[df.isnull().sum() > 0])
```

---

## 5. Machine Learning Pipeline & Results

The system implements a scikit-learn `Pipeline` and `ColumnTransformer` with zero data leakage:
- **Numerical Pipeline:** Median imputation + standard feature scaling.
- **Categorical Pipeline:** Mode imputation + one-hot encoding (`handle_unknown="ignore"`).
- **Split:** Stratified 80/20 train/test partition ($N = 2,400$ training, $N = 600$ held-out test).

### Held-Out Evaluation Metrics ($N = 600$ Test Patients)

| Model | Accuracy | Precision | Recall | F1 Score | 5-Fold CV Accuracy | 5-Fold CV F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** *(Selected Best)* | **67.50%** | **63.56%** | **55.86%** | **59.46%** | 68.33% | 61.12% |
| **Random Forest** | **66.83%** | **63.90%** | **51.17%** | **56.83%** | 67.25% | 58.74% |
| **Decision Tree** | **64.33%** | **59.81%** | **50.00%** | **54.47%** | 64.92% | 56.40% |

---

## 6. Project Structure

```text
FinalHeartDisease/
├── full_body_heart_checkup_synthetic_3000.csv  # 3,000-record routine checkup dataset
├── data_dictionary.csv                         # Data dictionary and acceptable ranges
├── generate_dataset.py                         # Standalone generator and validator script
├── app.py                                      # Streamlit web application matching dark theme
├── train_model.py                              # Reproducible ML training pipeline CLI
├── requirements.txt                            # Dependencies
├── README.md                                   # Project documentation
├── models/
│   ├── best_model.joblib                       # Top-performing pipeline (Logistic Regression)
│   ├── all_models.joblib                       # All trained pipelines
│   └── model_metadata.json                     # Training metrics, schema, and confusion matrices
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                        # Imputation, scaling, and categorical encoding
│   ├── training.py                             # 5-fold CV and pipeline fitting
│   ├── prediction.py                           # Checkup bounds validation and inference
│   ├── evaluation.py                           # Metric computation and feature aggregation
│   └── visualization.py                        # Plotly dark-themed charts
└── tests/
    ├── test_preprocessing.py                   # Preprocessing unit tests
    ├── test_prediction.py                      # Checkup validation unit tests
    ├── test_evaluation.py                      # Evaluation metric unit tests
    └── test_ui_app.py                          # Streamlit UI integration tests
```

---

## 7. How to Run the Application

```powershell
# 1. Activate the environment
.\.venv\Scripts\activate

# 2. (Optional) Re-generate the synthetic dataset
python generate_dataset.py

# 3. (Optional) Re-train the machine learning models
python train_model.py

# 4. Run automated test suite
pytest tests/

# 5. Launch the Streamlit web application
streamlit run app.py
```

The application runs locally at: **`http://localhost:8501`**.

---

## 8. Medical & Ethical Disclaimer

> **IMPORTANT NOTICE:** This application and synthetic dataset are designed exclusively for educational, technical, and analytical review. The prediction outputs do not constitute a medical diagnosis, clinical evaluation, or treatment plan. A routine health checkup alone cannot diagnose or exclude cardiovascular disease. Anyone with cardiovascular symptoms (e.g., chest discomfort, shortness of breath, palpitations) should seek immediate medical evaluation from a licensed healthcare professional.
