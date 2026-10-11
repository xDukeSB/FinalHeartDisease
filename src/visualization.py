"""Plotly visualization components customized for dark mode theme."""

from typing import Any

import plotly.graph_objects as go

DARK_THEME_LAYOUT = {
    "paper_bgcolor": "#0B0E14",
    "plot_bgcolor": "#0B0E14",
    "font": {
        "color": "#F5F7FA",
        "family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
        "size": 12
    },
    "margin": {"l": 20, "r": 20, "t": 35, "b": 30}
}

CLINICAL_FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)",
    "total_cholesterol": "Total Cholesterol (mg/dL)",
    "ldl": "LDL Cholesterol (mg/dL)",
    "hdl": "HDL Cholesterol (mg/dL)",
    "triglycerides": "Triglycerides (mg/dL)",
    "fasting_blood_sugar": "Fasting Blood Sugar (mg/dL)",
    "hba1c": "HbA1c (%)",
    "height_cm": "Height (cm)",
    "weight_kg": "Weight (kg)",
    "bmi": "BMI (kg/m²)",
    "smoking_status": "Smoking Status",
    "diabetes_history": "Diabetes History",
    "family_history_heart_disease": "Family History of CAD",
    "physical_activity": "Physical Activity Tier"
}


def create_feature_importance_figure(
    feature_importance_dict: dict[str, float],
    model_name: str = "Random Forest",
    is_coef: bool = False
) -> go.Figure:
    """Create a sleek, dark-themed horizontal bar chart of feature importance."""
    # Format labels and values
    features = list(feature_importance_dict.keys())
    values = [float(v) * 100 for v in feature_importance_dict.values()]  # percentage

    # Translate to friendly clinical labels
    pretty_labels = [CLINICAL_FEATURE_LABELS.get(f, f.replace("cat__", "").replace("num__", "").replace("_", " ").title()) for f in features]

    # Invert so highest appears at top
    pretty_labels = pretty_labels[::-1]
    values = values[::-1]

    metric_name = "Absolute Coefficient Weight (%)" if is_coef else "Relative Predictive Importance (%)"

    fig = go.Figure(go.Bar(
        x=values,
        y=pretty_labels,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=[
                [0.0, "#1F3A60"],
                [0.5, "#2E86DE"],
                [1.0, "#54A0FF"]
            ],
            line=dict(color="#303540", width=1)
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        height=420,
        title=dict(
            text=f"<b>{model_name} — Feature Importance</b>",
            font=dict(size=14, color="#F5F7FA"),
            x=0.0,
            xanchor="left"
        ),
        xaxis=dict(
            title=dict(text=metric_name, font=dict(size=11, color="#AAB3C2")),
            gridcolor="#252833",
            showgrid=True,
            zeroline=False,
            tickfont=dict(color="#AAB3C2", size=10)
        ),
        yaxis=dict(
            gridcolor="#151922",
            showgrid=False,
            tickfont=dict(color="#F5F7FA", size=11)
        )
    )

    return fig


def create_confusion_matrix_figure(
    cm_data: dict[str, Any],
    model_name: str = "Selected Model"
) -> go.Figure:
    """Create an annotated, high-contrast dark heatmap confusion matrix."""
    matrix = cm_data["matrix"]
    total = cm_data["total"]

    tn, fp = matrix[0][0], matrix[0][1]
    fn, tp = matrix[1][0], matrix[1][1]

    # Cell text annotations
    text_annotations = [
        [
            f"<b>True Negative (TN)</b><br><span style='font-size:18px;'>{tn}</span><br>({tn/total*100:.1f}%)",
            f"<b>False Positive (FP)</b><br><span style='font-size:18px;'>{fp}</span><br>({fp/total*100:.1f}%)"
        ],
        [
            f"<b>False Negative (FN)</b><br><span style='font-size:18px;'>{fn}</span><br>({fn/total*100:.1f}%)",
            f"<b>True Positive (TP)</b><br><span style='font-size:18px;'>{tp}</span><br>({tp/total*100:.1f}%)"
        ]
    ]

    x_labels = ["Predicted: No Disease", "Predicted: Heart Disease"]
    y_labels = ["Actual: No Disease", "Actual: Heart Disease"]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=x_labels,
        y=y_labels,
        text=text_annotations,
        texttemplate="%{text}",
        textfont=dict(size=12, color="#FFFFFF", family="Inter"),
        colorscale=[
            [0.0, "#161B26"],
            [0.35, "#1E3A5F"],
            [0.7, "#285F9E"],
            [1.0, "#2E86DE"]
        ],
        showscale=False,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Count: %{z}<extra></extra>"
    ))

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        height=340,
        title=dict(
            text=f"<b>{model_name} — Confusion Matrix (N = {total} Test Patients)</b>",
            font=dict(size=13, color="#F5F7FA"),
            x=0.0,
            xanchor="left"
        ),
        xaxis=dict(
            tickfont=dict(color="#F5F7FA", size=11),
            side="bottom"
        ),
        yaxis=dict(
            tickfont=dict(color="#F5F7FA", size=11),
            autorange="reversed"
        )
    )

    return fig


def create_probability_gauge_figure(prob_disease: float) -> go.Figure:
    """Create a compact, modern horizontal probability risk gauge."""
    if prob_disease >= 65:
        bar_color = "#E74C3C"
        status_text = "High Likelihood"
    elif prob_disease >= 35:
        bar_color = "#F39C12"
        status_text = "Moderate Likelihood"
    else:
        bar_color = "#2ECC71"
        status_text = "Low Likelihood"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_disease,
        number={"suffix": "%", "font": {"size": 28, "color": "#F5F7FA"}},
        title={"text": f"<b>Disease Probability</b><br><span style='color:{bar_color}; font-size:12px;'>{status_text}</span>", "font": {"size": 13, "color": "#AAB3C2"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#303540", "tickfont": {"color": "#AAB3C2", "size": 10}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#161B26",
            "borderwidth": 1,
            "bordercolor": "#303540",
            "steps": [
                {"range": [0, 35], "color": "#12251B"},
                {"range": [35, 65], "color": "#2D2615"},
                {"range": [65, 100], "color": "#2C1717"}
            ],
            "threshold": {
                "line": {"color": "#FFFFFF", "width": 2},
                "thickness": 0.75,
                "value": prob_disease
            }
        }
    ))

    layout_args = {**DARK_THEME_LAYOUT, "height": 190, "margin": {"l": 25, "r": 25, "t": 35, "b": 15}}
    fig.update_layout(**layout_args)

    return fig
