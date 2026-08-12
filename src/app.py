from pathlib import Path

import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

model_path = (
    project_root
    / "models"
    / "calibrated_random_forest.pkl"
)

data_path = (
    project_root
    / "data"
    / "processed"
    / "credit_card_modeling.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

DECISION_THRESHOLD = 0.55


# ============================================================
# LOAD MODEL / DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(model_path)


@st.cache_data
def load_data():
    return pd.read_csv(data_path)


model = load_model()
df = load_data()

X = df.drop(columns=["default_flag"])


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def load_explainer():
    rf_model = model.calibrated_classifiers_[0].estimator
    return shap.TreeExplainer(rf_model)


explainer = load_explainer()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_category(probability):
    if probability < 0.25:
        return "LOW RISK"
    elif probability < DECISION_THRESHOLD:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


def get_decision(probability):
    if probability >= DECISION_THRESHOLD:
        return "HIGH-RISK REVIEW"
    elif probability >= 0.35:
        return "ENHANCED MONITORING"
    else:
        return "STANDARD MONITORING"


def get_shap_values(customer):
    shap_values = explainer.shap_values(customer)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[0, :, 1]
        elif values.ndim == 2:
            values = values[0]

    return np.asarray(values).reshape(-1)


def build_explanation(customer):
    shap_values = get_shap_values(customer)

    explanation = pd.DataFrame(
        {
            "feature": X.columns,
            "value": customer.iloc[0].values,
            "shap_value": shap_values,
        }
    )

    explanation["abs_shap"] = explanation["shap_value"].abs()

    return explanation.sort_values(
        "abs_shap",
        ascending=False
    ).head(10)


# ============================================================
# HEADER
# ============================================================

st.title("💳 Credit Risk Assessment")

st.caption(
    "Machine-learning-based credit default risk assessment "
    "using a calibrated Random Forest model."
)

st.divider()


# ============================================================
# CUSTOMER SELECTION
# ============================================================

st.subheader("Customer Selection")

customer_index = st.number_input(
    "Enter customer index",
    min_value=0,
    max_value=len(X) - 1,
    value=15000,
    step=1,
)

customer = X.iloc[[customer_index]]

probability = model.predict_proba(customer)[0, 1]

risk_category = get_risk_category(probability)
decision = get_decision(probability)


# ============================================================
# RISK ASSESSMENT
# ============================================================

st.subheader("Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Default Probability",
        f"{probability:.2%}",
    )

with col2:
    st.metric(
        "Risk Category",
        risk_category,
    )

with col3:
    st.metric(
        "Decision Threshold",
        f"{DECISION_THRESHOLD:.0%}",
    )


if probability >= DECISION_THRESHOLD:
    st.error(
        f"Recommended decision: **{decision}**"
    )
elif probability >= 0.35:
    st.warning(
        f"Recommended decision: **{decision}**"
    )
else:
    st.success(
        f"Recommended decision: **{decision}**"
    )


# ============================================================
# RISK BAR
# ============================================================

st.markdown("### Risk Probability")

st.progress(
    min(int(probability * 100), 100)
)

st.caption(
    f"Predicted probability of default: {probability:.2%} "
    f"| Decision threshold: {DECISION_THRESHOLD:.0%}"
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.subheader("Customer Information")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Credit Limit",
        f"${customer['limit_bal'].iloc[0]:,.0f}",
    )

with col2:
    st.metric(
        "Recent Bill",
        f"${customer['bill_amt1'].iloc[0]:,.0f}",
    )

with col3:
    st.metric(
        "Recent Payment",
        f"${customer['pay_amt1'].iloc[0]:,.0f}",
    )

with col4:
    st.metric(
        "Delinquent Months",
        f"{customer['delinquent_months'].iloc[0]:.0f}",
    )

with col5:
    utilization = customer["credit_utilization_capped"].iloc[0]

    st.metric(
        "Credit Utilization",
        f"{utilization:.1%}",
    )


# ============================================================
# RISK SIGNALS
# ============================================================

st.subheader("Risk Signals")

signals = []

if customer["max_delinquency"].iloc[0] >= 2:
    signals.append(
        "⚠️ Significant historical delinquency detected"
    )

if customer["delinquent_months"].iloc[0] >= 2:
    signals.append(
        "⚠️ Multiple delinquent months"
    )

if customer["credit_utilization_capped"].iloc[0] >= 0.90:
    signals.append(
        "⚠️ High credit utilization"
    )

if customer["pay_amt1"].iloc[0] <= 0:
    signals.append(
        "⚠️ No recent payment recorded"
    )

if not signals:
    signals.append(
        "✅ No major behavioral risk signals detected"
    )

for signal in signals:
    st.write(signal)


# ============================================================
# SHAP EXPLANATION
# ============================================================

st.subheader("Model Explainability")

st.caption(
    "SHAP values show which customer characteristics "
    "most influenced the model prediction."
)

shap_df = build_explanation(customer)

positive = shap_df[
    shap_df["shap_value"] > 0
].copy()

negative = shap_df[
    shap_df["shap_value"] < 0
].copy()

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔴 Factors Increasing Risk")

    if len(positive) == 0:
        st.info("No major risk-increasing factors in the top drivers.")
    else:
        positive_display = positive[
            ["feature", "value", "shap_value"]
        ].copy()

        positive_display.columns = [
            "Feature",
            "Value",
            "SHAP Impact",
        ]

        st.dataframe(
            positive_display,
            hide_index=True,
            width="stretch",
        )

with col2:
    st.markdown("#### 🟢 Factors Reducing Risk")

    if len(negative) == 0:
        st.info("No major risk-reducing factors in the top drivers.")
    else:
        negative_display = negative[
            ["feature", "value", "shap_value"]
        ].copy()

        negative_display.columns = [
            "Feature",
            "Value",
            "SHAP Impact",
        ]

        st.dataframe(
            negative_display,
            hide_index=True,
            width="stretch",
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        "Calibrated Random Forest",
    )

with col2:
    st.metric(
        "ROC-AUC",
        "0.7766",
    )

with col3:
    st.metric(
        "PR-AUC",
        "0.5560",
    )

with col4:
    st.metric(
        "Brier Score",
        "0.1361",
    )


# ============================================================
# GOVERNANCE
# ============================================================

st.divider()

st.subheader("Model Governance")

st.info(
    """
    **Research / Portfolio Model**

    This model is intended for credit-risk analysis and
    demonstration purposes. It should not be used as a
    production credit-decision system without additional
    validation, monitoring, fairness testing, stress testing,
    and regulatory review.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Credit Risk Forecasting Project • Calibrated Random Forest"
)