from pathlib import Path

import joblib
import shap
import numpy as np
import pandas as pd


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
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load(model_path)

df = pd.read_csv(data_path)

X = df.drop(columns=["default_flag"])


# ============================================================
# CUSTOMER
# ============================================================

customer_index = int(
    input("Enter customer index (0-29964): ")
)

customer = X.iloc[[customer_index]]


# ============================================================
# PREDICTION
# ============================================================

probability = model.predict_proba(customer)[0, 1]


# ============================================================
# SHAP EXPLANATION
# ============================================================

# CalibratedClassifierCV contains calibrated Random Forest
# estimators. We explain the underlying Random Forest.

rf_model = model.calibrated_classifiers_[0].estimator

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(customer)


# ============================================================
# HANDLE SHAP OUTPUT
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP versions
    customer_shap = np.asarray(shap_values[1][0])

else:

    shap_values = np.asarray(shap_values)

    # Newer SHAP versions may return:
    # (rows, features, classes)

    if shap_values.ndim == 3:

        customer_shap = shap_values[0, :, 1]

    elif shap_values.ndim == 2:

        customer_shap = shap_values[0]

    else:

        customer_shap = shap_values


# Make sure SHAP values are one-dimensional
customer_shap = np.asarray(customer_shap).reshape(-1)


# ============================================================
# CREATE EXPLANATION TABLE
# ============================================================

shap_df = pd.DataFrame({
    "feature": X.columns,
    "value": customer.iloc[0].values,
    "shap_value": customer_shap
})

shap_df["abs_shap"] = shap_df["shap_value"].abs()

shap_df = (
    shap_df
    .sort_values("abs_shap", ascending=False)
    .head(10)
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER MODEL EXPLANATION")
print("=" * 60)

print(f"\nCustomer index: {customer_index}")

print(
    f"Default probability: {probability:.2%}"
)

print("\nTop model drivers:")
print("-" * 60)

print(
    shap_df[
        ["feature", "value", "shap_value"]
    ].to_string(index=False)
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\nInterpretation:")
print("-" * 60)

for _, row in shap_df.head(5).iterrows():

    if row["shap_value"] > 0:
        direction = "increases"
    else:
        direction = "decreases"

    print(
        f"{row['feature']} = {row['value']:.2f} "
        f"{direction} predicted default risk"
    )


print("\n" + "=" * 60)