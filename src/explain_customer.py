from pathlib import Path
import sys
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# --------------------------------
# Load data
# --------------------------------

project_root = Path(__file__).resolve().parent.parent

data_path = (
    project_root
    / "data"
    / "processed"
    / "credit_card_modeling.csv"
)

df = pd.read_csv(data_path)


# --------------------------------
# Features
# --------------------------------

features = [
    "limit_bal",
    "age",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
    "max_delinquency",
    "delinquent_months",
    "recent_delinquency",
    "avg_pay_status",
    "credit_utilization_capped",
    "payment_ratio_capped",
    "payment_ratio_missing",
    "high_utilization",
    "negative_bill_flag",
]


X = df[features]
y = df["default_flag"]


# --------------------------------
# Imputation
# --------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index,
)


# --------------------------------
# Train model
# --------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

model.fit(X_imputed, y)


# --------------------------------
# SHAP
# --------------------------------

explainer = shap.TreeExplainer(model)

# Pick one customer
if len(sys.argv) > 1:
    customer_index = int(sys.argv[1])
else:
    customer_index = 100

customer = X_imputed.iloc[[customer_index]]

shap_values = explainer.shap_values(
    customer
)


# --------------------------------
# Extract default-class SHAP
# --------------------------------

if isinstance(shap_values, list):

    customer_shap = shap_values[1][0]

else:

    customer_shap = shap_values[0, :, 1]


# --------------------------------
# Create explanation table
# --------------------------------

explanation = pd.DataFrame(
    {
        "feature": features,
        "value": customer.iloc[0].values,
        "shap_value": customer_shap,
    }
)

explanation["abs_shap"] = (
    explanation["shap_value"].abs()
)

explanation = explanation.sort_values(
    "abs_shap",
    ascending=False
)


# --------------------------------
# Prediction
# --------------------------------

probability = model.predict_proba(
    customer
)[0, 1]


print("=" * 60)
print("INDIVIDUAL CUSTOMER EXPLANATION")
print("=" * 60)

print(
    f"Customer index: {customer_index}"
)

print(
    f"Predicted default probability: "
    f"{probability:.2%}"
)

print("\nTop factors:")

print(
    explanation.head(10).to_string(
        index=False
    )
)