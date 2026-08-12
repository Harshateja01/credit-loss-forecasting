from pathlib import Path

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
# Handle missing values
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
# SHAP analysis
# --------------------------------

explainer = shap.TreeExplainer(model)

sample = X_imputed.sample(
    n=2000,
    random_state=42
)

shap_values = explainer.shap_values(
    sample
)


# --------------------------------
# Global feature importance
# --------------------------------

if isinstance(shap_values, list):
    values = shap_values[1]
else:
    values = shap_values[:, :, 1]


importance = pd.DataFrame(
    {
        "feature": sample.columns,
        "mean_abs_shap": abs(values).mean(axis=0),
    }
).sort_values(
    "mean_abs_shap",
    ascending=False,
)


print("=" * 60)
print("GLOBAL MODEL EXPLAINABILITY")
print("=" * 60)

print(
    importance.head(15).to_string(
        index=False
    )
)