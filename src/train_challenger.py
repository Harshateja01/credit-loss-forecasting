from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


# --------------------------------
# 1. Load data
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
# 2. Features
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
# 3. Train/test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------
# 4. Handle missing values
# --------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(X_train)

X_test = imputer.transform(X_test)


# --------------------------------
# 5. Random Forest
# --------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)


# --------------------------------
# 6. Train
# --------------------------------

model.fit(
    X_train,
    y_train,
)


# --------------------------------
# 7. Predictions
# --------------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = model.predict(
    X_test
)


# --------------------------------
# 8. Evaluate
# --------------------------------

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

pr_auc = average_precision_score(
    y_test,
    probabilities,
)


print("=" * 60)
print("RANDOM FOREST CHALLENGER")
print("=" * 60)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# --------------------------------
# 9. Feature importance
# --------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=features,
)

importance = (
    importance
    .sort_values(ascending=False)
)

print("\nTop 15 Feature Importances:")

print(
    importance.head(15)
)