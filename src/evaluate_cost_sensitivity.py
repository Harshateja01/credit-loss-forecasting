from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


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
# Train/test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------
# Imputation
# --------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


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

model.fit(X_train, y_train)


# --------------------------------
# Probabilities
# --------------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]


# --------------------------------
# Thresholds
# --------------------------------

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


# --------------------------------
# Cost scenarios
# --------------------------------

false_negative_costs = [
    1,
    2,
    3,
    5,
    10,
]


results = []


for fn_cost in false_negative_costs:

    best_threshold = None
    best_cost = float("inf")

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        false_positives = (
            (predictions == 1)
            & (y_test == 0)
        ).sum()

        false_negatives = (
            (predictions == 0)
            & (y_test == 1)
        ).sum()

        total_cost = (
            false_positives
            +
            false_negatives * fn_cost
        )

        if total_cost < best_cost:

            best_cost = total_cost
            best_threshold = threshold

    results.append(
        {
            "false_negative_cost": fn_cost,
            "best_threshold": best_threshold,
            "minimum_cost": best_cost,
        }
    )


results_df = pd.DataFrame(results)


# --------------------------------
# Output
# --------------------------------

print("=" * 60)
print("COST SENSITIVITY ANALYSIS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)