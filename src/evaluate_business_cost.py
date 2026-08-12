from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


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
# 4. Imputation
# --------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# --------------------------------
# 5. Model
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
# 6. Probabilities
# --------------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]


# --------------------------------
# 7. Business cost assumptions
# --------------------------------

# Cost of incorrectly flagging
# a customer who would NOT default.

FALSE_POSITIVE_COST = 1


# Cost of missing a customer
# who eventually defaults.

FALSE_NEGATIVE_COST = 3


# --------------------------------
# 8. Evaluate thresholds
# --------------------------------

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


results = []


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

    cost = (
        false_positives * FALSE_POSITIVE_COST
        +
        false_negatives * FALSE_NEGATIVE_COST
    )

    results.append(
        {
            "threshold": threshold,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_cost": cost,
        }
    )


results_df = pd.DataFrame(results)


# --------------------------------
# 9. Display
# --------------------------------

print("=" * 60)
print("BUSINESS COST ANALYSIS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------
# 10. Best threshold
# --------------------------------

best_row = results_df.loc[
    results_df["total_cost"].idxmin()
]

print("\nLowest-cost threshold:")
print(best_row.to_string())