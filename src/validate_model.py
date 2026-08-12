from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


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
# Pipeline
# --------------------------------

pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ]
)


# --------------------------------
# Cross-validation
# --------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


scores = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=[
        "roc_auc",
        "average_precision",
        "f1",
        "recall",
        "precision",
    ],
    n_jobs=-1,
)


# --------------------------------
# Results
# --------------------------------

print("=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)


metrics = {
    "ROC-AUC": "test_roc_auc",
    "PR-AUC": "test_average_precision",
    "F1": "test_f1",
    "Recall": "test_recall",
    "Precision": "test_precision",
}


for name, column in metrics.items():

    values = scores[column]

    print(
        f"{name}: "
        f"{values.mean():.4f} "
        f"+/- "
        f"{values.std():.4f}"
    )


print("\nIndividual fold results:")

results = pd.DataFrame(
    {
        "fold": range(1, 6),
        "roc_auc": scores["test_roc_auc"],
        "pr_auc": scores["test_average_precision"],
        "f1": scores["test_f1"],
        "recall": scores["test_recall"],
        "precision": scores["test_precision"],
    }
)

print(
    results.to_string(
        index=False
    )
)