from pathlib import Path
import pandas as pd


# -----------------------------
# Load cleaned data
# -----------------------------

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "credit_card_cleaned.csv"
)

df = pd.read_csv(input_path)


# -----------------------------
# Repayment columns
# -----------------------------

pay_columns = [
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
]


# -----------------------------
# Create repayment features
# -----------------------------

df["max_delinquency"] = df[pay_columns].max(axis=1)

df["delinquent_months"] = (
    (df[pay_columns] > 0)
    .sum(axis=1)
)

df["recent_delinquency"] = (
    df["pay_0"] > 0
).astype(int)

df["avg_pay_status"] = (
    df[pay_columns].mean(axis=1)
)


# -----------------------------
# Credit utilization
# -----------------------------

df["credit_utilization"] = (
    df["bill_amt1"] / df["limit_bal"]
)


# -----------------------------
# Payment ratio
# -----------------------------

df["payment_ratio"] = (
    df["pay_amt1"] /
    df["bill_amt1"].abs().replace(0, pd.NA)
)


# -----------------------------
# Inspect new features
# -----------------------------

new_features = [
    "max_delinquency",
    "delinquent_months",
    "recent_delinquency",
    "avg_pay_status",
    "credit_utilization",
    "payment_ratio",
]

print("=" * 60)
print("NEW FEATURES")
print("=" * 60)

print(df[new_features].describe())


# -----------------------------
# Missing / infinite values
# -----------------------------

print("\nMissing values:")

print(
    df[new_features]
    .isna()
    .sum()
)

print("\nInfinite values:")

print(
    df[new_features]
    .isin([float("inf"), float("-inf")])
    .sum()
)


# -----------------------------
# Save feature dataset
# -----------------------------

output_path = (
    project_root
    / "data"
    / "processed"
    / "credit_card_features.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nSaved feature dataset to:")
print(output_path)