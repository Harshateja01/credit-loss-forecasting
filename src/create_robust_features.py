from pathlib import Path
import pandas as pd


project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "credit_card_features.csv"
)

df = pd.read_csv(input_path)


# --------------------------------
# 1. Missing payment indicator
# --------------------------------

df["payment_ratio_missing"] = (
    df["payment_ratio"]
    .isna()
    .astype(int)
)


# --------------------------------
# 2. Cap extreme payment ratios
# --------------------------------

payment_cap = df["payment_ratio"].quantile(0.99)

df["payment_ratio_capped"] = (
    df["payment_ratio"]
    .clip(upper=payment_cap)
)


# --------------------------------
# 3. Cap extreme credit utilization
# --------------------------------

utilization_lower = (
    df["credit_utilization"]
    .quantile(0.01)
)

utilization_upper = (
    df["credit_utilization"]
    .quantile(0.99)
)

df["credit_utilization_capped"] = (
    df["credit_utilization"]
    .clip(
        lower=utilization_lower,
        upper=utilization_upper
    )
)


# --------------------------------
# 4. High utilization indicator
# --------------------------------

df["high_utilization"] = (
    df["credit_utilization"] > 1
).astype(int)


# --------------------------------
# 5. Negative bill indicator
# --------------------------------

bill_columns = [
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
]

df["negative_bill_flag"] = (
    df[bill_columns] < 0
).any(axis=1).astype(int)


# --------------------------------
# 6. Inspect
# --------------------------------

new_features = [
    "payment_ratio_missing",
    "payment_ratio_capped",
    "credit_utilization_capped",
    "high_utilization",
    "negative_bill_flag",
]

print("=" * 60)
print("ROBUST FEATURES")
print("=" * 60)

print(df[new_features].describe())


print("\nMissing values:")

print(
    df[new_features].isna().sum()
)


print("\nPayment ratio cap:")
print(payment_cap)

print("\nUtilization caps:")
print(f"Lower: {utilization_lower}")
print(f"Upper: {utilization_upper}")


# --------------------------------
# 7. Save
# --------------------------------

output_path = (
    project_root
    / "data"
    / "processed"
    / "credit_card_modeling.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nSaved modeling dataset to:")
print(output_path)