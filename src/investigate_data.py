from pathlib import Path

import pandas as pd


# -----------------------------
# Load data
# -----------------------------

project_root = Path(__file__).resolve().parent.parent

features_path = (
    project_root
    / "data"
    / "raw"
    / "credit_card_features.csv"
)

target_path = (
    project_root
    / "data"
    / "raw"
    / "credit_card_target.csv"
)

X = pd.read_csv(features_path)
y = pd.read_csv(target_path)

column_mapping = {
    "X1": "limit_bal",
    "X2": "sex",
    "X3": "education",
    "X4": "marriage",
    "X5": "age",
    "X6": "pay_0",
    "X7": "pay_2",
    "X8": "pay_3",
    "X9": "pay_4",
    "X10": "pay_5",
    "X11": "pay_6",
    "X12": "bill_amt1",
    "X13": "bill_amt2",
    "X14": "bill_amt3",
    "X15": "bill_amt4",
    "X16": "bill_amt5",
    "X17": "bill_amt6",
    "X18": "pay_amt1",
    "X19": "pay_amt2",
    "X20": "pay_amt3",
    "X21": "pay_amt4",
    "X22": "pay_amt5",
    "X23": "pay_amt6",
}

X = X.rename(columns=column_mapping)

df = X.copy()
df["default_flag"] = y.iloc[:, 0]


# -----------------------------
# 1. Categorical distributions
# -----------------------------

print("=" * 60)
print("CATEGORICAL VALUE DISTRIBUTIONS")
print("=" * 60)

for column in ["sex", "education", "marriage"]:
    print(f"\n{column}:")
    print(df[column].value_counts().sort_index())


# -----------------------------
# 2. Repayment status values
# -----------------------------

print("\n" + "=" * 60)
print("REPAYMENT STATUS VALUES")
print("=" * 60)

pay_columns = [
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
]

for column in pay_columns:
    print(f"\n{column}:")
    print(df[column].value_counts().sort_index())


# -----------------------------
# 3. Duplicate records
# -----------------------------

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

duplicates = df[df.duplicated(keep=False)]

print(f"Duplicate rows: {len(duplicates)}")

if len(duplicates) > 0:
    print("\nDuplicate examples:")
    print(duplicates.sort_values(
        by=list(df.columns)
    ).head(20))


# -----------------------------
# 4. Impossible financial values
# -----------------------------

print("\n" + "=" * 60)
print("NEGATIVE FINANCIAL VALUES")
print("=" * 60)

financial_columns = [
    "limit_bal",
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
]

for column in financial_columns:
    negative_count = (df[column] < 0).sum()

    print(
        f"{column}: "
        f"{negative_count:,} negative values"
    )


# -----------------------------
# 5. Age range
# -----------------------------

print("\n" + "=" * 60)
print("AGE CHECK")
print("=" * 60)

print(f"Minimum age: {df['age'].min()}")
print(f"Maximum age: {df['age'].max()}")

print("\nAge distribution:")
print(df["age"].describe())


# -----------------------------
# 6. Credit limit
# -----------------------------

print("\n" + "=" * 60)
print("CREDIT LIMIT CHECK")
print("=" * 60)

print(df["limit_bal"].describe())


# -----------------------------
# 7. Target by selected categories
# -----------------------------

print("\n" + "=" * 60)
print("DEFAULT RATE BY EDUCATION")
print("=" * 60)

education_default = (
    df.groupby("education")["default_flag"]
    .agg(["count", "mean"])
)

education_default["default_rate_pct"] = (
    education_default["mean"] * 100
)

print(education_default)


print("\n" + "=" * 60)
print("DEFAULT RATE BY AGE GROUP")
print("=" * 60)

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56+",
    ],
)

age_default = (
    df.groupby("age_group", observed=False)["default_flag"]
    .agg(["count", "mean"])
)

age_default["default_rate_pct"] = (
    age_default["mean"] * 100
)

print(age_default)