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
# Find duplicate groups
# -----------------------------

duplicate_mask = df.duplicated(keep=False)

duplicates = df[duplicate_mask].copy()

print("=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

print(f"Total rows: {len(df):,}")
print(f"Rows belonging to duplicate groups: {len(duplicates):,}")
print(
    f"Percentage of dataset: "
    f"{len(duplicates) / len(df) * 100:.3f}%"
)


# -----------------------------
# Number of unique duplicate
# groups
# -----------------------------

duplicate_groups = (
    duplicates
    .groupby(list(df.columns))
    .size()
    .reset_index(name="group_size")
)

print(
    f"\nUnique duplicate groups: "
    f"{len(duplicate_groups):,}"
)


# -----------------------------
# Target consistency
# -----------------------------

target_counts = (
    duplicates
    .groupby(
        [
            column for column in df.columns
            if column != "default_flag"
        ]
    )["default_flag"]
    .nunique()
)

conflicting_groups = target_counts[target_counts > 1]

print(
    f"Duplicate groups with conflicting targets: "
    f"{len(conflicting_groups):,}"
)


# -----------------------------
# Default rate
# -----------------------------

print("\nDefault rate among duplicate rows:")

print(
    duplicates["default_flag"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\nDefault rate among non-duplicate rows:")

non_duplicates = df[~duplicate_mask]

print(
    non_duplicates["default_flag"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# -----------------------------
# Duplicate group sizes
# -----------------------------

print("\nDuplicate group size distribution:")

print(
    duplicate_groups["group_size"]
    .value_counts()
    .sort_index()
)