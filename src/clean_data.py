from pathlib import Path
import pandas as pd


# -----------------------------
# 1. Locate project
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


# -----------------------------
# 2. Load raw data
# -----------------------------

X = pd.read_csv(features_path)
y = pd.read_csv(target_path)


# -----------------------------
# 3. Rename columns
# -----------------------------

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


# -----------------------------
# 4. Combine features + target
# -----------------------------

df = X.copy()
df["default_flag"] = y.iloc[:, 0]


# -----------------------------
# 5. Remove exact duplicates
# -----------------------------

original_rows = len(df)

df = df.drop_duplicates().reset_index(drop=True)

removed_rows = original_rows - len(df)


print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

print(f"Original rows: {original_rows:,}")
print(f"Removed duplicate rows: {removed_rows:,}")
print(f"Remaining rows: {len(df):,}")


# -----------------------------
# 6. Save processed dataset
# -----------------------------

processed_dir = (
    project_root
    / "data"
    / "processed"
)

processed_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    processed_dir
    / "credit_card_cleaned.csv"
)

df.to_csv(
    output_path,
    index=False
)


print("\nSaved cleaned dataset to:")
print(output_path)