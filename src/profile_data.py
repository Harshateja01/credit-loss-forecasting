from pathlib import Path

import pandas as pd


# Locate raw data
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "credit_card_features.csv"
target_path = project_root / "data" / "raw" / "credit_card_target.csv"


# Load data
X = pd.read_csv(data_path)
y = pd.read_csv(target_path)


# Combine features and target
df = X.copy()
df["default_flag"] = y.iloc[:, 0]


print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")

print("\nColumn names:")
for column in df.columns:
    print(f"- {column}")


print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

print(missing[missing > 0])

if missing.sum() == 0:
    print("No missing values found.")


print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print(f"Duplicate rows: {df.duplicated().sum():,}")


print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df["default_flag"].value_counts())
print("\nPercentages:")
print(df["default_flag"].value_counts(normalize=True).mul(100).round(2))


print("\n" + "=" * 60)
print("NUMERIC SUMMARY")
print("=" * 60)

print(df.describe().T)


print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for column in df.columns:
    print(f"{column}: {df[column].nunique():,}")