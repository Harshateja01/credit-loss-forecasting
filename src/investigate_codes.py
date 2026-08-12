from pathlib import Path
import pandas as pd


project_root = Path(__file__).resolve().parent.parent

data_path = (
    project_root
    / "data"
    / "processed"
    / "credit_card_cleaned.csv"
)

df = pd.read_csv(data_path)


# Education
print("=" * 60)
print("EDUCATION")
print("=" * 60)

print(df["education"].value_counts().sort_index())


# Marriage
print("\n" + "=" * 60)
print("MARRIAGE")
print("=" * 60)

print(df["marriage"].value_counts().sort_index())


# Repayment status
pay_columns = [
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
]

print("\n" + "=" * 60)
print("REPAYMENT STATUS")
print("=" * 60)

for column in pay_columns:
    print(f"\n{column}")
    print(df[column].value_counts().sort_index())


# Default rate by repayment status
print("\n" + "=" * 60)
print("DEFAULT RATE BY MOST RECENT REPAYMENT STATUS")
print("=" * 60)

result = (
    df.groupby("pay_0")["default_flag"]
    .agg(["count", "mean"])
)

result["default_rate_pct"] = result["mean"] * 100

print(result)