from pathlib import Path
import pandas as pd


project_root = Path(__file__).resolve().parent.parent

data_path = (
    project_root
    / "data"
    / "processed"
    / "credit_card_features.csv"
)

df = pd.read_csv(data_path)


# --------------------------------
# Default rate by delinquent months
# --------------------------------

print("=" * 60)
print("DEFAULT RATE BY DELINQUENT MONTHS")
print("=" * 60)

result = (
    df.groupby("delinquent_months")["default_flag"]
    .agg(["count", "mean"])
)

result["default_rate_pct"] = (
    result["mean"] * 100
).round(2)

print(result)


# --------------------------------
# Default rate by max delinquency
# --------------------------------

print("\n" + "=" * 60)
print("DEFAULT RATE BY MAX DELINQUENCY")
print("=" * 60)

result = (
    df.groupby("max_delinquency")["default_flag"]
    .agg(["count", "mean"])
)

result["default_rate_pct"] = (
    result["mean"] * 100
).round(2)

print(result)


# --------------------------------
# Credit utilization extremes
# --------------------------------

print("\n" + "=" * 60)
print("CREDIT UTILIZATION EXTREMES")
print("=" * 60)

print(
    df["credit_utilization"]
    .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
)


# --------------------------------
# Negative utilization
# --------------------------------

negative_utilization = (
    df["credit_utilization"] < 0
).sum()

print(
    f"\nNegative utilization rows: "
    f"{negative_utilization:,}"
)


# --------------------------------
# Very high utilization
# --------------------------------

high_utilization = (
    df["credit_utilization"] > 1
).sum()

print(
    f"Utilization > 100% rows: "
    f"{high_utilization:,}"
)


# --------------------------------
# Payment ratio
# --------------------------------

print("\n" + "=" * 60)
print("PAYMENT RATIO")
print("=" * 60)

print(
    df["payment_ratio"]
    .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
)

print(
    f"\nMissing payment ratios: "
    f"{df['payment_ratio'].isna().sum():,}"
)


# --------------------------------
# Default rate: payment ratio missing
# --------------------------------

print("\nDefault rate by payment-ratio availability:")

df["payment_ratio_missing"] = (
    df["payment_ratio"].isna().astype(int)
)

print(
    df.groupby("payment_ratio_missing")["default_flag"]
    .agg(["count", "mean"])
    .assign(
        default_rate_pct=lambda x: x["mean"] * 100
    )
)