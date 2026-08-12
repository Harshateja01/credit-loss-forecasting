from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

model_path = (
    project_root
    / "models"
    / "calibrated_random_forest.pkl"
)

data_path = (
    project_root
    / "data"
    / "processed"
    / "credit_card_modeling.csv"
)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load(model_path)

df = pd.read_csv(data_path)

X = df.drop(columns=["default_flag"])


# ============================================================
# CUSTOMER SELECTION
# ============================================================

customer_index = int(
    input("Enter customer index (0-29964): ")
)

customer = X.iloc[[customer_index]]


# ============================================================
# PREDICTION
# ============================================================

probability = model.predict_proba(customer)[0, 1]


# ============================================================
# RISK CATEGORY
# ============================================================

if probability >= 0.70:
    risk_category = "VERY HIGH RISK"
elif probability >= 0.55:
    risk_category = "HIGH RISK"
elif probability >= 0.30:
    risk_category = "MEDIUM RISK"
else:
    risk_category = "LOW RISK"


# ============================================================
# BUSINESS DECISION
# ============================================================

decision_threshold = 0.55

if probability >= decision_threshold:
    decision = "REVIEW"
else:
    decision = "STANDARD MONITORING"


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER CREDIT RISK ASSESSMENT")
print("=" * 60)

print(f"\nCustomer index:       {customer_index}")
print(f"Default probability:  {probability:.2%}")
print(f"Risk category:        {risk_category}")
print(f"Decision:             {decision}")
print(f"Decision threshold:   {decision_threshold:.0%}")


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

print("\n" + "-" * 60)
print("CUSTOMER INFORMATION")
print("-" * 60)

print(
    f"Credit limit:         "
    f"${customer['limit_bal'].iloc[0]:,.0f}"
)

print(
    f"Recent bill:          "
    f"${customer['bill_amt1'].iloc[0]:,.0f}"
)

print(
    f"Recent payment:       "
    f"${customer['pay_amt1'].iloc[0]:,.0f}"
)

print(
    f"Delinquent months:    "
    f"{customer['delinquent_months'].iloc[0]:.0f}"
)

print(
    f"Maximum delinquency:  "
    f"{customer['max_delinquency'].iloc[0]:.0f}"
)

print(
    f"Credit utilization:   "
    f"{customer['credit_utilization_capped'].iloc[0]:.1%}"
)


# ============================================================
# BASIC RISK SIGNALS
# ============================================================

print("\n" + "-" * 60)
print("RISK SIGNALS")
print("-" * 60)

signals = []

if customer["max_delinquency"].iloc[0] >= 2:
    signals.append("Recent history contains significant delinquency")

if customer["delinquent_months"].iloc[0] >= 2:
    signals.append("Customer had multiple delinquent months")

if customer["credit_utilization_capped"].iloc[0] >= 0.80:
    signals.append("Credit utilization is high")

if customer["pay_amt1"].iloc[0] == 0:
    signals.append("No payment recorded in the most recent month")

if customer["payment_ratio_missing"].iloc[0] == 1:
    signals.append("Payment ratio unavailable because denominator was zero")

if not signals:
    signals.append("No major rule-based risk signals detected")


for i, signal in enumerate(signals, start=1):
    print(f"{i}. {signal}")


print("\n" + "=" * 60)
print("NOTE: This is a portfolio/research model, not a")
print("production credit decision system.")
print("=" * 60)