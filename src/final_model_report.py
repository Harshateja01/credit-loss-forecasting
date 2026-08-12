from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

# --------------------------------------------------
# RESULTS FROM OUR MODELING
# --------------------------------------------------

results = pd.DataFrame({
    "model": [
        "Logistic Regression",
        "Random Forest",
        "Calibrated Random Forest"
    ],
    "roc_auc": [
        0.7474,
        0.7770,
        0.7766
    ],
    "pr_auc": [
        0.5157,
        0.5558,
        0.5560
    ],
    "brier_score": [
        None,
        0.1766,
        0.1361
    ]
})

# --------------------------------------------------
# BUSINESS THRESHOLD
# --------------------------------------------------

threshold = 0.55
false_positives = 713
false_negatives = 584
total_cost = 2465

# --------------------------------------------------
# PRINT REPORT
# --------------------------------------------------

print("=" * 65)
print("FINAL CREDIT RISK MODEL REPORT")
print("=" * 65)

print("\nMODEL COMPARISON")
print("-" * 65)

print(
    results.to_string(
        index=False,
        formatters={
            "roc_auc": "{:.4f}".format,
            "pr_auc": "{:.4f}".format,
            "brier_score": lambda x: (
                "N/A" if pd.isna(x) else f"{x:.4f}"
            )
        }
    )
)

print("\n" + "=" * 65)
print("SELECTED BUSINESS THRESHOLD")
print("=" * 65)

print(f"Threshold:       {threshold:.2f}")
print(f"False positives: {false_positives:,}")
print(f"False negatives: {false_negatives:,}")
print(f"Total cost:      {total_cost:,}")

print("\n" + "=" * 65)
print("KEY MODEL DRIVERS")
print("=" * 65)

features = [
    "max_delinquency",
    "delinquent_months",
    "pay_0",
    "recent_delinquency",
    "avg_pay_status",
    "credit_utilization_capped",
    "limit_bal"
]

for i, feature in enumerate(features, start=1):
    print(f"{i}. {feature}")

print("\n" + "=" * 65)
print("BUSINESS INTERPRETATION")
print("=" * 65)

print("""
The Random Forest provides better discrimination than the
logistic regression baseline.

Probability calibration substantially improves the reliability
of predicted default probabilities.

The selected threshold is based on modeled business costs rather
than simply using the default 0.50 classification threshold.

The strongest predictors are primarily related to repayment
behavior, delinquency history, and credit utilization.

This model should be treated as a research/portfolio model and
not as a production credit-decision system.
""")

print("=" * 65)