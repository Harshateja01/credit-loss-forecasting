from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


# ============================================================
# LOAD DATA
# ============================================================

project_root = Path(__file__).resolve().parent.parent

data_path = (
    project_root
    / "data"
    / "processed"
    / "credit_card_modeling.csv"
)

df = pd.read_csv(data_path)

X = df.drop(columns=["default_flag"])
y = df["default_flag"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

original_probability = rf.predict_proba(X_test)[:, 1]


# ============================================================
# CALIBRATED RANDOM FOREST
# ============================================================

calibrated_rf = CalibratedClassifierCV(
    rf,
    method="sigmoid",
    cv=5
)

calibrated_rf.fit(X_train, y_train)

calibrated_probability = calibrated_rf.predict_proba(X_test)[:, 1]


# ============================================================
# CALIBRATION CURVES
# ============================================================

original_fraction, original_mean = calibration_curve(
    y_test,
    original_probability,
    n_bins=10,
    strategy="quantile"
)

calibrated_fraction, calibrated_mean = calibration_curve(
    y_test,
    calibrated_probability,
    n_bins=10,
    strategy="quantile"
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 60)
print("CALIBRATION CURVE ANALYSIS")
print("=" * 60)

print("\nOriginal Random Forest:")

for predicted, actual in zip(
    original_mean,
    original_fraction
):
    print(
        f"Predicted: {predicted:.3f} | "
        f"Actual: {actual:.3f}"
    )


print("\nCalibrated Random Forest:")

for predicted, actual in zip(
    calibrated_mean,
    calibrated_fraction
):
    print(
        f"Predicted: {predicted:.3f} | "
        f"Actual: {actual:.3f}"
    )


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    original_mean,
    original_fraction,
    marker="o",
    label="Original Random Forest"
)

plt.plot(
    calibrated_mean,
    calibrated_fraction,
    marker="o",
    label="Calibrated Random Forest"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Default Rate")

plt.title("Random Forest Probability Calibration")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()