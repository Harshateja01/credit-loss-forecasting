from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


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
# ORIGINAL RANDOM FOREST
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
# METRICS
# ============================================================

print("=" * 60)
print("PROBABILITY CALIBRATION")
print("=" * 60)

print("\nORIGINAL RANDOM FOREST")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, original_probability):.4f}"
)

print(
    f"PR-AUC:  "
    f"{average_precision_score(y_test, original_probability):.4f}"
)

print(
    f"Brier Score: "
    f"{brier_score_loss(y_test, original_probability):.4f}"
)


print("\nCALIBRATED RANDOM FOREST")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, calibrated_probability):.4f}"
)

print(
    f"PR-AUC:  "
    f"{average_precision_score(y_test, calibrated_probability):.4f}"
)

print(
    f"Brier Score: "
    f"{brier_score_loss(y_test, calibrated_probability):.4f}"
)
# ============================================================
# SAVE CALIBRATED MODEL
# ============================================================

models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "calibrated_random_forest.pkl"

joblib.dump(calibrated_rf, model_path)

print("\nSaved calibrated model to:")
print(model_path)