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
# LOAD MODEL
# ============================================================

model = joblib.load(model_path)

print("=" * 60)
print("SAVED MODEL TEST")
print("=" * 60)

print("\nModel loaded successfully.")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(data_path)

X = df.drop(columns=["default_flag"])


# ============================================================
# TEST PREDICTION
# ============================================================

customer_index = 15000

customer = X.iloc[[customer_index]]

probability = model.predict_proba(customer)[0, 1]

print(f"\nCustomer index: {customer_index}")
print(f"Predicted default probability: {probability:.2%}")


# ============================================================
# MODEL TYPE
# ============================================================

print("\nLoaded model type:")
print(type(model).__name__)

print("\nSaved model test completed successfully.")