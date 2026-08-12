from pathlib import Path
import joblib

# --------------------------------------------------
# PATHS
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent
models_dir = project_root / "models"

models_dir.mkdir(exist_ok=True)

# --------------------------------------------------
# NOTE
# --------------------------------------------------

print("=" * 60)
print("MODEL PERSISTENCE SETUP")
print("=" * 60)

print("\nModels directory:")
print(models_dir)

print("""
The trained model objects need to be saved from the
training/calibration scripts.

Next we will modify the calibration script so that
the calibrated Random Forest is saved here.
""")