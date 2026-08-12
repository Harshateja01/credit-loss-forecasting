from pathlib import Path

from ucimlrepo import fetch_ucirepo


# Fetch UCI Default of Credit Card Clients dataset
dataset = fetch_ucirepo(id=350)

X = dataset.data.features
y = dataset.data.targets


# Create output directory
output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)


# Save raw data
X.to_csv(output_dir / "credit_card_features.csv", index=False)
y.to_csv(output_dir / "credit_card_target.csv", index=False)


print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Saved to: {output_dir}")