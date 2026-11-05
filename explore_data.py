from ucimlrepo import fetch_ucirepo
import pandas as pd

steel = fetch_ucirepo(id=198)        # Steel Plates Faults
X = steel.data.features
Y = steel.data.targets               # 7 one-hot fault columns

print("X shape:", X.shape, "| Y shape:", Y.shape)
print("Fault columns:", Y.columns.tolist())

# Assumption check: exactly one fault per plate
print("One fault per row:", (Y.sum(axis=1) == 1).all())

y = Y.idxmax(axis=1)                 # collapse to a single label
print("\nClass counts:\n", y.value_counts())
print("\nMissing values:", X.isnull().sum().sum())

pd.set_option("display.width", 200)
print("\nFeature ranges:\n", X.describe().T[["min", "max", "mean"]])

# Save a local copy so you don't re-download every run
df = X.copy()
df["Fault"] = y
df.to_csv("steel_plates_full.csv", index=False)
print("\nSaved steel_plates_full.csv:", df.shape)

