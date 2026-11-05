"""
Evidence checks behind the README observations:
overfitting, feature redundancy, per-class results, effect of scaling on KNN.
Uses the same split as train_models.py (same SEED).
"""
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

SEED = 17  # must match SEED in train_models.py
df = pd.read_csv("steel_plates_full.csv")
X, y = df.drop(columns=["Fault"]), df["Fault"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

dt = joblib.load("model/decision_tree.joblib")
print("DT train accuracy:", dt.score(X_tr, y_tr))
print("Steel types complementary:",
      (X["TypeOfSteel_A300"] + X["TypeOfSteel_A400"] == 1).all())
print("Corr Pixels_Areas vs X_Perimeter:",
      round(X["Pixels_Areas"].corr(X["X_Perimeter"]), 3))

for name in ["naive_bayes", "svm"]:
    m = joblib.load(f"model/{name}.joblib")
    print(f"\n{name}\n", classification_report(y_te, m.predict(X_te), zero_division=0))

knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
print("KNN without scaling, test accuracy:", round(knn_raw.score(X_te, y_te), 4))
