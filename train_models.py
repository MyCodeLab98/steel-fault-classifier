"""
Train five classifiers on the Steel Plates Faults dataset (UCI id 198),
evaluate them on a held-out test split, and save the fitted pipelines.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 17          
TARGET = "Fault"
MODEL_DIR = Path("model")


# Load data ; split and save to csv
df = pd.read_csv("steel_plates_full.csv")
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Test class counts:\n", y_test.value_counts(), "\n")

test_df = X_test.copy()
test_df[TARGET] = y_test
test_df.to_csv("test_data.csv", index=False)


# Apply mentioned 5 models

models = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000)),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "KNN": make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": GaussianNB(),
    "SVM": make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", probability=True, random_state=SEED)),
}


# Apply 6 metric
def evaluate(model, X_eval, y_eval):
    pred = model.predict(X_eval)
    proba = model.predict_proba(X_eval)
    return {
        "Accuracy": accuracy_score(y_eval, pred),
        "AUC": roc_auc_score(y_eval, proba, multi_class="ovr",
                             average="macro", labels=model.classes_),
        "Precision": precision_score(y_eval, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_eval, pred, average="macro", zero_division=0),
        "F1": f1_score(y_eval, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_eval, pred),
    }


# Train , evaluate and test
MODEL_DIR.mkdir(exist_ok=True)
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)                      # train split only
    scores = evaluate(model, X_test, y_test)         # test split only
    rows.append({"Model": name, **scores})
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, MODEL_DIR / filename)

results = pd.DataFrame(rows).set_index("Model").round(4)
print(results)
results.to_csv(MODEL_DIR / "metrics_summary.csv")
print("\nSaved 5 models and metrics_summary.csv to", MODEL_DIR.resolve())
