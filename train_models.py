"""
Train five classifiers on the Steel Plates Faults dataset 
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
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 17          
TARGET = "Fault"
MODEL_DIR = Path("model")


#Load data , split and save csv
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


# 5 given models to train dataset ; tuned and selected best possible parameters - check tune* python files
models = {
    # tune_lr.py: default C=1 was best; balanced class weights hurt
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(C=1, max_iter=2000)),
    # tune_tree.py: entropy splits beat gini; depth 12 + leaf 3 is simpler and more stable
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=12,
                                            min_samples_leaf=3, random_state=SEED),
    # tune_knn.py: Manhattan distance (p=1) + distance weighting; k=7 for stability
    "KNN": make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=7, weights="distance", p=1)),
    # tune_nb.py: PowerTransformer makes skewed features bell-shaped for GaussianNB
    "Naive Bayes": make_pipeline(PowerTransformer(), GaussianNB()),
    # tune_svm.py: C=10, gamma=0.1 best and stable; probability=True needed for AUC
    "SVM": make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=10, gamma=0.1, probability=True,
                             random_state=SEED)),
}


# Evaluate with mentioned 6 metrics 
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


# Train evaluate and save
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
