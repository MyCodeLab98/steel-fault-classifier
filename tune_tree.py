"""
Tune the Decision Tree with 5-fold cross-validation on the training data only.
The test set is not touched here.
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

SEED = 17  # same as train_models.py

df = pd.read_csv("steel_plates_full.csv")
X = df.drop(columns=["Fault"])
y = df["Fault"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 1. Settings to try: 6 depths x 4 leaf sizes = 24 combinations
param_grid = {
    "max_depth": [8, 12, 20, None],
    "min_samples_leaf": [1, 3],
    "criterion":["gini","entropy"],
    "class_weight":[None,"balanced"],
}

# 2. The search: each combination is scored with 5-fold CV using macro F1
search = GridSearchCV(
    DecisionTreeClassifier(random_state=SEED),
    param_grid,
    cv=5,
    scoring="f1_macro",
)

# 3. Fit on the training data ONLY; the test set stays unseen
search.fit(X_train, y_train)

# 4. Best settings and their cross-validated (practice) score
print("Best settings:", search.best_params_)
print("Best CV macro F1:", round(search.best_score_, 4))

# How every combination scored, best first
results = pd.DataFrame(search.cv_results_)
cols = ["param_criterion", "param_class_weight", "param_max_depth",
        "param_min_samples_leaf", "mean_test_score", "std_test_score"]
print(results[cols].sort_values("mean_test_score", ascending=False).head(10).to_string(index=False))
