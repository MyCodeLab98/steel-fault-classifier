
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 17  # same as train_models.py

df = pd.read_csv("steel_plates_full.csv")
X = df.drop(columns=["Fault"])
y = df["Fault"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# probability=False here: tuning only needs predictions, and it runs ~5x faster.
# The final model in train_models.py keeps probability=True for AUC.
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=SEED))

param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1, 1],
    "svc__class_weight": [None, "balanced"],
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
search.fit(X_train, y_train)

print("Best settings:", search.best_params_)
print("Best CV macro F1:", round(search.best_score_, 4))

results = pd.DataFrame(search.cv_results_)
cols = ["param_svc__C", "param_svc__gamma", "param_svc__class_weight",
        "mean_test_score", "std_test_score"]
results = results[cols].rename(columns=lambda c: c.replace("param_svc__", ""))
print(results.sort_values("mean_test_score", ascending=False).head(10).to_string(index=False))

#Final tested settings that win- c=10 gamma=0.1 and weight =None