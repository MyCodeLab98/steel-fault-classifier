
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 17  # same as train_models.py

df = pd.read_csv("steel_plates_full.csv")
X = df.drop(columns=["Fault"])
y = df["Fault"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

param_grid = {
    "logisticregression__C": [0.01, 0.1, 1, 10, 100],
    "logisticregression__class_weight": [None, "balanced"],
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
search.fit(X_train, y_train)

print("Best settings:", search.best_params_)
print("Best CV macro F1:", round(search.best_score_, 4))

results = pd.DataFrame(search.cv_results_)
cols = ["param_logisticregression__C", "param_logisticregression__class_weight",
        "mean_test_score", "std_test_score"]
results = results[cols].rename(columns=lambda c: c.replace("param_logisticregression__", ""))
print(results.sort_values("mean_test_score", ascending=False).to_string(index=False))

#Result: Default c=1 gives best result