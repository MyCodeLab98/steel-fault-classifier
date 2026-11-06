
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.naive_bayes import GaussianNB

SEED = 17  # same as train_models.py

df = pd.read_csv("steel_plates_full.csv")
X = df.drop(columns=["Fault"])
y = df["Fault"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# "prep" is either left out ("passthrough") or a PowerTransformer
pipe = Pipeline([("prep", "passthrough"), ("nb", GaussianNB())])

param_grid = {
    "prep": ["passthrough", PowerTransformer()],
    "nb__var_smoothing": np.logspace(-11, -5, 7),
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
search.fit(X_train, y_train)

print("Best settings:", search.best_params_)
print("Best CV macro F1:", round(search.best_score_, 4))

results = pd.DataFrame(search.cv_results_)
results["prep"] = results["param_prep"].astype(str).str[:16]
cols = ["prep", "param_nb__var_smoothing", "mean_test_score", "std_test_score"]
print(results[cols].sort_values("mean_test_score", ascending=False).head(10).to_string(index=False))

#Result : Optimal with Powertransformer+GaussianNB