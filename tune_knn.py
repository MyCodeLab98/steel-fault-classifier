
"""
Tune KNN with 5-fold cross-validation on the training data only.
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

SEED = 17  # same as train_models.py

df = pd.read_csv("steel_plates_full.csv")
X = df.drop(columns=["Fault"])
y = df["Fault"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())

# Inside a pipeline, settings are named <step>__<parameter> (two underscores)
param_grid = {
    "kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
    "kneighborsclassifier__weights": ["uniform", "distance"],
    "kneighborsclassifier__p": [1, 2],
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro")
search.fit(X_train, y_train)

print("Best settings:", search.best_params_)
print("Best CV macro F1:", round(search.best_score_, 4))

results = pd.DataFrame(search.cv_results_)
cols = ["param_kneighborsclassifier__n_neighbors", "param_kneighborsclassifier__weights",
        "param_kneighborsclassifier__p", "mean_test_score", "std_test_score"]
results = results[cols].rename(columns=lambda c: c.replace("param_kneighborsclassifier__", ""))
print(results.sort_values("mean_test_score", ascending=False).head(10).to_string(index=False))

#Final approved settings - neighbors 7 ; weights = distance and p=1