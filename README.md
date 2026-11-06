# Steel Plate Fault Classifier

Machine Learning – Assignment 2 

- **GitHub repository:** [https://github.com/MyCodeLab98/steel-fault-classifier](https://github.com/MyCodeLab98/steel-fault-classifier)
- **Live Streamlit app:** [https://steel-fault-classifierapp.streamlit.app/](https://steel-fault-classifierapp.streamlit.app/)

---

## a. Problem statement

Steel plates coming off a production line are inspected for surface defects. For every defect found, an inspection system records 27 measurements describing where the defect is on the plate, how large it is, how bright or dark it appears, and its geometric shape.

The goal is to **predict which of 7 fault types a defect belongs to** from these 27 measurements. This is a **multi-class classification** problem. Five classification models were trained on the same data, tuned with cross-validation, evaluated on a held-out test set with six metrics, and deployed in an interactive Streamlit app that loads the saved models and evaluates them on uploaded test data.

---



## b. Dataset description


| Property       | Value                                                                                                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Name           | Steel Plates Faults                                                                                                                                                          |
| Source         | UCI Machine Learning Repository (dataset id 198): [https://archive.ics.uci.edu/dataset/198/steel+plates+faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) |
| Instances      | 1,941 steel plates                                                                                                                                                           |
| Features       | 27 numeric features                                                                                                                                                          |
| Target         | `Fault`, 7 classes                                                                                                                                                           |
| Missing values | None                                                                                                                                                                         |


**Features.** The 27 measurements fall into five groups:

- **Position:** `X_Minimum`, `X_Maximum`, `Y_Minimum`, `Y_Maximum`
- **Size and perimeter:** `Pixels_Areas`, `X_Perimeter`, `Y_Perimeter`, `LogOfAreas`, `Log_X_Index`, `Log_Y_Index`, `SigmoidOfAreas`
- **Brightness:** `Sum_of_Luminosity`, `Minimum_of_Luminosity`, `Maximum_of_Luminosity`, `Luminosity_Index`
- **Plate properties:** `Length_of_Conveyer`, `TypeOfSteel_A300`, `TypeOfSteel_A400`, `Steel_Plate_Thickness`
- **Shape indexes:** `Edges_Index`, `Empty_Index`, `Square_Index`, `Outside_X_Index`, `Edges_X_Index`, `Edges_Y_Index`, `Outside_Global_Index`, `Orientation_Index`

**Target preparation.** In the original data the fault type is stored as 7 one-hot columns. It was verified that every plate has exactly one fault marked, and the 7 columns were collapsed into a single `Fault` label.

**Class distribution (imbalanced):**


| Fault        | Plates | Share |
| ------------ | ------ | ----- |
| Other_Faults | 673    | 34.7% |
| Bumps        | 402    | 20.7% |
| K_Scratch    | 391    | 20.1% |
| Z_Scratch    | 190    | 9.8%  |
| Pastry       | 158    | 8.1%  |
| Stains       | 72     | 3.7%  |
| Dirtiness    | 55     | 2.8%  |


**Notable characteristics found during exploration:**

- **Very different feature scales.** `Y_Minimum` and `Y_Maximum` reach about 13 million and `Sum_of_Luminosity` about 11.6 million, while most index features lie between 0 and 1.
- **Skewed features.** `Pixels_Areas` has a mean of about 1,894 but a maximum of 152,655.
- **Redundant features.** `TypeOfSteel_A300` and `TypeOfSteel_A400` are exact complements (they sum to 1 for every plate), and `Pixels_Areas` and `X_Perimeter` have a correlation of 0.967.
- **Class imbalance.** The largest class is about 12 times the size of the smallest, and `Other_Faults` is a catch-all class covering several defect types.

**Train/test split.** 80/20 stratified split (1,552 training plates, 389 test plates), so every fault type keeps the same proportion in both sets. The test set is saved as `test_data.csv` and is the file used in the Streamlit app.There is option also included in the app to click and download sample data for testing.

---



## c. GitHub repository

[https://github.com/MyCodeLab98/steel-fault-classifier](https://github.com/MyCodeLab98/steel-fault-classifier)

```
steel-fault-classifier/
├── app.py                  # Streamlit app 
├── train_models.py         # split, train 5 tuned models, compute metrics, save pipelines
├── explore_data.py         # load dataset, collapse labels, explore
├── checks.py               # evidence checks behind the observations
├── tune_tree.py            # Decision Tree
├── tune_knn.py             #  KNN
├── tune_svm.py             #  SVM
├── tune_lr.py              #  Logistic Regression
├── tune_nb.py              # Naive Bayes
├── steel_plates_full.csv   # full prepared dataset
├── test_data.csv           # held-out test set (27 features + Fault)
├── requirements.txt        # pinned dependencies
├── .streamlit/config.toml  # app theme
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metrics_summary.csv
└── README.md
```

---



## d. Models used

All five models were trained on the same training split and evaluated on the same test split. Each model's settings were chosen by 5-fold cross-validation on the training data (see *Model optimisation* below).


| Model                  | Final configuration                                                               |
| ---------------------- | --------------------------------------------------------------------------------- |
| Logistic Regression    | `StandardScaler` + `LogisticRegression(C=1, max_iter=2000)`                       |
| Decision Tree          | `DecisionTreeClassifier(criterion="entropy", max_depth=12, min_samples_leaf=3)`   |
| K-Nearest Neighbors    | `StandardScaler` + `KNeighborsClassifier(n_neighbors=7, weights="distance", p=1)` |
| Naive Bayes            | `PowerTransformer` + `GaussianNB`                                                 |
| Support Vector Machine | `StandardScaler` + `SVC(kernel="rbf", C=10, gamma=0.1, probability=True)`         |


Each model is saved as a complete pipeline, so the app applies exactly the preprocessing learned from the training data.

**Metrics.** Because this is a 7-class problem with imbalanced classes:

- **AUC** is computed one-vs-rest from predicted probabilities and macro-averaged.
- **Precision, Recall and F1** are macro-averaged, so every fault type counts equally regardless of its size.
- **MCC** (Matthews Correlation Coefficient) is computed on all 7 classes; 0 means no better than chance and 1 means perfect.



### Comparison table (test set, 389 plates)


| ML Model Name          | Accuracy   | AUC        | Precision  | Recall     | F1         | MCC        |
| ---------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression    | 0.7147     | 0.9324     | 0.7489     | 0.7262     | 0.7361     | 0.6318     |
| Decision Tree          | 0.7121     | 0.8775     | 0.7282     | 0.7428     | 0.7320     | 0.6343     |
| KNN                    | **0.7352** | 0.9349     | 0.7576     | **0.7653** | 0.7568     | 0.6584     |
| Naive Bayes            | 0.5964     | 0.9041     | 0.6135     | 0.6811     | 0.6172     | 0.5233     |
| Support Vector Machine | **0.7352** | **0.9443** | **0.7895** | 0.7519     | **0.7682** | **0.6589** |


### Model optimisation

Hyperparameters were tuned with `GridSearchCV` using **5-fold cross-validation on the training data only**, scored by macro F1. The test set was not used for any choice, so the reported test scores are an honest estimate of performance on unseen plates. Where several settings scored within noise of each other, the simpler and more stable one (lower standard deviation across folds) was chosen.


| Model                  | What was tuned                                               | Finding                                                                                           | Test macro F1 (default → tuned) |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | ------------------------------- |
| Logistic Regression    | `C`, `class_weight`                                          | Default `C=1` was already best; `C` from 0.1 to 100 made little difference; balanced weights hurt | 0.7361 → 0.7361                 |
| Decision Tree          | `max_depth`, `min_samples_leaf`, `criterion`, `class_weight` | Entropy splits beat gini (CV 0.734 → 0.756); depth limits alone did not help                      | 0.6922 → 0.7320                 |
| KNN                    | `n_neighbors`, `weights`, `p`                                | Manhattan distance and distance weighting clearly better; k=7 chosen for stability                | 0.7279 → 0.7568                 |
| Naive Bayes            | `PowerTransformer` on/off, `var_smoothing`                   | Reshaping skewed features raised CV macro F1 from about 0.40 to 0.63                              | 0.4155 → 0.6172                 |
| Support Vector Machine | `C`, `gamma`, `class_weight`                                 | `C=10`, `gamma=0.1` best in CV (0.780)                                                            | 0.7707 → 0.7682                 |




### Observations on model performance


| ML Model Name                        | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression                  | Strong probability ranking (AUC 0.93) but mid-table on accuracy (0.71). Tuning showed it is insensitive to regularisation: any `C` from 0.1 to 100 gives the same CV score, and only extreme regularisation (`C=0.01`) breaks it. Balanced class weights reduced CV macro F1 from 0.714 to 0.673. Its ceiling comes from its linear class boundaries, which cannot separate the irregular, overlapping fault groups that curved or local models handle.                                                                                                                                |
| Decision Tree                        | Tuning improved every metric. The untuned tree memorised the training data (train accuracy 1.00 vs test 0.68); the tuned tree (entropy, `max_depth=12`, `min_samples_leaf=3`) reaches 0.93 train vs 0.71 test, so the overfitting gap shrank from 0.32 to 0.21. AUC rose from 0.81 to 0.88 because leaves with at least 3 plates give graded probabilities instead of only 0 or 1. It still has the lowest AUC, reflecting the instability of a single tree.                                                                                                                           |
| KNN                                  | Improved on all six metrics after tuning and now has the best recall (0.765) and joint-best accuracy (0.735). Manhattan distance (`p=1`) clearly beat Euclidean in CV: with skewed features, squaring large differences lets a few extreme values dominate the distance. Distance weighting lets the closest plates count most. Scaling is essential, because unscaled distances are dominated by position features in the millions.                                                                                                                                                   |
| Naive Bayes                          | Still the weakest model, but improved most: macro F1 rose from 0.42 to 0.62 and AUC from 0.84 to 0.90 once a `PowerTransformer` made the skewed features closer to bell-shaped, repairing one broken assumption. The independence assumption remains violated (the two steel-type columns are exact complements, and area and perimeter correlate at 0.967), so evidence is still double-counted. It now over-predicts small classes: Pastry recall 0.72 with precision 0.35, Dirtiness recall 0.73 with precision 0.36, while the mixed `Other_Faults` class has recall of only 0.33. |
| Support Vector Machine               | Best on AUC (0.944), precision (0.790), F1 (0.768) and MCC (0.659). The RBF kernel's curved boundaries fit the overlapping fault groups. CV chose `C=10`, `gamma=0.1`; on the test set this scored 6 plates lower in accuracy than the default settings, while recall rose slightly. The CV-chosen settings were kept, since choosing by test score would bias the evaluation. Rare classes are handled very well (Dirtiness F1 0.95, Stains 0.89, K_Scratch 0.97); its weak spot is Pastry (recall 0.41, F1 0.45).                                                                    |
| **Overall, winner for this dataset** | **Support Vector Machine**, leading on four of six metrics including MCC and macro F1, the fairest measures under class imbalance. Tuned KNN is a very close second, with the same accuracy, the best recall, and an MCC only 0.0005 lower. A model that always predicted `Other_Faults` would score about 34.7% accuracy but an MCC of 0, which is why MCC and macro F1 were weighted above accuracy. Across models, K_Scratch is the easiest fault type and Pastry the hardest.                                                                                                      |


---



## Streamlit app

The app loads the five saved models from `model/` and evaluates them on an uploaded CSV. It never retrains.

- **Upload test data (CSV)** with the 27 features and the `Fault` column. A sample `test_data.csv` can be downloaded from the sidebar.
- **Model selection dropdown** for the five models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- **Confusion matrix** and a **per-fault classification report**.
- **Compare all models** on the uploaded data, with the best score per metric highlighted.
- **Predictions** table with correct/incorrect flags, downloadable as CSV.



## How to run locally

```bash
pip install -r requirements.txt
python train_models.py          # optional
python -m streamlit run app.py
```



## Environment

Developed and executed on BITS Virtual Lab: Python 3.12, scikit-learn 1.5.2, NumPy 1.26.4, pandas 2.2.3, pyarrow 17.0.0, Streamlit 1.66.1.