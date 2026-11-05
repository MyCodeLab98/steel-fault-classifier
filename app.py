"""
Steel plate fault classifier: evaluate five saved models on an uploaded test CSV.
The models are loaded from model/ and are never retrained here.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "Fault"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

st.set_page_config(page_title="Steel plate fault classifier", page_icon="🔩",
                   layout="wide")


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


def score(model, X, y):
    pred = model.predict(X)
    try:
        auc = roc_auc_score(y, model.predict_proba(X), multi_class="ovr",
                            average="macro", labels=model.classes_)
    except ValueError:
        auc = np.nan  # AUC needs every fault type present in the uploaded file
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred


# ---------- Sidebar: inputs ----------
with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    model_name = st.selectbox("Model", list(MODEL_FILES))
    st.caption(f"The CSV needs the 27 plate measurements and a `{TARGET}` column.")

    sample = BASE_DIR / "test_data.csv"
    if sample.exists():
        st.download_button("Download sample test file", sample.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

# ---------- Header ----------
st.title("Steel plate fault classifier")
st.write("Each plate is described by 27 measurements of a surface defect: position, "
         "size, brightness and shape. The models predict which of 7 fault types it is "
         "(UCI Steel Plates Faults dataset).")

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to evaluate the models. "
            "No file yet? Download the sample test file there.")
    st.stop()

# ---------- Validate upload ----------
data = pd.read_csv(uploaded)
model = load_model(MODEL_FILES[model_name])
features = list(model.feature_names_in_)

missing = [c for c in features + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {', '.join(missing)}")
    st.stop()

X, y = data[features], data[TARGET]
st.caption(f"{len(data)} plates loaded · {y.nunique()} fault types present")

# ---------- Metrics for the selected model ----------
metrics, pred = score(model, X, y)
st.subheader(f"{model_name} on your test data")
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, "n/a" if np.isnan(value) else f"{value:.4f}")
if np.isnan(metrics["AUC"]):
    st.caption("AUC is n/a because not every fault type appears in the uploaded file.")

tab_cm, tab_report, tab_compare, tab_pred = st.tabs(
    ["Confusion matrix", "Per-fault report", "Compare all models", "Predictions"])

with tab_cm:
    labels = list(model.classes_)
    cm = confusion_matrix(y, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted fault")
    ax.set_ylabel("Actual fault")
    plt.xticks(rotation=35, ha="right")
    st.pyplot(fig)
    st.caption("Diagonal cells are correct predictions; everything else is a mix-up "
               "between two fault types.")

with tab_report:
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

with tab_compare:
    rows = []
    for name, filename in MODEL_FILES.items():
        m, _ = score(load_model(filename), X, y)
        rows.append({"Model": name, **m})
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe0ef"),
                 use_container_width=True)
    st.caption("Highlighted cells are the best score for each metric.")

with tab_pred:
    out = data.copy()
    out["Predicted"] = pred
    out["Correct"] = out["Predicted"] == out[TARGET]
    st.dataframe(out[[TARGET, "Predicted", "Correct"]], use_container_width=True)
    st.download_button("Download predictions", out.to_csv(index=False),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")
