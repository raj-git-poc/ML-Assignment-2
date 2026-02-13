# ==========================================
# ML Assignment 2 - Ordered as per Rubric
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ==========================================
# Page Setup
# ==========================================

st.set_page_config(page_title="ML Classification App", layout="wide")
st.title("Machine Learning Classification Models")

# ==========================================
# Load Dataset
# ==========================================

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Define Models
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# ==========================================
# a) DATASET UPLOAD OPTION (FIRST)
# ==========================================

st.header("a) Dataset Upload Option")

uploaded_file = st.file_uploader(
    "Upload Test CSV File (Must contain 'target' column)",
    type=["csv"]
)

# ==========================================
# b) MODEL SELECTION DROPDOWN (SECOND)
# ==========================================

st.header("b) Model Selection")

model_choice = st.selectbox("Select a Model", list(models.keys()))
selected_model = models[model_choice]

# ==========================================
# Train model for selected choice
# ==========================================

if model_choice in ["Logistic Regression", "KNN"]:
    selected_model.fit(X_train_scaled, y_train)
    y_pred = selected_model.predict(X_test_scaled)
    y_prob = selected_model.predict_proba(X_test_scaled)[:, 1]
else:
    selected_model.fit(X_train, y_train)
    y_pred = selected_model.predict(X_test)
    y_prob = selected_model.predict_proba(X_test)[:, 1]

# ==========================================
# c) DISPLAY OF EVALUATION METRICS (THIRD)
# ==========================================

st.header("c) Evaluation Metrics (Test Dataset)")

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", round(accuracy, 4))
col2.metric("AUC", round(auc, 4))
col3.metric("Precision", round(precision, 4))

col1.metric("Recall", round(recall, 4))
col2.metric("F1 Score", round(f1, 4))
col3.metric("MCC Score", round(mcc, 4))

# ==========================================
# d) CONFUSION MATRIX & CLASSIFICATION REPORT (FOURTH)
# ==========================================

st.header("d) Confusion Matrix and Classification Report")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

st.subheader("Confusion Matrix")
st.dataframe(cm_df)

st.subheader("Classification Report")

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df.round(4))

# ==========================================
# If CSV Uploaded → Evaluate on Uploaded Data
# ==========================================

if uploaded_file is not None:

    st.markdown("---")
    st.subheader("Evaluation on Uploaded Dataset")

    test_data = pd.read_csv(uploaded_file)

    if "target" not in test_data.columns:
        st.error("Uploaded CSV must contain 'target' column.")
    else:
        X_upload = test_data.drop("target", axis=1)
        y_upload = test_data["target"]

        if model_choice in ["Logistic Regression", "KNN"]:
            X_upload_scaled = scaler.transform(X_upload)
            preds = selected_model.predict(X_upload_scaled)
        else:
            preds = selected_model.predict(X_upload)

        st.subheader("Confusion Matrix (Uploaded Data)")
        st.dataframe(pd.DataFrame(confusion_matrix(y_upload, preds)))

        st.subheader("Classification Report (Uploaded Data)")
        upload_report = classification_report(
            y_upload,
            preds,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(upload_report).transpose().round(4))

# ==========================================
# Model Performance Comparison (All Models)
# ==========================================

st.markdown("---")
st.header("Model Performance Comparison (All Models)")

comparison_results = []

for name, model in models.items():

    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred_all = model.predict(X_test_scaled)
        y_prob_all = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred_all = model.predict(X_test)
        y_prob_all = model.predict_proba(X_test)[:, 1]

    comparison_results.append([
        name,
        accuracy_score(y_test, y_pred_all),
        roc_auc_score(y_test, y_prob_all),
        precision_score(y_test, y_pred_all),
        recall_score(y_test, y_pred_all),
        f1_score(y_test, y_pred_all),
        matthews_corrcoef(y_test, y_pred_all)
    ])

comparison_df = pd.DataFrame(comparison_results, columns=[
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC Score"
])

st.dataframe(comparison_df.round(4))
