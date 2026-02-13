# ==========================================
# ML Assignment 2 - Final Streamlit App
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
# Page Configuration
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
# a) Dataset Upload Option
# ==========================================

st.header("a) Dataset Upload Option")

uploaded_file = st.file_uploader(
    "Upload Test CSV File (Must contain 'target' column)",
    type=["csv"]
)

# ==========================================
# b) Model Selection Dropdown
# ==========================================

st.header("b) Model Selection")

model_choice = st.selectbox("Select a Model", list(models.keys()))
selected_model = models[model_choice]

# Train selected model
if model_choice in ["Logistic Regression", "KNN"]:
    selected_model.fit(X_train_scaled, y_train)
    y_pred_test = selected_model.predict(X_test_scaled)
    y_prob_test = selected_model.predict_proba(X_test_scaled)[:, 1]
else:
    selected_model.fit(X_train, y_train)
    y_pred_test = selected_model.predict(X_test)
    y_prob_test = selected_model.predict_proba(X_test)[:, 1]

# ==========================================
# Uploaded Dataset Evaluation (Displayed First)
# ==========================================

if uploaded_file is not None:

    st.markdown("---")
    st.header("Evaluation on Uploaded Dataset")

    uploaded_data = pd.read_csv(uploaded_file)

    if "target" not in uploaded_data.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        X_upload = uploaded_data.drop("target", axis=1)
        y_upload = uploaded_data["target"]

        if model_choice in ["Logistic Regression", "KNN"]:
            X_upload_scaled = scaler.transform(X_upload)
            y_pred_upload = selected_model.predict(X_upload_scaled)
            y_prob_upload = selected_model.predict_proba(X_upload_scaled)[:, 1]
        else:
            y_pred_upload = selected_model.predict(X_upload)
            y_prob_upload = selected_model.predict_proba(X_upload)[:, 1]

        # Evaluation Metrics
        st.subheader("Evaluation Metrics (Uploaded Data)")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(accuracy_score(y_upload, y_pred_upload), 4))
        col2.metric("AUC", round(roc_auc_score(y_upload, y_prob_upload), 4))
        col3.metric("Precision", round(precision_score(y_upload, y_pred_upload), 4))

        col1.metric("Recall", round(recall_score(y_upload, y_pred_upload), 4))
        col2.metric("F1 Score", round(f1_score(y_upload, y_pred_upload), 4))
        col3.metric("MCC Score", round(matthews_corrcoef(y_upload, y_pred_upload), 4))

        # Confusion Matrix
        st.subheader("Confusion Matrix (Uploaded Data)")
        cm_upload = confusion_matrix(y_upload, y_pred_upload)
        st.dataframe(pd.DataFrame(cm_upload))

        # Classification Report
        st.subheader("Classification Report (Uploaded Data)")
        report_upload = classification_report(
            y_upload,
            y_pred_upload,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report_upload).transpose().round(4))

# ==========================================
# c) Evaluation Metrics (Test Dataset)
# ==========================================

st.markdown("---")
st.header("c) Evaluation Metrics (Test Dataset)")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", round(accuracy_score(y_test, y_pred_test), 4))
col2.metric("AUC", round(roc_auc_score(y_test, y_prob_test), 4))
col3.metric("Precision", round(precision_score(y_test, y_pred_test), 4))

col1.metric("Recall", round(recall_score(y_test, y_pred_test), 4))
col2.metric("F1 Score", round(f1_score(y_test, y_pred_test), 4))
col3.metric("MCC Score", round(matthews_corrcoef(y_test, y_pred_test), 4))

# ==========================================
# d) Confusion Matrix & Classification Report (Test)
# ==========================================

st.header("d) Confusion Matrix and Classification Report (Test Dataset)")

st.subheader("Confusion Matrix")
cm_test = confusion_matrix(y_test, y_pred_test)
st.dataframe(pd.DataFrame(cm_test))

st.subheader("Classification Report")
report_test = classification_report(
    y_test,
    y_pred_test,
    output_dict=True
)
st.dataframe(pd.DataFrame(report_test).transpose().round(4))

# ==========================================
# Model Performance Comparison
# ==========================================

st.markdown("---")
st.header("Model Performance Comparison (All Models)")

comparison_data = []

for name, model_instance in models.items():

    if name in ["Logistic Regression", "KNN"]:
        model_instance.fit(X_train_scaled, y_train)
        y_pred = model_instance.predict(X_test_scaled)
        y_prob = model_instance.predict_proba(X_test_scaled)[:, 1]
    else:
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        y_prob = model_instance.predict_proba(X_test)[:, 1]

    comparison_data.append([
        name,
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ])

comparison_df = pd.DataFrame(comparison_data, columns=[
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC Score"
])

st.dataframe(comparison_df.round(4))
