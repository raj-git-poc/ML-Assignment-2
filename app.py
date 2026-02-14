# ==========================================
# Load Library
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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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
import requests
# ==========================================
# Load Dataset
# ==========================================
data = load_breast_cancer()
X = data.data  # Features (30 features)
y = data.target  # Labels (0: malignant, 1: benign)
# Split 80/20 training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

record_count = len(data.data)
target_count = len(data.target)
feature_count = len(data.feature_names)

# ==========================================
# Define Models
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(),
    "XGBoost (Ensemble)": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

# ==========================================
# Download Sample Test Data (CSV)
# ==========================================

st.header("Download Sample Test Data (CSV)")

# GitHub raw file URL
url = "https://raw.githubusercontent.com/raj-git-poc/TestRepo2Raj/edit/main/sample_test_data.csv"

# Fetch the file content
response = requests.get(url)
file_bytes = response.content

st.download_button(
    label="Download Sample Test Data CSV",
    data=file_bytes,
    file_name="sample_test_data.csv",
    mime="text/csv"
)

# ==========================================
# Dataset upload option (CSV)
# ==========================================
st.header("Dataset Upload Option")
uploaded_file = st.file_uploader(
    "Upload Test CSV File (Must contain 'target' column)",
    type=["csv"]
)

# ==========================================
# Model selection dropdown (if multiple models)
# ==========================================

st.header("Model Selection")

model_options_dropdown = st.selectbox("Select a Model", list(models.keys()))
selected_model = models[model_options_dropdown]
#st.text("Selected Model: " + model_options_dropdown)
# Train Selected Model
if model_options_dropdown in ["Logistic Regression", "KNN"]:
    selected_model.fit(X_train_scaled, y_train)
    y_pred_test = selected_model.predict(X_test_scaled)
    y_prob_test = selected_model.predict_proba(X_test_scaled)[:, 1]
else:
    selected_model.fit(X_train, y_train)
    y_pred_test = selected_model.predict(X_test)
    y_prob_test = selected_model.predict_proba(X_test)[:, 1]

# ==========================================
# Uploaded Dataset Evaluation (First)
# ==========================================

if uploaded_file is not None:
    st.markdown("---")
    st.header("Evaluation on Uploaded Dataset")    
    uploaded_data = pd.read_csv(uploaded_file)
    if "target" not in uploaded_data.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        st.text("Uploaded CSV has a 'target' column.")
        X_uploadedData = uploaded_data.drop("target", axis=1)
        y_uploadedData = uploaded_data["target"]
        if model_options_dropdown in ["Logistic Regression", "KNN"]:
            X_upload_scaled = scaler.transform(X_uploadedData)
            y_pred_uploadedData = selected_model.predict(X_upload_scaled)
            y_prob_uploadedData = selected_model.predict_proba(X_upload_scaled)[:, 1]
        else:
            y_pred_uploadedData = selected_model.predict(X_uploadedData)
            y_prob_uploadedData = selected_model.predict_proba(X_uploadedData)[:, 1]

        st.subheader("Evaluation Metrics (Uploaded Data)")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(accuracy_score(y_uploadedData, y_pred_uploadedData), 4))
        col2.metric("AUC", round(roc_auc_score(y_uploadedData, y_prob_uploadedData), 4))
        col3.metric("Precision", round(precision_score(y_uploadedData, y_pred_uploadedData), 4))

        col1.metric("Recall", round(recall_score(y_uploadedData, y_pred_uploadedData), 4))
        col2.metric("F1 Score", round(f1_score(y_uploadedData, y_pred_uploadedData), 4))
        col3.metric("MCC Score", round(matthews_corrcoef(y_uploadedData, y_pred_uploadedData), 4))

        st.subheader("Confusion Matrix (Uploaded Data)")
        st.dataframe(pd.DataFrame(confusion_matrix(y_uploadedData, y_pred_uploadedData)))

        st.subheader("Classification Report (Uploaded Data)")
        report_upload = classification_report(
            y_uploadedData,
            y_pred_uploadedData,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report_upload).transpose().round(4))

# ==========================================
# Evaluation Metrics (Model Dataset)
# ==========================================
st.header("Evaluation Metrics (Test Dataset)")
col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", round(accuracy_score(y_test, y_pred_test), 4))
col2.metric("AUC", round(roc_auc_score(y_test, y_prob_test), 4))
col3.metric("Precision", round(precision_score(y_test, y_pred_test), 4))

col1.metric("Recall", round(recall_score(y_test, y_pred_test), 4))
col2.metric("F1 Score", round(f1_score(y_test, y_pred_test), 4))
col3.metric("MCC Score", round(matthews_corrcoef(y_test, y_pred_test), 4))

# ==========================================
# Confusion Matrix (Model Dataset)
# ==========================================

st.header("Confusion Matrix (Model Dataset)")

st.subheader("Confusion Matrix")
st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred_test)))

# ==========================================
# Classification Report (Model Dataset)
# ==========================================

st.subheader("Classification Report (Model Dataset)")
report_test = classification_report(
    y_test,
    y_pred_test,
    output_dict=True
)
st.dataframe(pd.DataFrame(report_test).transpose().round(4))

# ==========================================
# Model Performance Comparison (All Models Dataset)
# ==========================================

st.markdown("---")
st.header("Model Performance Comparison (All Models Dataset)")

comparison_data = []

for name in models.keys():

    model_instance = models[name]

    try:
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

    except Exception as e:
        st.warning(f"{name} failed: {e}")

# Only display if data exists
if len(comparison_data) > 0:

    comparison_df = pd.DataFrame(comparison_data, columns=[
        "ML Model Name",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ])

    st.dataframe(comparison_df.round(4), use_container_width=True)

else:
    st.error("No models were successfully evaluated.")
