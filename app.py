# ==========================================
# ML Assignment 2 - Final Working App
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
st.title("Machine Learning Classification Models Comparison")

# ==========================================
# Load Dataset
# ==========================================

data = load_breast_cancer()
X = data.data
y = data.target

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Scaling
# ==========================================

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
# Train & Evaluate All Models
# ==========================================

results = []

for name, model in models.items():

    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    results.append([
        name,
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ])

results_df = pd.DataFrame(results, columns=[
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC Score"
])

# ==========================================
# Display Comparison Table
# ==========================================

st.subheader("Model Performance Comparison")
st.dataframe(results_df.round(4))

# ==========================================
# Model Selection
# ==========================================

model_choice = st.selectbox("Select a Model", list(models.keys()))
selected_model = models[model_choice]

# ==========================================
# Display Evaluation Metrics
# ==========================================

st.subheader("Evaluation Metrics (Test Dataset)")

selected_row = results_df[results_df["Model"] == model_choice]

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", round(selected_row["Accuracy"].values[0], 4))
col2.metric("AUC", round(selected_row["AUC"].values[0], 4))
col3.metric("Precision", round(selected_row["Precision"].values[0], 4))

col1.metric("Recall", round(selected_row["Recall"].values[0], 4))
col2.metric("F1 Score", round(selected_row["F1 Score"].values[0], 4))
col3.metric("MCC Score", round(selected_row["MCC Score"].values[0], 4))

# ==========================================
# Confusion Matrix
# ==========================================

st.subheader("Confusion Matrix (Test Dataset)")

if model_choice in ["Logistic Regression", "KNN"]:
    selected_model.fit(X_train_scaled, y_train)
    predictions = selected_model.predict(X_test_scaled)
else:
    selected_model.fit(X_train, y_train)
    predictions = selected_model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

st.dataframe(cm_df)

# ==========================================
# Classification Report (Tabular)
# ==========================================

st.subheader("Classification Report (Test Dataset)")

report_dict = classification_report(
    y_test,
    predictions,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df.round(4))

# ==========================================
# Optional Dataset Upload
# ==========================================

st.markdown("---")
st.subheader("Upload Your Own Test CSV (Optional)")

uploaded_file = st.file_uploader(
    "Upload CSV with 'target' column",
    type=["csv"]
)

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    if "target" not in test_data.columns:
        st.error("CSV must contain 'target' column.")
    else:
        X_upload = test_data.drop("target", axis=1)
        y_upload = test_data["target"]

        if model_choice in ["Logistic Regression", "KNN"]:
            X_upload_scaled = scaler.transform(X_upload)
            preds = selected_model.predict(X_upload_scaled)
        else:
            preds = selected_model.predict(X_upload)

        st.subheader("Confusion Matrix (Uploaded Data)")
        upload_cm = confusion_matrix(y_upload, preds)
        st.dataframe(pd.DataFrame(upload_cm))

        st.subheader("Classification Report (Uploaded Data)")
        upload_report = classification_report(
            y_upload,
            preds,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(upload_report).transpose().round(4))
