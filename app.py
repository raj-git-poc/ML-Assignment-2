# ==========================================
# ML Assignment 2 - Streamlit App
# Auto-Generates model_results.csv
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import os

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

st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("Machine Learning Classification Models Comparison")

# =====================================================
# STEP 1: Load Dataset
# =====================================================

data = load_breast_cancer()
X = data.data
y = data.target

# =====================================================
# STEP 2: Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# STEP 3: Scaling
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# STEP 4: Define Models
# =====================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# =====================================================
# STEP 5: Train & Evaluate ALL Models
# =====================================================

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

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

# =====================================================
# STEP 6: Create model_results.csv Automatically
# =====================================================

results_df = pd.DataFrame(results, columns=[
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC Score"
])

# Create model folder if not exists
if not os.path.exists("model"):
    os.makedirs("model")

results_df.to_csv("model/model_results.csv", index=False)

# =====================================================
# Display Results Table
# =====================================================

st.subheader("Model Performance Comparison")
st.dataframe(results_df)

# =====================================================
# Model Selection Dropdown
# =====================================================

model_choice = st.selectbox(
    "Select a Model",
    results_df["Model"]
)

# =====================================================
# Dataset Upload Feature
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Test CSV File (Must contain 'target' column)",
    type=["csv"]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(test_data.head())

    if "target" not in test_data.columns:
        st.error("The uploaded file must contain a 'target' column.")
    else:

        X_upload = test_data.drop("target", axis=1)
        y_upload = test_data["target"]

        model = models[model_choice]

        if model_choice in ["Logistic Regression", "KNN"]:
            model.fit(X_train_scaled, y_train)
            X_upload_scaled = scaler.transform(X_upload)
            predictions = model.predict(X_upload_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_upload)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_upload, predictions)
        st.write(cm)

        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_upload, predictions)
        st.text(report)
