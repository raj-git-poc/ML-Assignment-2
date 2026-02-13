# ==========================================
# ML Assignment 2 - Streamlit App
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

from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("Machine Learning Classification Models Comparison")

# ==========================================
# Load Results Table
# ==========================================

results_df = pd.read_csv("model/model_results.csv")

st.subheader("Model Performance Comparison")
st.dataframe(results_df)

# ==========================================
# Model Selection Dropdown
# ==========================================

model_choice = st.selectbox(
    "Select a Model",
    results_df["Model"]
)

# ==========================================
# Dataset Upload
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Test CSV File (Must contain target column)",
    type=["csv"]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(test_data.head())

    if "target" not in test_data.columns:
        st.error("The uploaded file must contain a 'target' column.")
    else:

        X = test_data.drop("target", axis=1)
        y = test_data["target"]

        # Load dataset structure for consistency
        data = load_breast_cancer()

        X_full = data.data
        y_full = data.target

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full,
            test_size=0.2,
            random_state=42,
            stratify=y_full
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_scaled = scaler.transform(X)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=5000),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        model = models[model_choice]

        # Train model
        if model_choice in ["Logistic Regression", "KNN"]:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X)

        # ==========================================
        # Confusion Matrix
        # ==========================================

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, predictions)
        st.write(cm)

        # ==========================================
        # Classification Report
        # ==========================================

        st.subheader("Classification Report")
        report = classification_report(y, predictions)
        st.text(report)
