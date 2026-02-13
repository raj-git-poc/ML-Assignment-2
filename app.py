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
# a) DATASET UPLOAD OPTION
# ==========================================

st.header("a) Dataset Upload Option")

uploaded_file = st.file_uploader(
    "Upload Test CSV File (Must contain 'target' column)",
    type=["csv"]
)

# ==========================================
# b) MODEL SELECTION
# ==========================================

st.header("b) Model Selection")

model
