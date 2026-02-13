# ==========================================
# ML Assignment 2
# Model Training and Evaluation Script
# ==========================================

import os
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
    matthews_corrcoef
)

# ==========================================
# Step 1: Load Dataset
# ==========================================

data = load_breast_cancer()
X = data.data
y = data.target

feature_names = data.feature_names

print("Dataset Shape:", X.shape)
print("Number of Features:", len(feature_names))

# ==========================================
# Step 2: Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Set Shape:", X_train.shape)
print("Test Set Shape:", X_test.shape)

# ==========================================
# Step 3: Feature Scaling
# ==========================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Step 4: Define Models
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
}

# ==========================================
# Step 5: Train & Evaluate Models
# ==========================================

results = []

for name, model in models.items():

    print("\nTraining:", name)

    # Scaling required only for LR and KNN
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

# ==========================================
# Step 6: Save Results to CSV
# ==========================================

# Create folder if not exists
os.makedirs("model", exist_ok=True)

results_df = pd.DataFrame(results, columns=[
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC Score"
])

results_df.to_csv("model/model_results.csv", index=False)

print("\nModel Evaluation Completed Successfully!")
print(results_df)
