
Machine Learning Classification Models – Assignment 2
1.	Problem Statement
The objective of this assignment is to implement multiple machine learning classification models on a public dataset and compare their performance using standard evaluation metrics.
An interactive Streamlit web application has also been developed to:
•	Compare model performances
•	Allow model selection
•	Display evaluation metrics
•	Show confusion matrix and classification report
•	Allow dataset upload for prediction and validation
This project demonstrates a complete end-to-end Machine Learning workflow including:
•	Data preprocessing
•	Feature scaling
•	Model training
•	Model evaluation
•	Performance comparison
•	Web deployment using Streamlit
________________________________________
2.	Dataset Description
Dataset Name: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
The dataset is also available in Scikit-learn via:
from sklearn.datasets import load_breast_cancer
Dataset Details
•	Total Instances: 569
•	Total Features: 30 numerical features
•	Target Classes:
o	0 → Malignant
o	1 → Benign
•	Problem Type: Binary Classification
This dataset satisfies the assignment requirements:
•	Minimum 500 instances ✅
•	Minimum 12 features ✅
Data Preprocessing Steps
•	Data split into:
o	80% Training set
o	20% Testing set
•	Feature scaling applied using StandardScaler
o	Required for Logistic Regression and KNN
•	Stratified sampling used to maintain class balance
________________________________________
3.	Models Implemented & Evaluation Metrics
The following 6 machine learning models were implemented:
1.	Logistic Regression
2.	Decision Tree Classifier
3.	K-Nearest Neighbors (KNN)
4.	Naive Bayes (Gaussian)
5.	Random Forest (Ensemble – Bagging)
6.	XGBoost (Extreme Gradient Boosting – Ensemble Boosting Model)
About XGBoost
XGBoost is an optimized implementation of Gradient Boosting that includes:
•	Regularization
•	Efficient tree boosting
•	Parallel processing
•	Better generalization performance
It is widely used in industry and machine learning competitions due to its high predictive accuracy.
________________________________________
Evaluation Metrics Used
Each model was evaluated using the following metrics:
•	Accuracy
•	AUC (Area Under ROC Curve)
•	Precision
•	Recall
•	F1 Score
•	Matthews Correlation Coefficient (MCC)
The results are automatically saved in:
model/model_results.csv
________________________________________
4.	Model Performance Comparison
ML Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC Score
Logistic Regression	Generated in app					
Decision Tree	Generated in app					
KNN	Generated in app					
Naive Bayes	Generated in app					
Random Forest	Generated in app					
XGBoost	Generated in app					
Note: Exact values are dynamically generated in the Streamlit application and stored in model_results.csv.
________________________________________
5.	Observations on Model Performance
ML Model	Observation
Logistic Regression	Performs very well due to good linear separability of features. Balanced bias-variance tradeoff.
Decision Tree	Performs reasonably well but may slightly overfit without pruning.
KNN	Sensitive to feature scaling; performs well after normalization.
Naive Bayes	Assumes feature independence; performs decently but slightly lower than ensemble models.
Random Forest	Strong performance due to ensemble learning and reduced overfitting.
XGBoost	Typically achieves the highest performance due to optimized boosting, regularization, and efficient tree construction.
Overall, ensemble models (Random Forest and XGBoost) generally outperform individual classifiers due to better generalization and variance reduction.
________________________________________
6.	Streamlit Application Features
The deployed Streamlit application includes:
•	Dataset upload option (CSV file with target column)
•	Model selection dropdown
•	Evaluation on uploaded dataset
•	Evaluation metrics display:
    o	Accuracy
    o	AUC
    o	Precision
    o	Recall
    o	F1 Score
    o	MCC
•	Confusion Matrix display
•	Classification Report display
•	Model performance comparison table
The application ensures interactive evaluation and visualization of model performance.
