**a. Problem statement**

This project implements and compares multiple Machine Learning classification models using a real-world medical dataset.

The objective is to evaluate how different algorithms perform on the same dataset using standard performance metrics and present the results through an interactive Streamlit web application.

The project demonstrates a complete end-to-end ML workflow including:

Data preprocessing

Model training

Model evaluation

Performance comparison

Web deployment using Streamlit


**b. Dataset Description**

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

The dataset was loaded using:

from sklearn.datasets import load_breast_cancer

**Dataset Details:**

569 total instances

30 numerical features

Binary classification problem:

0 → Malignant

1 → Benign

This dataset satisfies assignment requirements:

Minimum 500 instances

Minimum 12 features

**Data Preprocessing**

The following steps were performed before training:

Dataset loading

Train-test split (80% training, 20% testing)

Feature scaling using StandardScaler (for models requiring scaling)

**c. Models used**

**The following six classification models were implemented:**

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbor Classifier

Naive Bayes Classifier - Gaussian

Ensemble Model - Random Forest

Ensemble Model - XGBoost

**Evaluation Metrics Used**

**Each model was evaluated using the following metrics:**

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC Score)

**Model results are stored in:**

model/model_results.csv

**Model Performance Comparison**

| ML Model Name              | Accuracy  | AUC     | Precision | Recall   | F1       | MCC     |
|----------------------------|-----------|---------|-----------|----------|----------|---------|
| Logistic Regression        | 0.9825    | 0.9954  | 0.9861    | 0.9861   | 0.9861   | 0.9623  |
| Decision Tree              | 0.9123    | 0.9157  | 0.9559    | 0.9028   | 0.9286   | 0.8174  |
| KNN                        | 0.9561    | 0.9788  | 0.9589    | 0.9722   | 0.9655   | 0.9054  |
| Naive Bayes                | 0.9386    | 0.9878  | 0.9452    | 0.9583   | 0.9517   | 0.8676  |
| Random Forest (Ensemble)   | 0.9561    | 0.9937  | 0.9589    | 0.9722   | 0.9655   | 0.9054  |
| XGBoost (Ensemble)         | 0.9561    | 0.9901  | 0.9467    | 0.9861   | 0.9660   | 0.9058  |


**Model Performance Comparison**

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Performs very well on this dataset due to good linear separability of features. Shows strong accuracy, precision, and AUC. Provides a good balance between bias and variance. |
| Decision Tree | Achieves good accuracy but may slightly overfit the training data. Performance is reasonable but less stable compared to ensemble methods. |
| K-Nearest Neighbors (KNN) | Performance improves significantly after feature scaling. Sensitive to the choice of K and distance metric. Works well but may be slower for larger datasets. |
| Naive Bayes | Performs moderately well. Since it assumes feature independence, it may not fully capture complex relationships between features, leading to slightly lower performance compared to other models. |
| Random Forest (Ensemble) | Shows strong and stable performance due to ensemble learning. Reduces overfitting by combining multiple decision trees and provides high generalization ability. |
| XGBoost (Ensemble) | Often achieves the best or near-best performance. Uses sequential boosting and regularization to reduce errors and overfitting. Captures complex patterns effectively and delivers high predictive power. |


**Streamlit Application Features**

The Streamlit web application includes:

  Dataset upload option (CSV with target column)
  
  Model selection dropdown
  
  Evaluation metrics display
  
  Confusion matrix
  
  Classification report
  
  Model performance comparison table

The app dynamically trains and evaluates models based on user selection.

**Project Structure**

```bash
ML_Assignment_2/
├── model/
│   ├── train_models.py         # Model training & evaluation script
│   └── model_results.csv       # Saved model performance results
└── README.md                   # Project documentation
├── app.py                      # Streamlit web application
|── breast_cancer_dataset.csv   # Training dataset downloaded from UCI Portal. This is not used as code is downloading the data directly at runtime
├── requirements.txt            # Required Python libraries
|── sample_test_data.csv        # sample test data to download and validate from Streamlit portal
```

Below mentioned libraries are used (in requirements.txt) for code execution and Streamlit deployment:

streamlit

scikit-learn

numpy

pandas

matplotlib

seaborn

xgboost

**How to Run the Project**

Option 1: Run Using Deployed Streamlit App (Recommended)

Access the application directly using the deployed URL:

https://ml-assignment-2-sit8csx4g6uujvztxniryk.streamlit.app/

The above URL is also mentioned in the assignment PDF file.

Option 2: Run Locally

Step 1: Install Dependencies

pip install -r requirements.txt

Step 2: Generate Model Results

python model/train_models.py
