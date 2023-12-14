# Fraud Detection Dataset Analysis
## Overview:
I conducted an in-depth analysis of the Fraud Detection dataset, sourced from Kaggle, as an educational project. The dataset comprises over 20,000 records and 114 columns, including a target variable indicating whether a transaction is fraudulent (1) or not (0).

## Goal:
The primary objective of this analysis was to develop an effective prediction model to identify potential fraudant transactions, minimalizing undetected frauds (false negatives).

## Analysis Steps:
### 1. Data Preprocessing:
Removed unnecessary "Unnamed: 0" column serving as record ID.
Dropped columns with all zero values, reducing dimensions from 114 to 98.
### 2. Feature Exploration:
#### a. Correlation Analysis:
Investigated and retained features highly correlated with the target variable.
#### b. Data Types and Distribution:
Split the dataset into numerical and binary features.
Explored feature variances, identifying outliers and low-variance columns.
### 3. Preprocessing:
Split the dataset into training and testing sets.
Balanced the dataset using SMOTE to address class imbalance.
### 4. Model Training:
Utilized Logistic Regression as the chosen model due to its simplicity, interpretability, and efficiency.
Employed hyperparameter tuning using Optuna to enhance model performance.
### 5. Model Evaluation:
Achieved a ROC AUC score of 0.96 on the training set and 0.93 on the test set.
Conducted a detailed confusion matrix analysis with varying thresholds to optimize fraud detection.
### 6. Interpretability:
Leveraged SHAP (SHapley Additive exPlanations) values to interpret the model's predictions.
Examined the top features influencing the model's decisions.
### 7. Visualization:
Created visualizations such as dependence plots and partial dependence plots to enhance interpretability.
Results:
The final model demonstrated a robust ability to detect fraudulent transactions, with flexibility for further tuning and interpretation. This project showcases my skills in data preprocessing, model development, and interpretation for real-world problem-solving.

## Future Enhancements:
Continued efforts may include exploring advanced model architectures and comparing their performance.

