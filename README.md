# Bank Marketing Campaign Prediction

## Overview

This project develops and compares multiple machine learning models to predict whether a customer will subscribe to a term deposit using the UCI Bank Marketing dataset. The workflow includes exploratory data analysis, feature engineering, feature selection, hyperparameter tuning, model evaluation, and model interpretation.

---

## Dataset

- **Source:** UCI Machine Learning Repository – Bank Marketing Dataset
- **Observations:** 45,211 customers
- **Features:** 16 input features
- **Target:** Predict whether a customer subscribes to a term deposit (`yes` or `no`)

---

## Machine Learning Models

- Logistic Regression
- Random Forest
- Gaussian Naive Bayes
- XGBoost

---

## Project Workflow

1. Data preprocessing
2. Exploratory data analysis (EDA)
3. Feature selection using Random Forest feature importance
4. Model training and hyperparameter tuning with 3-fold cross-validation
5. Final model evaluation
6. Model interpretation using SHAP

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision
- Precision-Recall Curve

---

## Visualizations

- ROC Curve
- Precision-Recall Curve
- SHAP Feature Importance

## Business Analysis

- Revenue, Cost, and Profit vs. Probability Threshold
- Campaign Reach vs. Successful Subscriptions

## Results

The best-performing model was a Random Forest classifier 
(n_estimators = 500, max_depth = 10) trained on the top 10 selected 
features. The model was chosen based on the highest mean cross-validated 
F1 score and evaluated on a held-out test set.





