# Fraud Detection for E-Commerce and Banking

## Overview
This project focuses on **detecting fraudulent transactions** in e-commerce and banking datasets using machine learning. The goal is to improve fraud detection while handling severe class imbalance and providing interpretable insights for financial risk management.

---

## Dataset
- The processed dataset (`fraud_data_processed.csv`) includes:
  - **Time-based features**: transaction timestamps, frequency, etc.
  - **Behavioral features**: customer behavior patterns.
  - **Geolocation features**: transaction location information.
- **Target variable:** `class` (0 = non-fraud, 1 = fraud)
- The dataset is **highly imbalanced**, with fraudulent transactions representing a very small fraction of the data.

> **Note:** Raw data is not included due to privacy; only the processed CSV is required to run the notebook.

---

## Project Structure

.
├── data/
│ └── fraud_data_processed.csv
├── models/
│ ├── logistic_regression.pkl
│ └── random_forest.pkl
├── notebooks/
│ └── modeling_and_evaluation.ipynb
├── README.md


- `notebooks/`: Jupyter notebook containing full preprocessing, model training, evaluation, and cross-validation.  
- `models/`: Saved trained models (`.pkl`) for Logistic Regression and Random Forest.  
- `data/`: Contains processed dataset used for modeling.

---

## Preprocessing
1. **Missing Values:** All missing values were imputed or removed to ensure models can train correctly.  
2. **Categorical Features:** Encoded using `OneHotEncoder` to convert categories into numeric features.  
3. **Feature Scaling:** Logistic Regression benefits from normalized features (handled via the model or preprocessing if needed).  
4. **Class Imbalance:** Addressed using **SMOTE** (Synthetic Minority Over-sampling Technique) applied **only to the training set** to avoid data leakage.

---

## Models
Two machine learning models were trained and evaluated:

1. **Logistic Regression**
   - Interpretable baseline model.
   - Weighted classes to account for imbalance (`class_weight="balanced"`).
   - Metrics: F1-score, PR-AUC.

2. **Random Forest Classifier**
   - Ensemble model capturing non-linear interactions.
   - Higher capacity improves detection of rare fraud cases.
   - Metrics: F1-score, PR-AUC.
   - Feature importance can be extracted for interpretability.

---

## Evaluation
- **Metrics used:**  
  - **F1-Score**: Balances precision and recall in imbalanced datasets.  
  - **Precision-Recall AUC (PR-AUC)**: Preferred over ROC-AUC for highly imbalanced data.  

- **Cross-Validation:**  
  - Stratified K-Fold (5 splits) was used to get a robust estimate of model performance.  

- **Results Summary:**

| Model                 | F1 Score | PR-AUC |
|-----------------------|----------|--------|
| Logistic Regression   | 0.XX     | 0.XX   |
| Random Forest         | 0.XX     | 0.XX   |

> Random Forest achieves higher detection performance, while Logistic Regression provides interpretability.
