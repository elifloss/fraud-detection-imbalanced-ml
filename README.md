Fraud Detection with Imbalanced Learning
End-to-end machine learning pipeline for credit card fraud detection on a severely imbalanced dataset (0.17% fraud rate). Compares 7 model configurations across baseline, class-weighting, and SMOTE oversampling strategies — evaluated on metrics that actually matter under imbalance.

The Problem
Standard accuracy is useless here. A model that predicts "legitimate" for every transaction achieves 99.83% accuracy while catching zero fraud. This project focuses on PR-AUC, ROC-AUC, and threshold-optimized F1 — the metrics that drive real fraud prevention decisions.

Results
ModelROC-AUCPR-AUCF1Random Forest (baseline)0.97710.86970.8743SMOTE + Random Forest0.98100.83360.6484Random Forest (balanced)0.97410.82410.8229Logistic Regression (baseline)0.95600.74330.7241SMOTE + Logistic Regression0.97000.72130.1061Logistic Regression (balanced)0.97140.71560.1105Gradient Boosting (balanced)0.59750.34550.5765
Best model: Random Forest (baseline) at optimal threshold of 0.419 — Precision: 93.3% · Recall: 84.7% · F1: 88.8%
Key finding: The baseline Random Forest outperforms SMOTE on F1 while being significantly faster to train. Class imbalance is better handled through threshold optimization than oversampling for this dataset.

Visualizations
Show Image
Show Image
Show Image
Show Image
Show Image
Show Image

Run It
bash# 1. Download the dataset
# kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in this directory

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full analysis
python fraud_detection_analysis.py

What This Demonstrates

Handling real-world class imbalance (284,807 transactions, 492 fraud cases)
Proper evaluation under imbalance: PR-AUC over accuracy, ROC curves, confusion matrices
Decision threshold optimization for business-specific precision/recall tradeoffs
SHAP-based model explainability identifying which features drive fraud predictions
Clean modular ML pipeline with reproducible results


Stack
Python · scikit-learn · imbalanced-learn · SHAP · pandas · NumPy · matplotlib · seaborn

Dataset
Kaggle Credit Card Fraud Detection — 284,807 transactions, 492 fraud cases (0.172%). Features V1–V28 are PCA-transformed for confidentiality.

Author
Elijah Legall · Penn State Data Science · GitHub · LinkedInSonnet 4.6Claude is AI and can make mistakes. Please double-check responses.
