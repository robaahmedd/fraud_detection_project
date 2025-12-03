# fraud_detection_project\
Machine Learning Project for Healthcare Provider Fraud Detection

This project implements a machine learning solution to assist the Centers for Medicare & Medicaid Services in detecting fraudulent healthcare providers. The primary objective is to build a robust model capable of identifying the highest-risk providers across four merged datasets, prioritizing Recall (catching fraud) while managing Precision (minimizing costly false investigations).

Goal: Develop a data-driven system to replace traditional rule-based methods and significantly reduce the estimated $68 billion annual cost of healthcare fraud.


Team & Role Assignments:
Yahia Walid – First Notebook: Data Exploration & Feature Engineering

Cleaned data and validated all four datasets

Merged inpatient, outpatient, and beneficiary data

Engineered provider-level features

Generated EDA plots

Exported final dataset: provider_level.csv


Abdelrhman Shady – Second Notebook: Class Imbalance & Modeling

Applied class imbalance techniques (SMOTE, class weights)

Trained Logistic Regression, Random Forest, and Gradient Boosting

Compared performance across metrics

Tuned hyperparameters

Saved final model: best_model.pkl

Exported predictions: test_predictions_for_evaluation.csv


Ali Yasser – First part of third notebook: Evaluation & Error Analysis

Computed confusion matrix, ROC-AUC, and PR-AUC

Analyzed false positives & false negatives

Identified behavioral patterns and risk indicators

Provided recommendations for model improvement
 
 
Roba Ahmed – Second part of third notebook + Documentation & Reporting

Generated evaluation plots and tables (ROC, PR, metrics summary)

Organized full project documentation

Wrote README.md, technical_report.pdf, and presentation.pptx



Summary of results:

Multiple models were evaluated under both class weighting and SMOTE.
The two strongest models before tuning were:

Random Forest (Class Weight) – higher Precision

Random Forest (SMOTE) – highest Recall & better sensitivity to fraud

To meet the project goal (maximizing fraud detection), the team selected Random Forest (with SMOTE) for final tuning.

A lightweight GridSearchCV hyperparameter tuning was then applied on the SMOTE version, leading to the final best-performing model.
Final Selected Model: Random Forest (SMOTE + Tuned)
Final Hyperparameters

n_estimators = 200

max_depth = 10

min_samples_split = 5

Overall Performance on the Test Set:

Accuracy	0.91
Precision (Fraud)	0.50
Recall (Fraud)	0.67
F1-score (Fraud)	0.58
ROC-AUC	0.922
PR-AUC	0.627

Interpretation of Model Behavior:
ROC-AUC = 0.922

Strong separation between fraud and non-fraud providers.

PR-AUC = 0.627

Solid performance under severe class imbalance (~9% fraud cases).

Recall = 0.67

Detects 67% of all fraud cases — matching the project’s highest priority.

Precision = 0.50

Error Analysis:
False Positives (FP): 67
Legitimate providers incorrectly flagged.
False Negatives (FN): 33
Fraudulent providers that were missed.
Insights:
FPs often show borderline-high fraud-probability scores.
FNs are mostly low-activity providers with subtle fraud patterns.
Additional temporal and specialty-based features may reduce FN cases.

Why This Model Was Selected

The SMOTE-tuned Random Forest offered the best overall balance for fraud detection:
Highest fraud Recall
High ROC-AUC
Competitive Precision
Strong PR-AUC
Best tradeoff between FP vs FN


Reproduction Instructions:
1. git clone https://github.com/robaahmedd/fraud_detection_project.git
or download using code then ZIP on GitHub

2.If running locally, install the required Python libraries
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn joblib

3. Download the dataset
Healthcare Provider Fraud Detection Analysis
https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
and upload files into first colab notebook: 01_data_exploration_and_feature_engineering

4. Run all 3 colabs in order:
Notebook 1: 01_data_exploration_and_feature_engineering.ipynb
This notebook: cleans the data, merges the datasets, creates features and saves a new file: provider_level.csv, this file will be used in Notebook 2.

Notebook 2: 02_modeling.ipynb
This notebook: loads provider_level.csv, fixes class imbalance, trains multiple models and picks the best one and saves: best_model.pkl, test_predictions_for_evaluation.csv, these files will be used in Notebook 3

Notebook 3: 03_evaluation.ipynb
This notebook: loads best_model.pkl, loads test_predictions_for_evaluation.csv, computes all metrics, plots ROC & PR curves, shows confusion matrix, shows false positives and false negatives and gives the final performance numbers.


