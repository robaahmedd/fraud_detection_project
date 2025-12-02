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

After evaluating multiple models (Logistic Regression, Random Forest, Gradient Boosting) under both class-weighting and SMOTE strategies, two top-performing models emerged:

Random Forest (Class Weight)

Random Forest (SMOTE)

To comply with ML best practices and course requirements, the top model (Random Forest with class weights) was hyperparameter-tuned using a lightweight GridSearchCV.
This produced the best-performing model in the entire project, outperforming SMOTE-based models across all major metrics.

The final selected model is:
Random Forest (Tuned)

Overall Performance on the Test Set:

Accuracy	0.92
Precision (Fraud)	0.65
Recall (Fraud)	0.72
F1-score (Fraud)	0.69
ROC-AUC	0.94
PR-AUC	0.74

Interpretation of Model Behavior

High ROC-AUC (0.94)
Indicates excellent separation between fraud and non-fraud providers.

High PR-AUC (0.74)
Strong performance under severe class imbalance (fraud ≈ 9%).

Recall = 0.72
Captures 72% of fraudulent providers → meets project priority (catch fraud).

Precision = 0.65
Means 65% of flagged providers are actually fraudulent → strong for real-world fraud screening where false positives are acceptable.

Error Analysis Findings:
False Positives (FP): 51 
False Negatives (FN): 28

Key Insights:
Model effectively captures high-risk behavior.
Missed fraud cases tend to be low-volume providers.
Adding specialty-based normalization and temporal trends can reduce FN cases.

After comparing all baseline models, the two top-performing candidates were identified:
Random Forest (Class Weight) – highest PR-AUC and strongest precision
Random Forest (SMOTE) – strongest Recall & higher sensitivity to fraud

To refine and confirm the best performer, a lightweight GridSearchCV hyperparameter tuning was applied to the class-weighted Random Forest using a reduced hyperparameter grid (n_estimators, max_depth, min_samples_split).

The tuning produced the final best model:

RF_tuned (Final Model):

n_estimators = 200

max_depth = 10

min_samples_split = 5

This tuned model achieved the best overall balance of
Recall, Precision, F1-score, ROC-AUC, and PR-AUC, and therefore became the final selected model for the project.


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
This notebook: loads provider_level.csv, fixes class imbalance, trains multiple models and picks the best one and saves: best_model.pkl, test_predictions_for_evaluation.csv, these files will be used in Notebook 3.


Notebook 3: 03_evaluation.ipynb

This notebook: loads best_model.pkl, loads test_predictions_for_evaluation.csv, computes all metrics, plots ROC & PR curves, shows confusion matrix, shows false positives and false negatives and gives the final performance numbers.


