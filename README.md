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

The final selected model for fraud detection was a Tuned Random Forest classifier (RF_tuned). 
This model was chosen based on PR-AUC, Recall, F1-score, and Precision — the key metrics for 
imbalanced fraud detection. Hyperparameter tuning significantly improved model precision and PR-AUC 
while maintaining high recall.

Overall Performance on the Test Set (RF_tuned):

Accuracy 0.91
Precision (Fraud) 0.65
Recall (Fraud) 0.72
F1-score (Fraud) 0.685
ROC-AUC 0.94
PR-AUC 0.738

Interpretation of Model Behavior:

High ROC-AUC (0.94)
Shows that the model is excellent at separating fraudulent from legitimate providers across all decision thresholds.

Strong PR-AUC (0.738)
This is especially important in highly imbalanced datasets (~9% fraud).
It indicates the model ranks fraud cases much better than baseline and handles imbalance effectively.

Recall of 0.72
The model successfully captures 72% of all fraudulent providers, meeting the project’s priority of maximizing fraud detection.

Precision of 0.65
This means 65% of providers flagged as “fraud” are truly fraudulent, reducing unnecessary investigations compared to earlier models.
This balance between Precision and Recall is ideal for fraud detection, where missing fraud is more costly than investigating some legitimate cases.

Error Analysis Findings:
False Positives (FP): 39 
False Negatives (FN): 28

False Positives tend to have unusually high predicted fraud scores (~0.71) and show aggressive 
claim behavior.
False Negatives show low predicted fraud scores (~0.24) and are typically low-activity providers.

After comparing all baseline models, the top-performing models were:

1. Random Forest (Class Weight) – highest PR-AUC
2. Random Forest (SMOTE) – strongest Recall & F1

A lightweight GridSearchCV tuning was applied to Random Forest using a reduced hyperparameter 
grid (n_estimators, max_depth, min_samples_split). The tuning produced the final best model:

RF_tuned:
- n_estimators = 200
- max_depth = 10
- min_samples_split = 5

This tuned model achieved the best balance of Recall, Precision, F1-score, and PR-AUC.


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


