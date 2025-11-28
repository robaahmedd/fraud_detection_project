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

The final selected model for fraud detection was a Random Forest classifier combined with SMOTE to address severe class imbalance in the dataset. The model demonstrated strong overall performance, especially in recovering fraudulent providers (positive class), which is the primary objective in healthcare fraud detection systems.

Overall Performance on the Test Set:
Accuracy        	0.91
Precision (Fraud)	0.53
Recall (Fraud)	        0.72
F1-score (Fraud)	0.61
ROC-AUC           	0.93
PR-AUC          	0.65

Interpretation of Model Behavior:
High ROC-AUC (0.93) indicates the model is very good at distinguishing fraudulent from legitimate providers across all classification thresholds.

Moderate PR-AUC (0.65) is expected given the dataset’s strong imbalance (only ~9% fraud cases). This value reflects the realistic difficulty of identifying rare fraud patterns.

Recall of 0.72 for fraudulent providers shows that the model successfully captures 72% of all fraud cases, meeting the project’s priority of maximizing fraud detection.

Precision of 0.53 indicates that roughly half of providers flagged as “fraud” are actually legitimate, which is acceptable in fraud prevention settings, where catching fraud is far more critical than investigating a few legitimate cases.

Error Analysis Findings:
False Positives (FP)	65	
Legitimate providers flagged as fraud.
False Positives tend to have unusually high predicted fraud scores (y_proba = 0.71)
False Negatives (FN)	28
Fraudulent providers missed by the model.
False Negatives show low predicted fraud scores (y_proba = 0.24)


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


