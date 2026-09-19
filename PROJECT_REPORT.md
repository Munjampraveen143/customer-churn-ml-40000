# Customer Churn Prediction Using Machine Learning

## 1. Introduction

Customer churn refers to a customer discontinuing their relationship with a company or service provider. Predicting churn can help organizations identify customers who may be at risk and better understand the factors associated with customer retention.

This project develops an end-to-end machine learning system for predicting customer churn using customer demographic, service, billing, payment, support, and satisfaction information.

The system includes data preprocessing, feature engineering, machine learning model training, evaluation, model verification, a FastAPI prediction service, and an interactive Streamlit dashboard.

---

## 2. Problem Statement

The objective of this project is to develop a machine learning model that predicts whether a customer is likely to churn.

The target variable is:

* `0` — No Churn
* `1` — Churn

The system should accept customer information and produce a churn prediction together with a churn probability and risk category.

---

## 3. Project Objectives

The main objectives are:

1. Prepare customer data for machine learning.
2. Handle missing values and categorical variables.
3. Engineer additional customer-risk features.
4. Compare multiple machine learning algorithms.
5. Tune model hyperparameters.
6. Evaluate the final model using multiple classification metrics.
7. Save the trained model for future predictions.
8. Provide a REST API for predictions.
9. Provide an interactive dashboard.
10. Verify the saved model and API using automated tests.

---

## 4. Dataset

The project uses a synthetic dataset containing:

* 40,000 customer records
* 24 columns

The dataset contains demographic, service, contract, billing, payment, support, and satisfaction information.

### Important Variables

| Category          | Variables                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| Demographics      | gender, age, senior_citizen, partner, dependents                                                 |
| Services          | phone_service, internet_service, online_security, online_backup, device_protection, tech_support |
| Streaming         | streaming_tv, streaming_movies                                                                   |
| Contract          | contract, paperless_billing, payment_method                                                      |
| Billing           | monthly_charges, total_charges                                                                   |
| Customer behavior | tenure, support_tickets, late_payments                                                           |
| Satisfaction      | satisfaction_score                                                                               |
| Target            | churn                                                                                            |

---

## 5. Data Preprocessing

The preprocessing pipeline separates numerical and categorical variables.

### Numerical Processing

Numerical features are processed using:

* Median imputation
* Standard scaling

### Categorical Processing

Categorical features are processed using:

* Most-frequent-value imputation
* One-hot encoding
* Unknown-category handling

The preprocessing operations are included in the machine learning pipeline so that the same transformations are applied during prediction.

---

## 6. Feature Engineering

Additional features were created to provide the model with additional customer-level signals.

### Average Monthly Spend

Calculated from total charges and tenure.

### Number of Services

Counts the number of subscribed services across security, backup, protection, support, and streaming services.

### Customer Risk Signals

Combines support tickets and late payments.

### Charge Per Tenure

Combines monthly charges with customer tenure.

### Low Satisfaction

Identifies customers with satisfaction scores of 2 or below.

### Payment Problem

Identifies customers with two or more late payments.

### Short Tenure

Identifies customers with tenure of 12 months or less.

### Long Tenure

Identifies customers with tenure of 36 months or more.

---

## 7. Machine Learning Models

Four algorithms were evaluated:

1. Random Forest
2. Extra Trees
3. XGBoost
4. Logistic Regression

Randomized hyperparameter search with 3-fold cross-validation was used during model tuning.

### Cross-Validation Results

| Model               | Best CV Accuracy |
| ------------------- | ---------------: |
| Random Forest       |           71.63% |
| Extra Trees         |           70.79% |
| XGBoost             |           72.42% |
| Logistic Regression |           71.45% |

XGBoost produced the highest cross-validation accuracy and was selected as the final model.

---

## 8. Final Model Evaluation

The final model was evaluated using an untouched test set containing 8,000 customers.

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 72.02% |
| Precision | 74.75% |
| Recall    | 82.26% |
| F1 Score  | 78.33% |
| ROC-AUC   | 78.17% |

The model correctly classified:

**5,762 out of 8,000 customers.**

It incorrectly classified:

**2,238 customers.**

---

## 9. Confusion Matrix

The final confusion matrix was:

|                 | Predicted No Churn | Predicted Churn |
| --------------- | -----------------: | --------------: |
| Actual No Churn |              1,718 |           1,366 |
| Actual Churn    |                872 |           4,044 |

This shows that the model correctly identified 4,044 customers who churned in the held-out test set.

---

## 10. Model Verification

The saved model was independently verified using the same held-out test split.

The verification reproduced the evaluation results:

* Accuracy: 72.02%
* Precision: 74.75%
* Recall: 82.26%
* F1 Score: 78.33%
* ROC-AUC: 78.17%

This confirms consistency between the trained model, saved model, and prediction pipeline.

---

## 11. FastAPI Backend

A FastAPI backend was developed to provide real-time predictions.

### API Endpoints

#### Health Check

`GET /health`

Used to verify that the API is running.

#### Prediction

`POST /predict`

Accepts customer information and returns:

* Prediction
* Churn label
* Churn probability
* Risk level

### Risk Classification

The application uses the following probability thresholds:

| Probability | Risk   |
| ----------- | ------ |
| < 0.40      | LOW    |
| 0.40–0.69   | MEDIUM |
| >= 0.70     | HIGH   |

---

## 12. Streamlit Dashboard

A Streamlit dashboard provides an interactive interface for customer churn prediction.

Users can enter customer information including:

* Demographics
* Services
* Contract information
* Billing information
* Payment behavior
* Support activity
* Satisfaction score

The dashboard then performs the same feature engineering used during model training and displays the prediction and risk level.

---

## 13. Automated Testing

Automated API tests were implemented using Pytest and FastAPI's testing client.

The final test suite produced:

**3 passed**

This verifies the main API functionality, including the health endpoint and prediction behavior.

---

## 14. Project Architecture

The overall workflow is:

Customer Data

↓

Data Preprocessing

↓

Feature Engineering

↓

XGBoost Model

↓

Prediction

↓

Churn Probability

↓

Risk Classification

↓

FastAPI / Streamlit

---

## 15. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* FastAPI
* Pydantic
* Streamlit
* Pytest
* Joblib

---

## 16. Project Files

Important project components include:

* `src/train.py` — model training
* `src/evaluate.py` — model evaluation
* `src/eda.py` — exploratory data analysis
* `src/predict.py` — prediction utilities
* `verify_model.py` — model verification
* `api/main.py` — FastAPI backend
* `dashboard/app.py` — Streamlit dashboard
* `models/churn_model.pkl` — trained model
* `reports/` — evaluation charts and results
* `tests/test_api.py` — automated API tests
* `requirements.txt` — Python dependencies
* `README.md` — project documentation

---

## 17. Example API Prediction

A test customer was submitted through the live API.

The API returned:

* HTTP status: `200`
* Prediction: `Churn`
* Churn probability: `97.9%`
* Risk: `HIGH`

This demonstrates that the deployed API can successfully process customer information and return a prediction.

The 97.9% value is the probability produced for that individual test customer and should not be interpreted as the overall model accuracy.

---

## 18. Limitations

The dataset used in this project is synthetic, so performance on this dataset does not guarantee the same performance on real-world customer data.

The model achieved 72.02% accuracy on the held-out test set. Model performance can change when applied to a different population, different time period, or real-world operational data.

The system should therefore be evaluated and monitored before being used for real business decisions.

---

## 19. Future Improvements

Potential future improvements include:

* Testing additional algorithms
* More extensive hyperparameter optimization
* Probability calibration
* Threshold optimization based on business costs
* Explainable AI using SHAP
* Model monitoring
* Real-world customer data validation
* Database integration
* Authentication for the API
* Cloud deployment
* Automated retraining

---

## 20. Conclusion

This project demonstrates a complete machine learning workflow for customer churn prediction.

The workflow begins with customer data preprocessing and feature engineering, followed by model comparison and hyperparameter tuning. XGBoost was selected based on cross-validation performance and achieved 72.02% accuracy, 82.26% recall, 78.33% F1 score, and 78.17% ROC-AUC on the held-out test set.

The trained model was successfully integrated into a FastAPI backend and Streamlit dashboard. Automated tests also confirmed that the API functions correctly.

Overall, the project provides a complete foundation for a customer churn prediction application, while recognizing that additional validation would be required before applying the system to real-world customer data.
