# Customer Churn Prediction using Machine Learning



## Project Overview



This project develops an end-to-end machine learning system for predicting customer churn.



The system analyzes customer demographic information, service usage, billing information, payment behavior, support activity, and satisfaction scores to predict whether a customer is likely to churn.



The project includes:



\- Data preprocessing

\- Feature engineering

\- Machine learning model training

\- Hyperparameter tuning

\- Model evaluation

\- Model verification

\- FastAPI prediction API

\- Streamlit dashboard

\---



## Dataset



The project uses a synthetic customer dataset containing \*\*40,000 customer records\*\* and \*\*24 columns\*\*.



### Target Variable



`churn`



\* `0` = Customer retained

\* `1` = Customer churned



### Main Features



| Category             | Features                                                                                         |

| -------------------- | ------------------------------------------------------------------------------------------------ |

| Customer information | gender, age, senior\_citizen, partner, dependents                                                 |

| Services             | phone\_service, internet\_service, online\_security, online\_backup, device\_protection, tech\_support |

| Streaming            | streaming\_tv, streaming\_movies                                                                   |

| Contract             | contract, paperless\_billing, payment\_method                                                      |

| Billing              | monthly\_charges, total\_charges                                                                   |

| Customer behavior    | tenure, support\_tickets, late\_payments                                                           |

| Satisfaction         | satisfaction\_score                                                                               |



\---



## Data Preprocessing



The machine learning pipeline handles preprocessing automatically.



### Numerical Features



Numerical columns use:



\* Median imputation for missing values

\* StandardScaler for feature scaling



### Categorical Features



Categorical columns use:



\* Most-frequent-value imputation

\* One-hot encoding

\* Unknown categories are handled safely



The raw CSV dataset is not manually modified.



\---



## Feature Engineering



Additional features were created to provide the model with useful customer-level signals.



### Average Monthly Spend



Calculated from total charges and tenure.



`avg\\\_monthly\\\_spend = total\\\_charges / tenure`



### Number of Services



Counts the number of subscribed services across:



\* Online security

\* Online backup

\* Device protection

\* Tech support

\* Streaming TV

\* Streaming movies



### Customer Risk Signals



Combines:



\* Support tickets

\* Late payments



### Charge Per Tenure



Calculated using monthly charges and tenure.



### Low Satisfaction



Identifies customers with a satisfaction score of 2 or below.



### Payment Problem



Identifies customers with two or more late payments.



### Short Tenure



Identifies customers with tenure of 12 months or less.



### Long Tenure



Identifies customers with tenure of 36 months or more.



\---



## Model Training



Four machine learning algorithms were evaluated:



1\. Random Forest

2\. Extra Trees

3\. XGBoost

4\. Logistic Regression



Randomized hyperparameter search with 3-fold cross-validation was used for model tuning.



The model-selection metric was cross-validation accuracy.



### Cross-Validation Results



| Model               | Best CV Accuracy |

| ------------------- | ---------------: |

| Random Forest       |           71.63% |

| Extra Trees         |           70.79% |

| XGBoost             |           72.42% |

| Logistic Regression |           71.45% |



The final selected model was \*\*XGBoost\*\* based on the highest cross-validation accuracy.



\---



## Final Model Performance



The final model was evaluated on an untouched test set containing \*\*8,000 customers\*\*.



| Metric    |  Score |

| --------- | -----: |

| Accuracy  | 72.02% |

| Precision | 74.75% |

| Recall    | 82.26% |

| F1 Score  | 78.33% |

| ROC-AUC   | 78.17% |



### Prediction Results



\* Test customers: \*\*8,000\*\*

\* Correct predictions: \*\*5,762\*\*

\* Incorrect predictions: \*\*2,238\*\*



### Confusion Matrix



|                 | Predicted No Churn | Predicted Churn |

| --------------- | -----------------: | --------------: |

| Actual No Churn |              1,718 |           1,366 |

| Actual Churn    |                872 |           4,044 |



\---



## Model Verification



The saved model was independently verified against the same unseen test set.



The verification produced the same results as the training evaluation:



\* Accuracy: \*\*72.02%\*\*

\* Precision: \*\*74.75%\*\*

\* Recall: \*\*82.26%\*\*

\* F1 Score: \*\*78.33%\*\*

\* ROC-AUC: \*\*78.17%\*\*



This confirms that the saved model and feature-engineering pipeline are working consistently.



\---



## FastAPI



The project includes a FastAPI backend for real-time customer churn prediction.



Run the API with:



`uvicorn api.main:app --reload`



The API provides:



\* Customer data input

\* Churn prediction

\* Churn probability

\* Risk classification



Risk levels are calculated as:



\* `HIGH` — probability >= 0.70

\* `MEDIUM` — probability >= 0.40

\* `LOW` — probability < 0.40



FastAPI documentation is available through the `/docs` endpoint when the server is running.



\---



## Streamlit Dashboard



The project also includes a Streamlit dashboard for interactive predictions.



Run:



`streamlit run dashboard/app.py`



The dashboard allows users to enter customer information and receive:



\* Churn prediction

\* Churn probability

\* Risk level



The dashboard applies the same feature-engineering logic used during model training.



\---



## Project Structure



```text

customer-churn-ml-40000/

│

├── api/

│   └── main.py

│

├── dashboard/

│   └── app.py

│

├── data/

│   └── raw/

│       └── customers\\\_40000.csv

│

├── models/

│   ├── churn\\\_model.pkl

│   └── model\\\_metadata.json

│

├── reports/

│






