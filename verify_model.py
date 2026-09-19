import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/raw/customers_40000.csv"
MODEL_PATH = "models/churn_model.pkl"


print("=" * 60)
print("CUSTOMER CHURN MODEL VERIFICATION")
print("=" * 60)

# Load data
df = pd.read_csv(DATA_PATH)

print(f"Dataset rows: {len(df)}")
print(f"Dataset columns: {len(df.columns)}")

# ---------------------------------------------------------
# SAME FEATURE ENGINEERING USED DURING TRAINING
# ---------------------------------------------------------

df["avg_monthly_spend"] = (
    df["total_charges"] /
    df["tenure"].replace(0, np.nan)
)

service_columns = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

df["num_services"] = (
    df[service_columns]
    .eq("Yes")
    .sum(axis=1)
)

df["customer_risk_signals"] = (
    df["support_tickets"] +
    df["late_payments"]
)

df["charge_per_tenure"] = (
    df["monthly_charges"] /
    (df["tenure"] + 1)
)

df["low_satisfaction"] = (
    df["satisfaction_score"] <= 2
).astype(int)

df["payment_problem"] = (
    df["late_payments"] >= 2
).astype(int)

df["short_tenure"] = (
    df["tenure"] <= 12
).astype(int)

df["long_tenure"] = (
    df["tenure"] >= 36
).astype(int)

# ---------------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------------

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print()
print(f"Training customers: {len(X_train)}")
print(f"Test customers:     {len(X_test)}")

# ---------------------------------------------------------
# LOAD SAVED MODEL
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)

print()
print("Running predictions on unseen test data...")

y_pred = model.predict(X_test)

# Probability for ROC-AUC
if hasattr(model, "predict_proba"):
    y_probability = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probability)
else:
    roc_auc = None

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print()
print("=" * 60)
print("VERIFICATION RESULTS")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

if roc_auc is not None:
    print(f"ROC-AUC:   {roc_auc:.4f}")

print()
print("Classification Report")
print("-" * 60)
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print("-" * 60)
print(confusion_matrix(y_test, y_pred))

correct = int((y_pred == y_test).sum())
wrong = int((y_pred != y_test).sum())

print()
print(f"Test rows: {len(y_test)}")
print(f"Correct:   {correct}")
print(f"Wrong:     {wrong}")

print()
print("=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)