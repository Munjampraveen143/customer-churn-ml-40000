import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/customers_40000.csv"
MODEL_PATH = "models/churn_model.pkl"
RESULTS_PATH = "reports/upgraded_model_results.json"
METADATA_PATH = "models/model_metadata.json"

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("=" * 70)
print("CUSTOMER CHURN - FAST LEGITIMATE MODEL TRAINING")
print("=" * 70)

# -------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

# -------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------------------------------------------

df["avg_monthly_spend"] = df["total_charges"] / df["tenure"].replace(0, np.nan)

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
    df["support_tickets"] + df["late_payments"]
)

df["charge_per_tenure"] = (
    df["monthly_charges"] / (df["tenure"] + 1)
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

# -------------------------------------------------------------------
# 3. FEATURES / TARGET
# -------------------------------------------------------------------

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training rows: {len(X_train)}")
print(f"Test rows:     {len(X_test)}")

# -------------------------------------------------------------------
# 4. PREPROCESSING
# -------------------------------------------------------------------

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "string", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ]
)

# -------------------------------------------------------------------
# 5. MODELS
# -------------------------------------------------------------------

models = {
    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        ),
        {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"],
        },
    ),

    "Extra Trees": (
        ExtraTreesClassifier(
            random_state=42,
            n_jobs=-1,
        ),
        {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"],
        },
    ),

    "XGBoost": (
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        {
            "model__n_estimators": [150, 250],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
    ),

    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        {
            "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
        },
    ),
}

# -------------------------------------------------------------------
# 6. FAST RANDOMIZED SEARCH
# -------------------------------------------------------------------

results = []
best_model = None
best_name = None
best_cv_accuracy = -1

for name, (model, param_grid) in models.items():

    print("\n" + "=" * 70)
    print(f"TUNING: {name}")
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=8,
        scoring="accuracy",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)

    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    if search.best_score_ > best_cv_accuracy:
        best_cv_accuracy = search.best_score_
        best_model = search.best_estimator_
        best_name = name

# -------------------------------------------------------------------
# 7. FINAL TEST EVALUATION
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL")
print("=" * 70)

print(f"Selected model: {best_name}")
print(f"Best CV accuracy: {best_cv_accuracy:.4f}")

y_pred = best_model.predict(X_test)

if hasattr(best_model, "predict_proba"):
    y_probability = best_model.predict_proba(X_test)[:, 1]
else:
    y_probability = None

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

if y_probability is not None:
    roc_auc = roc_auc_score(y_test, y_probability)
else:
    roc_auc = None

cm = confusion_matrix(y_test, y_pred)

print("\nTest Results")
print("-" * 40)
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

if roc_auc is not None:
    print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(cm)

# -------------------------------------------------------------------
# 8. SAVE MODEL
# -------------------------------------------------------------------

joblib.dump(best_model, MODEL_PATH)

results_data = {
    "selected_model": best_name,
    "cv_accuracy": float(best_cv_accuracy),
    "test_accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc) if roc_auc is not None else None,
    "test_rows": int(len(X_test)),
    "correct_predictions": int((y_pred == y_test).sum()),
    "wrong_predictions": int((y_pred != y_test).sum()),
    "confusion_matrix": cm.tolist(),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results_data, f, indent=4)

metadata = {
    "model": best_name,
    "feature_engineering": [
        "avg_monthly_spend",
        "num_services",
        "customer_risk_signals",
        "charge_per_tenure",
        "low_satisfaction",
        "payment_problem",
        "short_tenure",
        "long_tenure",
    ],
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "random_state": 42,
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Model saved to: {MODEL_PATH}")
print(f"Results saved to: {RESULTS_PATH}")
print(f"Metadata saved to: {METADATA_PATH}")