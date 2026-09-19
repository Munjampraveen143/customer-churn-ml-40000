import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


MODEL_PATH = "models/churn_model.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Customer churn prediction using the trained XGBoost model.",
    version="2.0.0",
)


class CustomerData(BaseModel):
    gender: str
    age: int
    senior_citizen: int
    partner: str
    dependents: str
    tenure: int
    phone_service: str
    internet_service: str
    online_security: Optional[str] = None
    online_backup: Optional[str] = None
    device_protection: Optional[str] = None
    tech_support: Optional[str] = None
    streaming_tv: Optional[str] = None
    streaming_movies: Optional[str] = None
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: Optional[float] = None
    support_tickets: int
    late_payments: int
    satisfaction_score: int


def create_features(data: CustomerData) -> pd.DataFrame:
    row = data.model_dump()

    df = pd.DataFrame([row])

    df["avg_monthly_spend"] = (
        df["total_charges"]
        / df["tenure"].replace(0, np.nan)
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
        df["support_tickets"]
        + df["late_payments"]
    )

    df["charge_per_tenure"] = (
        df["monthly_charges"]
        / (df["tenure"] + 1)
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

    return df


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "status": "ok",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Customer Churn Prediction API is running",
    }


@app.post("/predict")
def predict(data: CustomerData):

    df = create_features(data)

    prediction = int(model.predict(df)[0])

    probability = float(
        model.predict_proba(df)[0][1]
    )

    if prediction == 1:
        label = "Churn"
    else:
        label = "No Churn"

    if probability >= 0.70:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "prediction": prediction,
        "label": label,
        "churn_probability": round(probability, 4),
        "risk": risk,
    }