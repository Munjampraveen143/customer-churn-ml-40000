import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.pkl"

def predict_customer(customer: dict):
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([customer])
    probability = float(model.predict_proba(df)[0, 1])
    prediction = int(probability >= 0.5)
    risk = "HIGH" if probability >= 0.70 else "MEDIUM" if probability >= 0.40 else "LOW"
    return {
        "prediction": prediction,
        "prediction_label": "Churn" if prediction else "No Churn",
        "churn_probability": round(probability, 4),
        "risk": risk
    }
