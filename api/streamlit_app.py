from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details below to estimate churn probability "
    "and identify customers who may be at risk of leaving."
)


# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = Path(__file__).parent / "models" / "churn_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Please check that "
        "'models/churn_model.pkl' exists in your GitHub repository."
    )
    st.stop()
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


# -------------------------------
# Customer input form
# -------------------------------
st.subheader("Customer Information")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", 18, 120, 35)
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", 0, 120, 12)

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    st.subheader("Billing and Customer Risk")

    col4, col5, col6 = st.columns(3)

    with col4:
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, value=840.0, step=10.0
        )

    with col5:
        support_tickets = st.number_input(
            "Support Tickets", min_value=0, value=1, step=1
        )
        late_payments = st.number_input(
            "Late Payments", min_value=0, value=0, step=1
        )

    with col6:
        satisfaction_score = st.slider(
            "Satisfaction Score", 1, 5, 3
        )

    submitted = st.form_submit_button(
        "🔍 Predict Churn",
        use_container_width=True,
    )


# -------------------------------
# Prepare features and predict
# -------------------------------
if submitted:

    customer = {
        "gender": gender,
        "age": int(age),
        "senior_citizen": int(senior_citizen),
        "partner": partner,
        "dependents": dependents,
        "tenure": int(tenure),
        "phone_service": phone_service,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": float(monthly_charges),
        "total_charges": float(total_charges),
        "support_tickets": int(support_tickets),
        "late_payments": int(late_payments),
        "satisfaction_score": int(satisfaction_score),
    }

    df = pd.DataFrame([customer])

    # Engineered features matching the supplied FastAPI code
    df["avg_monthly_spend"] = (
        df["total_charges"]
        / df["tenure"].replace(0, np.nan)
    )

    df["num_services"] = (
        df[
            [
                "online_security",
                "online_backup",
                "device_protection",
                "tech_support",
                "streaming_tv",
                "streaming_movies",
            ]
        ]
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

    try:
        prediction = int(model.predict(df)[0])

        if not hasattr(model, "predict_proba"):
            st.error(
                "This model does not support probability predictions. "
                "The churn label cannot be accompanied by a probability."
            )
            st.stop()

        probability = float(model.predict_proba(df)[0][1])

        st.divider()
        st.subheader("Prediction Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            if prediction == 1:
                st.error("⚠️ Predicted Outcome: Churn")
            else:
                st.success("✅ Predicted Outcome: No Churn")

        with result_col2:
            st.metric(
                "Churn Probability",
                f"{probability:.1%}",
            )

        with result_col3:
            if probability >= 0.70:
                st.error("Risk Level: HIGH")
            elif probability >= 0.40:
                st.warning("Risk Level: MEDIUM")
            else:
                st.success("Risk Level: LOW")

        st.progress(max(0.0, min(1.0, probability)))

        st.caption(
            "This result is a model estimate, not a guarantee "
            "that a customer will leave."
        )

    except Exception as exc:
        st.error(
            f"Prediction failed: {exc}"
        )
        st.info(
            "The model may expect a different set or order of features "
            "than the supplied input data. Check the training code "
            "and model's expected feature names."
        )
