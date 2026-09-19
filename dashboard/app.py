import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"


# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer details below. The ML model will predict whether "
    "the customer is likely to churn."
)

st.info(
    "The actual churn value is NOT entered. The model must predict it."
)


# --------------------------------------------------
# Customer information
# --------------------------------------------------
st.header("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    customer_id = st.text_input(
        "Customer ID",
        value="TEST_CHURN_001"
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=28,
        step=1
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=3,
        step=1
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No"]
    )


with col3:

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=1000.0,
        value=92.50,
        step=0.01
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=100000.0,
        value=277.50,
        step=0.01
    )

    support_tickets = st.number_input(
        "Support Tickets",
        min_value=0,
        max_value=100,
        value=4,
        step=1
    )

    late_payments = st.number_input(
        "Late Payments",
        min_value=0,
        max_value=100,
        value=3,
        step=1
    )

    satisfaction_score = st.number_input(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )


# --------------------------------------------------
# Prediction button
# --------------------------------------------------
st.divider()

predict_button = st.button(
    "🔮 Predict Churn",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if predict_button:

    # --------------------------------------------------
    # Original model features
    # --------------------------------------------------
    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
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
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "satisfaction_score": satisfaction_score
    }])


    # --------------------------------------------------
    # Feature engineering
    # Must match src/train.py exactly
    # --------------------------------------------------

    input_data["avg_monthly_spend"] = (
        input_data["total_charges"]
        / input_data["tenure"].replace(0, np.nan)
    )

    service_columns = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    input_data["num_services"] = (
        input_data[service_columns]
        .eq("Yes")
        .sum(axis=1)
    )

    input_data["customer_risk_signals"] = (
        input_data["support_tickets"]
        + input_data["late_payments"]
    )

    input_data["charge_per_tenure"] = (
        input_data["monthly_charges"]
        / (input_data["tenure"] + 1)
    )

    input_data["low_satisfaction"] = (
        input_data["satisfaction_score"] <= 2
    ).astype(int)

    input_data["payment_problem"] = (
        input_data["late_payments"] >= 2
    ).astype(int)

    input_data["short_tenure"] = (
        input_data["tenure"] <= 12
    ).astype(int)

    input_data["long_tenure"] = (
        input_data["tenure"] >= 36
    ).astype(int)


    # --------------------------------------------------
    # Model prediction
    # --------------------------------------------------

    prediction = int(
        model.predict(input_data)[0]
    )


    # --------------------------------------------------
    # Probability of churn
    # --------------------------------------------------

    if hasattr(model, "predict_proba"):

        churn_probability = float(
            model.predict_proba(input_data)[0][1]
        )

    else:

        churn_probability = None


    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    st.header("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        if prediction == 1:

            st.error("🔴 CHURN")

        else:

            st.success("🟢 NO CHURN")


    with result_col2:

        if churn_probability is not None:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        else:

            st.metric(
                "Churn Probability",
                "N/A"
            )


    with result_col3:

        if churn_probability is not None:

            if churn_probability >= 0.70:

                risk = "HIGH"

            elif churn_probability >= 0.40:

                risk = "MEDIUM"

            else:

                risk = "LOW"

            st.metric(
                "Risk Level",
                risk
            )


    # --------------------------------------------------
    # Prediction explanation
    # --------------------------------------------------

    st.subheader("Prediction Details")

    st.write(
        f"Customer ID: **{customer_id}**"
    )

    if prediction == 1:

        st.write(
            "The ML model predicts that this customer is likely to churn."
        )

    else:

        st.write(
            "The ML model predicts that this customer is likely to remain."
        )


    # --------------------------------------------------
    # Show input data
    # --------------------------------------------------

    with st.expander("View data sent to the ML model"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


    st.caption(
        "Note: This is a genuine ML prediction. The actual `churn` "
        "value was not provided to the model."
    )