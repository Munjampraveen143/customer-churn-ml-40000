from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    div.stButton > button[kind="primaryFormSubmit"] {
        font-weight: bold;
        min-height: 48px;
    }

    div.stButton > button {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Analyze customer information, estimate churn probability,
    and identify customers who may be at risk of leaving.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("📌 About the Project")

    st.write(
        """
        This application uses a trained machine learning model
        to estimate customer churn.

        **Available features**
        - Customer profile
        - Subscription information
        - Billing information
        - Customer risk indicators
        - Churn probability
        """
    )

    st.info(
        "Predictions are estimates from a machine learning model "
        "and should be interpreted alongside other customer information."
    )


# =====================================================
# MODEL LOADING
# =====================================================

# Repository structure:
#
# customer-churn-ml-40000/
# ├── api/
# │   ├── main.py
# │   └── streamlit_app.py
# └── models/
#     └── churn_model.pkl

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "churn_model.pkl"
)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.is_file():
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Check that models/churn_model.pkl exists in your repository."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))

except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.info(
        "Check requirements.txt and ensure the libraries used to train "
        "the model are installed in Streamlit Cloud."
    )
    st.stop()


# =====================================================
# FORM RESET
# =====================================================

DEFAULTS = {
    "gender": "Female",
    "age": 35,
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.0,
    "total_charges": 840.0,
    "support_tickets": 1,
    "late_payments": 0,
    "satisfaction_score": 3,
}


def reset_form():
    """Reset form fields and clear the previous prediction."""
    for key in DEFAULTS:
        st.session_state.pop(key, None)

    st.session_state.pop("prediction_result", None)


# =====================================================
# CUSTOMER INPUT FORM
# =====================================================

st.subheader("👤 Customer Information")

# Clear button is outside the form so its callback can reset
# widget values before the next run renders the form.

_, clear_column = st.columns([4, 1])

with clear_column:
    st.button(
        "🔄 Clear Form",
        on_click=reset_form,
        use_container_width=True,
    )


with st.form("customer_form", clear_on_submit=False):

    # -------------------------------------------------
    # SECTION 1: CUSTOMER PROFILE
    # -------------------------------------------------

    st.markdown("### 👤 Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.selectbox(
            "Gender",
            ["Female", "Male"],
            key="gender",
        )

        st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            key="age",
        )

        st.selectbox(
            "Senior Citizen",
            ["No", "Yes"],
            key="senior_citizen",
        )

    with col2:
        st.selectbox(
            "Has Partner?",
            ["Yes", "No"],
            key="partner",
        )

        st.selectbox(
            "Has Dependents?",
            ["Yes", "No"],
            key="dependents",
        )

        st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=120,
            key="tenure",
        )

    with col3:
        st.selectbox(
            "Phone Service",
            ["Yes", "No"],
            key="phone_service",
        )

        st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            key="internet_service",
        )

    st.divider()

    # -------------------------------------------------
    # SECTION 2: SUBSCRIPTION SERVICES
    # -------------------------------------------------

    st.markdown("### 📡 Subscription Services")

    col4, col5, col6 = st.columns(3)

    service_options = [
        "Yes",
        "No",
        "No internet service",
    ]

    with col4:
        st.selectbox(
            "Online Security",
            service_options,
            key="online_security",
        )

        st.selectbox(
            "Online Backup",
            service_options,
            key="online_backup",
        )

    with col5:
        st.selectbox(
            "Device Protection",
            service_options,
            key="device_protection",
        )

        st.selectbox(
            "Tech Support",
            service_options,
            key="tech_support",
        )

    with col6:
        st.selectbox(
            "Streaming TV",
            service_options,
            key="streaming_tv",
        )

        st.selectbox(
            "Streaming Movies",
            service_options,
            key="streaming_movies",
        )

    st.divider()

    # -------------------------------------------------
    # SECTION 3: BILLING INFORMATION
    # -------------------------------------------------

    st.markdown("### 💳 Billing Information")

    col7, col8, col9 = st.columns(3)

    with col7:
        st.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
            key="contract",
        )

    with col8:
        st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            key="paperless_billing",
        )

    with col9:
        st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            key="payment_method",
        )

    col10, col11 = st.columns(2)

    with col10:
        st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000000.0,
            step=1.0,
            key="monthly_charges",
        )

    with col11:
        st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000000.0,
            step=10.0,
            key="total_charges",
        )

    st.divider()

    # -------------------------------------------------
    # SECTION 4: BILLING AND CUSTOMER RISK
    # -------------------------------------------------

    st.markdown("### 📈 Billing and Customer Risk")

    col12, col13, col14 = st.columns(3)

    with col12:
        st.number_input(
            "Support Tickets",
            min_value=0,
            max_value=100000,
            step=1,
            key="support_tickets",
        )

    with col13:
        st.number_input(
            "Late Payments",
            min_value=0,
            max_value=100000,
            step=1,
            key="late_payments",
        )

    with col14:
        st.select_slider(
            "Satisfaction Score",
            options=[1, 2, 3, 4, 5],
            key="satisfaction_score",
        )

    st.divider()

    # -------------------------------------------------
    # PREDICTION BUTTON
    # -------------------------------------------------

    submitted = st.form_submit_button(
        "🔍  PREDICT CHURN",
        type="primary",
        use_container_width=True,
    )


# =====================================================
# PREDICTION LOGIC
# =====================================================

if submitted:

    customer = {
        "gender": st.session_state["gender"],
        "age": int(st.session_state["age"]),
        "senior_citizen": (
            1 if st.session_state["senior_citizen"] == "Yes" else 0
        ),
        "partner": st.session_state["partner"],
        "dependents": st.session_state["dependents"],
        "tenure": int(st.session_state["tenure"]),
        "phone_service": st.session_state["phone_service"],
        "internet_service": st.session_state["internet_service"],
        "online_security": st.session_state["online_security"],
        "online_backup": st.session_state["online_backup"],
        "device_protection": st.session_state["device_protection"],
        "tech_support": st.session_state["tech_support"],
        "streaming_tv": st.session_state["streaming_tv"],
        "streaming_movies": st.session_state["streaming_movies"],
        "contract": st.session_state["contract"],
        "paperless_billing": st.session_state["paperless_billing"],
        "payment_method": st.session_state["payment_method"],
        "monthly_charges": float(
            st.session_state["monthly_charges"]
        ),
        "total_charges": float(
            st.session_state["total_charges"]
        ),
        "support_tickets": int(
            st.session_state["support_tickets"]
        ),
        "late_payments": int(
            st.session_state["late_payments"]
        ),
        "satisfaction_score": int(
            st.session_state["satisfaction_score"]
        ),
    }

    df = pd.DataFrame([customer])

    # -------------------------------------------------
    # ENGINEERED FEATURES
    # -------------------------------------------------

    tenure_denominator = df["tenure"].replace(0, np.nan)

    df["avg_monthly_spend"] = (
        df["total_charges"] / tenure_denominator
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

    # Handle the zero-tenure case to prevent an infinite or NaN
    # feature from reaching a model that cannot accept missing values.
    df["avg_monthly_spend"] = (
        df["avg_monthly_spend"].replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    if df["avg_monthly_spend"].isna().any():
        df["avg_monthly_spend"] = (
            df["avg_monthly_spend"].fillna(
                df["monthly_charges"]
            )
        )

    # -------------------------------------------------
    # RUN MODEL
    # -------------------------------------------------

    try:
        with st.spinner("Analyzing customer information..."):

            prediction = int(model.predict(df)[0])

            if not hasattr(model, "predict_proba"):
                st.error(
                    "This model does not support probability predictions."
                )
                st.stop()

            probabilities = model.predict_proba(df)[0]

            # Assumes the model's positive class (1) means churn.
            classes = list(model.classes_)
            positive_class_index = classes.index(1)

            probability = float(
                probabilities[positive_class_index]
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

            st.session_state["prediction_result"] = {
                "prediction": prediction,
                "label": label,
                "probability": probability,
                "risk": risk,
            }

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

        st.warning(
            "The model may expect different feature names, feature "
            "types, or preprocessing than this form supplies. "
            "The input features must match the training pipeline."
        )

        if hasattr(model, "feature_names_in_"):
            st.write(
                "Model feature names:",
                list(model.feature_names_in_),
            )

        st.stop()


# =====================================================
# DISPLAY RESULTS
# =====================================================

result = st.session_state.get("prediction_result")

if result is not None:

    st.divider()

    st.markdown("## 🎯 Prediction Results")

    result1, result2, result3 = st.columns(3)

    with result1:
        if result["prediction"] == 1:
            st.error("⚠️ Predicted Outcome: Churn")
        else:
            st.success("✅ Predicted Outcome: No Churn")

    with result2:
        st.metric(
            "Churn Probability",
            f'{result["probability"]:.1%}',
        )

    with result3:
        if result["risk"] == "HIGH":
            st.error("🔴 Risk Level: HIGH")
        elif result["risk"] == "MEDIUM":
            st.warning("🟠 Risk Level: MEDIUM")
        else:
            st.success("🟢 Risk Level: LOW")

    st.markdown("### Probability Indicator")

    st.progress(
        max(
            0.0,
            min(1.0, result["probability"]),
        )
    )

    st.caption(
        f'Estimated churn probability: '
        f'{result["probability"]:.1%}'
    )

    if result["risk"] == "HIGH":
        st.warning(
            "This customer has a high estimated churn probability. "
            "Consider reviewing their service experience and billing concerns."
        )

    elif result["risk"] == "MEDIUM":
        st.info(
            "This customer has a moderate estimated churn probability. "
            "Monitoring their engagement may be useful."
        )

    else:
        st.success(
            "This customer has a low estimated churn probability "
            "according to the current model."
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Dashboard"
)
