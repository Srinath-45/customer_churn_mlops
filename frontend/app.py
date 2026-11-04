
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ==================================================
# CONFIGURATION
# ==================================================

API_URL = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM STYLING
# ==================================================

st.markdown("""
<style>
.stApp {
    background: #F4F7FC;
    color: #172033;
}

[data-testid="stHeader"] {
    background: #F4F7FC;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #14213D;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

h1, h2, h3, h4, p, label,
[data-testid="stMarkdownContainer"] {
    color: #172033;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: #FFFFFF;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E1E8F0;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.035);
}

div[data-testid="stForm"] {
    background: white;
    padding: 24px;
    border: 1px solid #E1E8F0;
    border-radius: 16px;
}

div[data-testid="stForm"] label {
    color: #172033 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #2457D6;
    color: white !important;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #173FA8;
    color: white !important;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE
# ==================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ==================================================
# API CONNECTION
# ==================================================

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


# ==================================================
# RISK CLASSIFICATION
# ==================================================

def classify_risk(probability):
    if probability >= 0.70:
        return "HIGH", "🔴", "Prioritize customer retention"
    elif probability >= 0.40:
        return "MEDIUM", "🟠", "Monitor customer engagement"
    else:
        return "LOW", "🟢", "Maintain customer satisfaction"


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("📊 Churn IQ")
    st.caption("Customer Analytics Platform")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select Page",
        [
            "Dashboard",
            "Predict Churn",
            "Prediction History",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("System Status")

    api_connected = check_api()

    if api_connected:
        st.success("API Connected")
    else:
        st.error("API Disconnected")

    st.caption("FastAPI Backend")
    st.caption(API_URL)

    st.divider()

    st.caption("End-to-End MLOps Project")
    st.caption("Customer Churn Prediction")


# ==================================================
# DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.title("Customer Churn Dashboard")
    st.caption(
        "Monitor customer behavior and predict telecom customer churn."
    )

    st.divider()

    # Summary metrics
    total_predictions = len(
        st.session_state.prediction_history
    )

    high_risk_count = sum(
        1 for item in st.session_state.prediction_history
        if item["risk"] == "HIGH"
    )

    average_probability = (
        sum(
            item["probability"]
            for item in st.session_state.prediction_history
        ) / total_predictions
        if total_predictions else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predictions",
        total_predictions
    )

    col2.metric(
        "High-Risk Customers",
        high_risk_count
    )

    col3.metric(
        "Average Churn Probability",
        f"{average_probability * 100:.1f}%"
    )

    st.divider()

    st.subheader("Platform Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "### 🤖 Machine Learning\n"
            "The application uses trained classification "
            "models to estimate customer churn."
        )

    with col2:
        st.info(
            "### ⚙️ MLOps Pipeline\n"
            "Model development and tracking use Python, "
            "MLflow, DVC, FastAPI and Docker."
        )

    st.subheader("Get Started")

    st.write(
        "Navigate to **Predict Churn** in the sidebar "
        "to enter customer information and generate a prediction."
    )


# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "Predict Churn":

    st.title("Customer Churn Prediction")

    st.caption(
        "Enter the customer's details to estimate their churn probability."
    )

    st.divider()

    with st.form("customer_form"):

        # Personal information
        st.subheader("👤 Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

        with col2:
            SeniorCitizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

        with col3:
            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=100,
                value=12
            )

        col1, col2 = st.columns(2)

        with col1:
            Partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

        with col2:
            Dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

        st.divider()

        # Services
        st.subheader("📡 Internet and Phone Services")

        col1, col2, col3 = st.columns(3)

        with col1:
            PhoneService = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            MultipleLines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            InternetService = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

        with col2:
            OnlineSecurity = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            OnlineBackup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            DeviceProtection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

        with col3:
            TechSupport = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            StreamingTV = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            StreamingMovies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        st.divider()

        # Contract and billing
        st.subheader("💳 Contract and Billing")

        col1, col2, col3 = st.columns(3)

        with col1:
            Contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            PaperlessBilling = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

        with col2:
            PaymentMethod = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col3:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                value=70.0,
                step=1.0
            )

            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                value=840.0,
                step=10.0
            )

        st.divider()

        submitted = st.form_submit_button(
            "🚀 Generate Churn Prediction",
            use_container_width=True
        )

    # Send prediction request
    if submitted:

        customer_data = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        st.subheader("Prediction Results")

        if not check_api():
            st.error("FastAPI is not running. Start the backend first.")

        else:
            try:
                with st.spinner("Analyzing customer data..."):

                    response = requests.post(
                        f"{API_URL}/predict",
                        json=customer_data,
                        timeout=30
                    )

                if response.status_code == 200:

                    result = response.json()

                    if "error" in result:
                        st.error(result["error"])

                    else:
                        probability = float(
                            result["churn_probability"]
                        )

                        prediction = result["prediction"]

                        risk, emoji, recommendation = classify_risk(
                            probability
                        )

                        st.divider()

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Prediction",
                            str(prediction)
                        )

                        col2.metric(
                            "Churn Probability",
                            f"{probability * 100:.2f}%"
                        )

                        col3.metric(
                            "Risk Category",
                            risk
                        )

                        st.subheader("Churn Probability")

                        st.progress(
                            max(0.0, min(1.0, probability))
                        )

                        st.write(
                            f"**{probability * 100:.2f}% probability of churn**"
                        )

                        if risk == "HIGH":
                            st.error(
                                f"{emoji} HIGH RISK — {recommendation}"
                            )

                        elif risk == "MEDIUM":
                            st.warning(
                                f"{emoji} MEDIUM RISK — {recommendation}"
                            )

                        else:
                            st.success(
                                f"{emoji} LOW RISK — {recommendation}"
                            )

                        st.subheader("Retention Recommendation")

                        if risk == "HIGH":
                            st.write(
                                "- Contact the customer proactively.\n"
                                "- Review contract and pricing concerns.\n"
                                "- Offer suitable retention benefits."
                            )

                        elif risk == "MEDIUM":
                            st.write(
                                "- Monitor customer satisfaction.\n"
                                "- Offer relevant service improvements.\n"
                                "- Review customer engagement."
                            )

                        else:
                            st.write(
                                "- Maintain current service quality.\n"
                                "- Continue regular customer engagement."
                            )

                        # Store session history
                        st.session_state.prediction_history.append({
                            "time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "prediction": str(prediction),
                            "probability": probability,
                            "risk": risk
                        })

                else:
                    st.error(
                        f"API error: {response.status_code}"
                    )
                    st.code(response.text)

            except requests.RequestException as error:
                st.error(f"Connection error: {error}")

            except (KeyError, TypeError, ValueError) as error:
                st.error(f"Unexpected API response: {error}")


# ==================================================
# HISTORY PAGE
# ==================================================

elif page == "Prediction History":

    st.title("Prediction History")

    st.caption(
        "Predictions generated during the current application session."
    )

    history = st.session_state.prediction_history

    if history:

        df = pd.DataFrame(history)

        df["probability"] = (
            df["probability"] * 100
        ).round(2)

        df = df.rename(columns={
            "time": "Prediction Time",
            "prediction": "Prediction",
            "probability": "Churn Probability (%)",
            "risk": "Risk Level"
        })

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()

    else:
        st.info(
            "No predictions yet. Generate a prediction to see it here."
        )


# ==================================================
# ABOUT PAGE
# ==================================================

elif page == "About Project":

    st.title("About Customer Churn Prediction")

    st.write(
        "An end-to-end machine learning and MLOps application "
        "designed to estimate the likelihood of telecom customer churn."
    )

    st.subheader("Technology Stack")

    technologies = {
        "Machine Learning": "Scikit-learn",
        "Experiment Tracking": "MLflow",
        "Data Version Control": "DVC",
        "Backend API": "FastAPI",
        "Frontend": "Streamlit",
        "Containerization": "Docker",
        "Testing": "Pytest",
        "CI/CD": "GitHub Actions"
    }

    for category, tool in technologies.items():
        st.write(f"**{category}:** {tool}")

    st.subheader("Machine Learning Models")

    st.write(
        "- Logistic Regression\n"
        "- Decision Tree\n"
        "- Random Forest"
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Churn Intelligence | End-to-End MLOps Customer Churn Prediction"
)
