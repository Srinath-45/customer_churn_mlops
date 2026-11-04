import os
import joblib
import pandas as pd


PREPROCESSOR_PATH = "artifacts/preprocessor/preprocessor.joblib"
MODEL_DIR = "artifacts/models"

DEFAULT_MODEL = "random_forest"


def load_artifacts(model_name=DEFAULT_MODEL):
    """Load the saved preprocessor and selected ML model."""

    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.joblib"
    )

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(model_path)

    return preprocessor, model


def predict_customer(customer_data, model_name=DEFAULT_MODEL):
    """
    Predict churn for one customer.

    customer_data must contain the original customer
    feature columns used during training.
    """

    # Load saved artifacts
    preprocessor, model = load_artifacts(model_name)

    # Convert customer information into a DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Remove customer ID if supplied
    if "customerID" in customer_df.columns:
        customer_df = customer_df.drop(columns=["customerID"])

    # Get the original input columns expected by the preprocessor
    expected_columns = list(preprocessor.feature_names_in_)

    # Check for missing input fields
    missing_columns = [
        column
        for column in expected_columns
        if column not in customer_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing customer information: {missing_columns}"
        )

    # Keep the same column order as training
    customer_df = customer_df[expected_columns]

    # Convert TotalCharges to numeric
    if "TotalCharges" in customer_df.columns:
        customer_df["TotalCharges"] = pd.to_numeric(
            customer_df["TotalCharges"],
            errors="coerce"
        )

    # Apply the SAME preprocessor used during training
    processed_customer = preprocessor.transform(customer_df)

    # Generate prediction
    prediction = int(model.predict(processed_customer)[0])

    # Generate churn probability
    probabilities = model.predict_proba(processed_customer)[0]

    # Find the probability corresponding to class 1 (Churn)
    churn_class_index = list(model.classes_).index(1)

    churn_probability = float(
        probabilities[churn_class_index]
    )

    # Convert numerical prediction into readable result
    prediction_label = (
        "Churn" if prediction == 1 else "No Churn"
    )

    return {
        "prediction": prediction_label,
        "prediction_value": prediction,
        "churn_probability": round(churn_probability, 4),
        "model_used": model_name
    }


if __name__ == "__main__":

    # Example customer for testing
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }

    result = predict_customer(sample_customer)

    print("\nCustomer Churn Prediction")
    print("------------------------")
    print("Prediction:", result["prediction"])
    print(
        "Churn Probability:",
        f"{result['churn_probability'] * 100:.2f}%"
    )
    print("Model:", result["model_used"])