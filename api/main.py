
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd
import logging

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-End MLOps Customer Churn Prediction System",
    version="1.0.0"
)

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "artifacts" / "models"

PREPROCESSOR_PATH = (
    BASE_DIR / "artifacts" / "preprocessor" / "preprocessor.joblib"
)

# --------------------------------------------------
# Customer input validation
# --------------------------------------------------

class ChurnInput(BaseModel):

    SeniorCitizen: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0, le=100)

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running",
        "status": "success"
    }


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "api": "running"
    }


# --------------------------------------------------
# Model information endpoint
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    models = [
        "logistic_regression",
        "decision_tree",
        "random_forest"
    ]

    available_models = [
        name for name in models
        if (MODEL_DIR / f"{name}.joblib").exists()
    ]

    return {
        "available_models": available_models,
        "selected_model": "random_forest",
        "preprocessor_available": PREPROCESSOR_PATH.exists()
    }


# --------------------------------------------------
# Customer churn prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: ChurnInput):

    try:

        # 1. Define model path
        model_path = MODEL_DIR / "random_forest.joblib"

        # 2. Check model availability
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Random Forest model not found. Train the model first."
            )

        # 3. Check preprocessor availability
        if not PREPROCESSOR_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Preprocessor not found. Run preprocessing first."
            )

        # 4. Load trained model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        # 5. Convert customer input into DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # 6. Apply the same preprocessing used during training
        processed_input = preprocessor.transform(input_df)

        # 7. Generate prediction
        prediction = int(model.predict(processed_input)[0])

        # 8. Calculate churn probability
        probability = float(
            model.predict_proba(processed_input)[0][1]
        )

        # 9. Prepare response
        result = {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(probability, 4),
            "churn_percentage": round(probability * 100, 2),
            "model": "random_forest"
        }

        logger.info("Customer churn prediction completed successfully")

        return result

    except HTTPException:
        raise

    except Exception as error:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
