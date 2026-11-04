
import os
import yaml
import joblib
import mlflow
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_params():
    """Load model configuration from params.yaml."""
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def train_models():
    """Train, evaluate and save all classification models."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer Churn Prediction")

    # Load configuration
    params = load_params()

    # Load processed datasets
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # Separate features and target
    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    # Define models
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=params["models"]["logistic_regression"]["max_iter"],
            class_weight=params["models"]["logistic_regression"]["class_weight"]
        ),

        "decision_tree": DecisionTreeClassifier(
            random_state=params["models"]["decision_tree"]["random_state"],
            class_weight=params["models"]["decision_tree"]["class_weight"]
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=params["models"]["random_forest"]["n_estimators"],
            random_state=params["models"]["random_forest"]["random_state"],
            class_weight=params["models"]["random_forest"]["class_weight"]
        )
    }

    # Create model output directory
    os.makedirs("artifacts/models", exist_ok=True)

    # Set MLflow experiment
    mlflow.set_experiment("Customer_Churn_Prediction")

    # Train and evaluate each model
    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            print(f"\nTraining {model_name}...")

            # Train model
            model.fit(X_train, y_train)

            # Generate predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            # Calculate evaluation metrics
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "roc_auc": roc_auc_score(y_test, probabilities)
            }

            # Log model parameters
            mlflow.log_params(model.get_params())

            # Log evaluation metrics
            mlflow.log_metrics(metrics)

            # Save model in MLflow using Pickle serialization
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                serialization_format="pickle"
            )

            # Save model locally
            model_path = f"artifacts/models/{model_name}.joblib"
            joblib.dump(model, model_path)

            # Display results
            print(f"\nModel: {model_name}")

            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            print(f"Model saved at: {model_path}")

    print("\nAll models trained and logged successfully!")


if __name__ == "__main__":
    train_models()
