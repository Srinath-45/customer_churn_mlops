import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "artifacts/models"
REPORT_DIR = "reports"

MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest"
]


def evaluate_models():
    """Evaluate all saved models using the test dataset."""

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Processed test dataset not found.")

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    os.makedirs(REPORT_DIR, exist_ok=True)

    results = []

    for model_name in MODEL_NAMES:

        model_path = os.path.join(
            MODEL_DIR,
            f"{model_name}.joblib"
        )

        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: model file not found.")
            continue

        print(f"\nEvaluating {model_name}...")

        model = joblib.load(model_path)

        # Generate predictions
        predictions = model.predict(X_test)

        # Generate probabilities
        probabilities = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "model": model_name,
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

        results.append(metrics)

        # Classification report
        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        )

        report_path = os.path.join(
            REPORT_DIR,
            f"{model_name}_classification_report.json"
        )

        with open(report_path, "w") as file:
            json.dump(report, file, indent=4)

        # Confusion matrix
        matrix = confusion_matrix(y_test, predictions)

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual_No_Churn", "Actual_Churn"],
            columns=["Predicted_No_Churn", "Predicted_Churn"]
        )

        matrix_df.to_csv(
            os.path.join(
                REPORT_DIR,
                f"{model_name}_confusion_matrix.csv"
            )
        )

        # Display metrics
        print(f"\nResults for {model_name}")

        for metric_name, value in metrics.items():
            if metric_name != "model":
                print(f"{metric_name}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(matrix)

    if not results:
        raise FileNotFoundError(
            "No trained model files were found in artifacts/models."
        )

    # Save comparison of all models
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="roc_auc",
        ascending=False
    )

    results_df.to_csv(
        os.path.join(REPORT_DIR, "model_comparison.csv"),
        index=False
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print("\nEvaluation completed successfully!")
    print("Reports saved inside:", REPORT_DIR)


if __name__ == "__main__":
    evaluate_models()