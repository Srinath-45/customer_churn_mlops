
import pandas as pd
import joblib

from src.data import preprocess


def test_preprocessing_creates_files(tmp_path, monkeypatch):

    # Create a sample dataset
    data = {
        "customerID": [f"C{i}" for i in range(30)],
        "gender": ["Male", "Female"] * 15,
        "SeniorCitizen": [0, 1] * 15,
        "Partner": ["Yes", "No"] * 15,
        "Dependents": ["No", "Yes"] * 15,
        "tenure": [i % 72 for i in range(30)],
        "PhoneService": ["Yes", "No"] * 15,
        "MultipleLines": ["No", "Yes"] * 15,
        "InternetService": ["DSL", "Fiber optic", "No"] * 10,
        "OnlineSecurity": ["Yes", "No", "No"] * 10,
        "OnlineBackup": ["No", "Yes", "No"] * 10,
        "DeviceProtection": ["Yes", "No", "No"] * 10,
        "TechSupport": ["No", "Yes", "No"] * 10,
        "StreamingTV": ["Yes", "No", "No"] * 10,
        "StreamingMovies": ["No", "Yes", "No"] * 10,
        "Contract": ["Month-to-month", "One year", "Two year"] * 10,
        "PaperlessBilling": ["Yes", "No"] * 15,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 15,
        "MonthlyCharges": [50.0 + i for i in range(30)],
        "TotalCharges": [100.0 + i * 10 for i in range(30)],
        "Churn": ["Yes", "No"] * 15,
    }

    raw_file = tmp_path / "dataset.csv"
    pd.DataFrame(data).to_csv(raw_file, index=False)

    processed_dir = tmp_path / "processed"

    # Configure temporary paths
    monkeypatch.setattr(
        preprocess,
        "load_parameters",
        lambda: {
            "data": {
                "raw_path": str(raw_file),
                "processed_dir": str(processed_dir),
            },
            "preprocessing": {
                "test_size": 0.2,
                "random_state": 42,
            },
        },
    )

    # Run preprocessing
    monkeypatch.chdir(tmp_path)
    preprocess.preprocess_data()

    # Check generated files
    train_file = processed_dir / "train.csv"
    test_file = processed_dir / "test.csv"

    preprocessor_file = (
        tmp_path / "artifacts" / "preprocessor" / "preprocessor.joblib"
    )

    assert train_file.exists()
    assert test_file.exists()
    assert preprocessor_file.exists()

    # Read processed datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Verify dataset sizes
    assert len(train_df) == 24
    assert len(test_df) == 6

    # Verify target column
    assert "Churn" in train_df.columns
    assert "Churn" in test_df.columns

    # Verify target values
    assert set(train_df["Churn"].unique()).issubset({0, 1})
    assert set(test_df["Churn"].unique()).issubset({0, 1})

    # Verify saved preprocessor
    saved_preprocessor = joblib.load(preprocessor_file)
    assert saved_preprocessor is not None
