
import pytest
from fastapi.testclient import TestClient
from api import main

client = TestClient(main.app)


@pytest.fixture
def customer_data():
    return {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 850.0,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "available_models" in response.json()


def test_prediction_endpoint(customer_data, tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    model_path = model_dir / "random_forest.joblib"
    model_path.touch()

    preprocessor_path = tmp_path / "preprocessor.joblib"
    preprocessor_path.touch()

    monkeypatch.setattr(main, "MODEL_DIR", model_dir)
    monkeypatch.setattr(main, "PREPROCESSOR_PATH", preprocessor_path)

    class FakePreprocessor:
        def transform(self, data):
            return data

    class FakeModel:
        def predict(self, data):
            return [1]

        def predict_proba(self, data):
            return [[0.2, 0.8]]

    def fake_load(path):
        if str(path).endswith("random_forest.joblib"):
            return FakeModel()
        return FakePreprocessor()

    monkeypatch.setattr(main.joblib, "load", fake_load)

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 200
    assert response.json()["prediction"] == "Churn"
    assert response.json()["churn_probability"] == 0.8


def test_prediction_without_model(customer_data, tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    preprocessor_path = tmp_path / "preprocessor.joblib"
    preprocessor_path.touch()

    monkeypatch.setattr(main, "MODEL_DIR", model_dir)
    monkeypatch.setattr(main, "PREPROCESSOR_PATH", preprocessor_path)

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 404


def test_invalid_customer_input(customer_data):
    customer_data["SeniorCitizen"] = 2

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 422
