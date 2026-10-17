import joblib
import pytest
from fastapi.testclient import TestClient

from heart_risk.api import create_app


@pytest.fixture
def client(tmp_path, fitted_model):
    path = tmp_path / "test_model.joblib"
    joblib.dump(fitted_model, path)
    with TestClient(create_app(path)) as client:
        yield client


def test_prediction_contract(client, patient):
    response = client.post("/predict", json=patient)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert 0 <= body["heart_disease_probability"] <= 1
    expected = body["heart_disease_probability"] if body["prediction"] else 1-body["heart_disease_probability"]
    assert body["confidence"] == expected
    assert response.headers["X-Request-ID"]


@pytest.mark.parametrize("field,value", [("age", -1), ("sex", 4), ("chest_pain_type", 0),
                                        ("thalassemia_test", 5), ("major_vessels", 4)])
def test_invalid_fields_rejected(client, patient, field, value):
    patient[field] = value
    assert client.post("/predict", json=patient).status_code == 422


def test_missing_required_field(client, patient):
    del patient["age"]
    assert client.post("/predict", json=patient).status_code == 422


def test_extra_field_rejected(client, patient):
    patient["heart_disease"] = 1
    assert client.post("/predict", json=patient).status_code == 422


def test_nullable_uci_fields(client, patient):
    patient["major_vessels"] = None
    patient["thalassemia_test"] = None
    assert client.post("/predict", json=patient).status_code == 200


def test_readiness_and_monitoring(client, patient):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["model_loaded"] is True
    client.post("/predict", json=patient)
    body = client.get("/metrics").text
    assert "heart_api_predictions_total" in body
    assert "heart_api_request_duration_seconds_bucket" in body


def test_missing_model_fails_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass
