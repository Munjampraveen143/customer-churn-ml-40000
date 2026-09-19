from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def sample_customer():
    return {
        "gender": "Male",
        "age": 35,
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 5,
        "phone_service": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 120.0,
        "total_charges": 600.0,
        "support_tickets": 5,
        "late_payments": 3,
        "satisfaction_score": 3
    }

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_prediction():
    response = client.post("/predict", json=sample_customer())
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert "risk" in body
    assert 0 <= body["churn_probability"] <= 1
