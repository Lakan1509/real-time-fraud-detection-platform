from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "Real-Time Fraud Detection Platform"
    assert data["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict():
    payload = {
        "transaction_id": "test-transaction-001",
        "user_id": "test-user",
        "amount": 250.00,
        "merchant_id": "merchant-test",
        "merchant_category": "retail",
        "country": "US",
        "device_id": "device-test",
        "timestamp": "2026-08-15T12:00:00+00:00",
        "is_card_present": True,
        "previous_transaction_count_24h": 2,
        "previous_transaction_amount_24h": 300.00,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "is_fraud" in data

    assert 0 <= data["fraud_probability"] <= 1
    assert isinstance(data["is_fraud"], bool)


def test_recent_events():
    response = client.get("/events/recent?limit=10")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)
    assert len(data) <= 10


def test_high_risk_events():
    response = client.get(
        "/events/high-risk?threshold=0.5&limit=20"
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)

    for event in data:
        assert event["fraud_probability"] >= 0.5


def test_stats():
    response = client.get("/stats")

    assert response.status_code == 200

    data = response.json()

    assert "total_transactions" in data
    assert "fraud_transactions" in data
    assert "fraud_rate" in data
    assert "average_fraud_probability" in data

    assert data["total_transactions"] >= 0
    assert data["fraud_transactions"] >= 0
    assert 0 <= data["fraud_rate"] <= 1