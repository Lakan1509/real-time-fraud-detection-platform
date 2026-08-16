from datetime import datetime

from app.models import Transaction
from ml.predict import FraudPredictor


def test_fraud_prediction():
    predictor = FraudPredictor()

    transaction = Transaction(
        transaction_id="fraud-tx-001",
        user_id="user-001",
        amount=1500.0,
        merchant_id="merchant-001",
        merchant_category="electronics",
        country="US",
        device_id="device-new",
        timestamp=datetime(2026, 8, 15, 2, 30),
        is_card_present=False,
        previous_transaction_count_24h=12,
        previous_transaction_amount_24h=300.0,
    )

    result = predictor.predict(transaction)

    assert result["transaction_id"] == "fraud-tx-001"
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert isinstance(result["is_fraud"], bool)


def test_low_risk_prediction():
    predictor = FraudPredictor()

    transaction = Transaction(
        transaction_id="normal-tx-001",
        user_id="user-002",
        amount=25.0,
        merchant_id="merchant-002",
        merchant_category="grocery",
        country="US",
        device_id="device-known",
        timestamp=datetime(2026, 8, 15, 14, 0),
        is_card_present=True,
        previous_transaction_count_24h=1,
        previous_transaction_amount_24h=200.0,
    )

    result = predictor.predict(transaction)

    assert result["transaction_id"] == "normal-tx-001"
    assert 0.0 <= result["fraud_probability"] <= 1.0