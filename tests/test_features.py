from datetime import datetime

from app.models import Transaction
from ml.features import FEATURE_NAMES, build_features, feature_vector


def sample_transaction() -> Transaction:
    return Transaction(
        transaction_id="tx-001",
        user_id="user-001",
        amount=500.0,
        merchant_id="merchant-001",
        merchant_category="electronics",
        country="US",
        device_id="device-001",
        timestamp=datetime(2026, 8, 15, 23, 30),
        is_card_present=False,
        previous_transaction_count_24h=3,
        previous_transaction_amount_24h=250.0,
    )


def test_build_features():
    transaction = sample_transaction()
    features = build_features(transaction)

    assert features["amount"] == 500.0
    assert features["hour"] == 23.0
    assert features["is_card_present"] == 0.0
    assert features["previous_transaction_count_24h"] == 3.0
    assert features["previous_transaction_amount_24h"] == 250.0
    assert features["amount_to_previous_ratio"] == 2.0


def test_feature_vector_order():
    transaction = sample_transaction()
    vector = feature_vector(transaction)

    assert len(vector) == len(FEATURE_NAMES)
    assert vector[0] == 500.0
    assert vector[-1] == 2.0