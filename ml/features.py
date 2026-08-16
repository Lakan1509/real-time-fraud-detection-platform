from app.models import Transaction


FEATURE_NAMES = [
    "amount",
    "hour",
    "is_card_present",
    "previous_transaction_count_24h",
    "previous_transaction_amount_24h",
    "amount_to_previous_ratio",
]


def build_features(transaction: Transaction) -> dict[str, float]:
    previous_amount = transaction.previous_transaction_amount_24h

    if previous_amount > 0:
        ratio = transaction.amount / previous_amount
    else:
        ratio = transaction.amount

    return {
        "amount": float(transaction.amount),
        "hour": float(transaction.timestamp.hour),
        "is_card_present": float(transaction.is_card_present),
        "previous_transaction_count_24h": float(
            transaction.previous_transaction_count_24h
        ),
        "previous_transaction_amount_24h": float(previous_amount),
        "amount_to_previous_ratio": float(ratio),
    }


def feature_vector(transaction: Transaction) -> list[float]:
    features = build_features(transaction)
    return [features[name] for name in FEATURE_NAMES]