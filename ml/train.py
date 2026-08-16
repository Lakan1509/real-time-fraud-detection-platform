from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


MODEL_PATH = Path("models/fraud_model.joblib")


def generate_training_data(
    n_samples: int = 3000,
    random_seed: int = 42,
):
    rng = np.random.default_rng(random_seed)

    amount = rng.gamma(shape=2.0, scale=120.0, size=n_samples)
    hour = rng.integers(0, 24, size=n_samples)
    card_present = rng.integers(0, 2, size=n_samples)
    tx_count_24h = rng.poisson(lam=3.0, size=n_samples)
    previous_amount_24h = rng.gamma(shape=2.0, scale=250.0, size=n_samples)

    ratio = amount / np.maximum(previous_amount_24h, 1.0)

    X = np.column_stack(
        [
            amount,
            hour,
            card_present,
            tx_count_24h,
            previous_amount_24h,
            ratio,
        ]
    )

    fraud_score = (
        (amount > 900).astype(int)
        + ((hour < 5) | (hour > 22)).astype(int)
        + (card_present == 0).astype(int)
        + (tx_count_24h > 8).astype(int)
        + (ratio > 2.5).astype(int)
    )

    y = (fraud_score >= 3).astype(int)

    return X, y


def train_model():
    X, y = generate_training_data()

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


if __name__ == "__main__":
    train_model()
    print(f"Model saved to {MODEL_PATH}")