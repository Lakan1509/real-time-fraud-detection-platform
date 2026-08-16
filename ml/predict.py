from pathlib import Path

import joblib

from app.models import Transaction
from ml.features import feature_vector
from ml.train import MODEL_PATH, train_model


class FraudPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            return train_model()

        return joblib.load(self.model_path)

    def predict(self, transaction: Transaction) -> dict:
        vector = [feature_vector(transaction)]

        probability = float(
            self.model.predict_proba(vector)[0][1]
        )

        prediction = int(probability >= 0.5)

        return {
            "transaction_id": transaction.transaction_id,
            "fraud_probability": round(probability, 4),
            "is_fraud": bool(prediction),
        }


predictor = FraudPredictor()