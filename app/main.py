from fastapi import FastAPI
from sqlalchemy import desc, func

from app.database import FraudEvent, SessionLocal
from app.models import Transaction
from ml.predict import predictor


app = FastAPI(
    title="Real-Time Fraud Detection Platform",
    description=(
        "Production-style fraud detection API with ML inference, "
        "Kafka streaming, Spark Structured Streaming, PostgreSQL persistence, "
        "and monitoring-ready endpoints."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "name": "Real-Time Fraud Detection Platform",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
    }


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    return predictor.predict(transaction)


@app.get("/events/recent")
def recent_events(limit: int = 20):
    db = SessionLocal()

    try:
        events = (
            db.query(FraudEvent)
            .order_by(desc(FraudEvent.timestamp))
            .limit(limit)
            .all()
        )

        return [
            {
                "transaction_id": event.transaction_id,
                "user_id": event.user_id,
                "amount": event.amount,
                "timestamp": event.timestamp,
                "fraud_probability": event.fraud_probability,
                "is_fraud": event.is_fraud,
            }
            for event in events
        ]

    finally:
        db.close()


@app.get("/events/high-risk")
def high_risk_events(
    threshold: float = 0.5,
    limit: int = 20,
):
    db = SessionLocal()

    try:
        events = (
            db.query(FraudEvent)
            .filter(FraudEvent.fraud_probability >= threshold)
            .order_by(desc(FraudEvent.fraud_probability))
            .limit(limit)
            .all()
        )

        return [
            {
                "transaction_id": event.transaction_id,
                "user_id": event.user_id,
                "amount": event.amount,
                "timestamp": event.timestamp,
                "fraud_probability": event.fraud_probability,
                "is_fraud": event.is_fraud,
            }
            for event in events
        ]

    finally:
        db.close()


@app.get("/stats")
def fraud_stats():
    db = SessionLocal()

    try:
        total_transactions = (
            db.query(func.count(FraudEvent.transaction_id))
            .scalar()
            or 0
        )

        fraud_transactions = (
            db.query(func.count(FraudEvent.transaction_id))
            .filter(FraudEvent.is_fraud.is_(True))
            .scalar()
            or 0
        )

        average_probability = (
            db.query(func.avg(FraudEvent.fraud_probability))
            .scalar()
            or 0.0
        )

        fraud_rate = (
            fraud_transactions / total_transactions
            if total_transactions > 0
            else 0.0
        )

        return {
            "total_transactions": total_transactions,
            "fraud_transactions": fraud_transactions,
            "fraud_rate": round(fraud_rate, 4),
            "average_fraud_probability": round(
                float(average_probability),
                4,
            ),
        }

    finally:
        db.close()