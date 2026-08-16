import json
import os
import sys
from pathlib import Path

from kafka import KafkaConsumer


# Allow this file to work both as:
# python streaming/fraud_stream.py
# and:
# python -m streaming.fraud_stream
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.database import FraudEvent, SessionLocal, init_db
from app.models import Transaction
from ml.predict import predictor


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

TRANSACTION_TOPIC = os.getenv(
    "TRANSACTION_TOPIC",
    "transactions",
)

CONSUMER_GROUP = os.getenv(
    "KAFKA_CONSUMER_GROUP",
    "fraud-detection-consumer",
)


def create_consumer() -> KafkaConsumer:
    """Create and configure the Kafka transaction consumer."""

    return KafkaConsumer(
        TRANSACTION_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=CONSUMER_GROUP,
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
    )


def process_transaction(payload: dict) -> None:
    """
    Validate, score, and persist a transaction.

    Flow:
    Kafka event
        -> Pydantic validation
        -> Feature engineering
        -> Fraud model inference
        -> PostgreSQL persistence
    """

    transaction = Transaction(**payload)

    prediction = predictor.predict(transaction)

    db = SessionLocal()

    try:
        event = FraudEvent(
            transaction_id=transaction.transaction_id,
            user_id=transaction.user_id,
            amount=transaction.amount,
            timestamp=transaction.timestamp,
            fraud_probability=prediction["fraud_probability"],
            is_fraud=prediction["is_fraud"],
        )

        # merge() makes processing idempotent for the same
        # transaction_id instead of blindly inserting duplicates.
        db.merge(event)

        db.commit()

        print(
            f"{transaction.transaction_id} "
            f"| amount=${transaction.amount:.2f} "
            f"| fraud_probability="
            f"{prediction['fraud_probability']:.4f} "
            f"| is_fraud={prediction['is_fraud']}"
        )

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def run_stream() -> None:
    """Continuously consume and score transactions from Kafka."""

    init_db()

    consumer = create_consumer()

    print(
        f"Listening to Kafka topic '{TRANSACTION_TOPIC}' "
        f"via {KAFKA_BOOTSTRAP_SERVERS}"
    )

    try:
        for message in consumer:
            try:
                process_transaction(message.value)

            except Exception as exc:
                print(
                    "Failed to process transaction "
                    f"at offset {message.offset}: {exc}"
                )

    except KeyboardInterrupt:
        print("\nStopping fraud stream...")

    finally:
        consumer.close()


if __name__ == "__main__":
    run_stream()