import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

TRANSACTION_TOPIC = os.getenv(
    "TRANSACTION_TOPIC",
    "transactions",
)


def create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        retries=5,
    )


def generate_transaction() -> dict:
    high_risk = random.random() < 0.12

    if high_risk:
        amount = round(random.uniform(900, 3000), 2)
        hour = random.choice([0, 1, 2, 3, 4, 23])
        card_present = False
        tx_count = random.randint(9, 18)
        previous_amount = round(random.uniform(100, 500), 2)
    else:
        amount = round(random.uniform(5, 500), 2)
        hour = random.randint(6, 22)
        card_present = random.choice([True, True, True, False])
        tx_count = random.randint(0, 6)
        previous_amount = round(random.uniform(50, 1000), 2)

    timestamp = datetime.now(timezone.utc).replace(
        hour=hour,
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
        microsecond=0,
    )

    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": f"user-{random.randint(1, 500)}",
        "amount": amount,
        "merchant_id": f"merchant-{random.randint(1, 100)}",
        "merchant_category": random.choice(
            [
                "grocery",
                "electronics",
                "travel",
                "restaurant",
                "retail",
                "entertainment",
            ]
        ),
        "country": random.choice(["US", "US", "US", "CA", "GB"]),
        "device_id": f"device-{random.randint(1, 1000)}",
        "timestamp": timestamp.isoformat(),
        "is_card_present": card_present,
        "previous_transaction_count_24h": tx_count,
        "previous_transaction_amount_24h": previous_amount,
    }


def publish_transactions(interval_seconds: float = 1.0) -> None:
    producer = create_producer()

    print(
        f"Publishing transactions to '{TRANSACTION_TOPIC}' "
        f"via {KAFKA_BOOTSTRAP_SERVERS}"
    )

    try:
        while True:
            transaction = generate_transaction()

            producer.send(
                TRANSACTION_TOPIC,
                value=transaction,
            )

            producer.flush()

            print(
                f"Published {transaction['transaction_id']} "
                f"| ${transaction['amount']}"
            )

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nStopping transaction producer...")

    finally:
        producer.close()


if __name__ == "__main__":
    publish_transactions()