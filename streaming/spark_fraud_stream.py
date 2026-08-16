import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

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


transaction_schema = StructType(
    [
        StructField("transaction_id", StringType(), False),
        StructField("user_id", StringType(), False),
        StructField("amount", DoubleType(), False),
        StructField("merchant_id", StringType(), False),
        StructField("merchant_category", StringType(), False),
        StructField("country", StringType(), False),
        StructField("device_id", StringType(), False),
        StructField("timestamp", StringType(), False),
        StructField("is_card_present", BooleanType(), False),
        StructField(
            "previous_transaction_count_24h",
            IntegerType(),
            False,
        ),
        StructField(
            "previous_transaction_amount_24h",
            DoubleType(),
            False,
        ),
    ]
)


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("RealTimeFraudDetection")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


def process_batch(batch_df, batch_id: int) -> None:
    """
    Score one Spark micro-batch and persist predictions to PostgreSQL.
    """

    if batch_df.isEmpty():
        return

    rows = batch_df.collect()

    db = SessionLocal()

    try:
        for row in rows:
            payload = row.asDict(recursive=True)

            transaction = Transaction(**payload)

            prediction = predictor.predict(transaction)

            event = FraudEvent(
                transaction_id=transaction.transaction_id,
                user_id=transaction.user_id,
                amount=transaction.amount,
                timestamp=transaction.timestamp,
                fraud_probability=prediction["fraud_probability"],
                is_fraud=prediction["is_fraud"],
            )

            db.merge(event)

            print(
                f"[Spark batch={batch_id}] "
                f"{transaction.transaction_id} "
                f"| amount=${transaction.amount:.2f} "
                f"| fraud_probability="
                f"{prediction['fraud_probability']:.4f} "
                f"| is_fraud={prediction['is_fraud']}"
            )

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def main() -> None:
    init_db()

    spark = create_spark_session()

    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            KAFKA_BOOTSTRAP_SERVERS,
        )
        .option("subscribe", TRANSACTION_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    transactions = (
        raw_stream
        .selectExpr("CAST(value AS STRING) AS json_value")
        .select(
            from_json(
                col("json_value"),
                transaction_schema,
            ).alias("transaction")
        )
        .select("transaction.*")
        .filter(col("transaction_id").isNotNull())
    )

    query = (
        transactions.writeStream
        .foreachBatch(process_batch)
        .outputMode("append")
        .option(
            "checkpointLocation",
            "data/checkpoints/fraud-stream",
        )
        .start()
    )

    print(
        f"Spark fraud-scoring pipeline is listening to "
        f"'{TRANSACTION_TOPIC}' via {KAFKA_BOOTSTRAP_SERVERS}"
    )

    try:
        query.awaitTermination()

    except KeyboardInterrupt:
        print("\nStopping Spark fraud stream...")

        query.stop()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()