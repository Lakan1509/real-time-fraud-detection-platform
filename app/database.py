import os

from sqlalchemy import Boolean, Column, DateTime, Float, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://fraud:fraud@localhost:5432/fraud_detection",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class FraudEvent(Base):
    __tablename__ = "fraud_events"

    transaction_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)