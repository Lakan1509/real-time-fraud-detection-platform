from datetime import datetime

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    amount: float = Field(gt=0)
    merchant_id: str
    merchant_category: str
    country: str
    device_id: str
    timestamp: datetime
    is_card_present: bool
    previous_transaction_count_24h: int = Field(ge=0)
    previous_transaction_amount_24h: float = Field(ge=0)