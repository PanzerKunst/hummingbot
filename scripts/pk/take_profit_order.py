from dataclasses import dataclass
from decimal import Decimal

from scripts.pk.tracked_order import TrackedOrder


@dataclass
class TakeProfitOrder:
    order_id: str
    tracked_order: TrackedOrder
    amount: Decimal
    entry_price: Decimal
    created_at: float
    exchange_order_id: str | None = None
    last_filled_at: float | None = None
    last_filled_price: Decimal | None = None
