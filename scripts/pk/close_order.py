from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from scripts.pk.tracked_order_details import TrackedOrderDetails


@dataclass
class CloseOrder:
    order_id: str
    tracked_order: TrackedOrderDetails
    amount: Decimal
    entry_price: Decimal
    created_at: float
    filled_amount: Decimal = Decimal(0)
    filled_at: Optional[float] = None
    filled_price: Optional[Decimal] = None
