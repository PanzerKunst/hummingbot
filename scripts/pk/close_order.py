from dataclasses import dataclass

from scripts.pk.tracked_order import TrackedOrder


@dataclass
class CloseOrder:
    order_id: str
    tracked_order: TrackedOrder
    created_at: float
