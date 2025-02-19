from dataclasses import dataclass

from scripts.pk.tracked_order_details import TrackedOrderDetails


@dataclass
class CloseOrder:
    order_id: str
    tracked_order: TrackedOrderDetails
    created_at: float
