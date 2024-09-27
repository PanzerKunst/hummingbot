from dataclasses import dataclass
from typing import Optional

from hummingbot.core.data_type.common import TradeType


@dataclass
class TrackedOrderDetails:
    connector_name: str
    trading_pair: str
    side: TradeType
    order_id: str
    position: str
    exchange_order_id: Optional[str] = None
    created_at: Optional[float] = None
    filled_at: Optional[float] = None
    cancelled_at: Optional[float] = None
