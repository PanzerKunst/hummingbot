from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig
from hummingbot.strategy_v2.models.executors import CloseType


@dataclass
class TrackedOrderDetails:
    connector_name: str
    trading_pair: str
    side: TradeType
    order_id: str
    position: str
    amount: Decimal
    entry_price: Decimal
    triple_barrier_config: TripleBarrierConfig
    filled_amount: Decimal = Decimal(0)
    exchange_order_id: Optional[str] = None
    created_at: Optional[float] = None
    last_filled_at: Optional[float] = None
    terminated_at: Optional[float] = None
    close_type: Optional[CloseType] = None
