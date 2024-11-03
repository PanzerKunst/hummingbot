from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.pk.pk_triple_barrier import TripleBarrier


@dataclass
class TrackedOrderDetails:
    connector_name: str
    trading_pair: str
    side: TradeType
    order_id: str
    position: str
    amount: Decimal
    entry_price: Decimal
    triple_barrier: TripleBarrier
    filled_amount: Decimal = Decimal(0)
    exchange_order_id: Optional[str] = None
    created_at: Optional[float] = None
    last_filled_at: Optional[float] = None
    last_filled_price: Optional[Decimal] = None
    terminated_at: Optional[float] = None
    close_type: Optional[CloseType] = None
    trailing_stop_best_price: Optional[Decimal] = None
    ref: Optional[str] = None
