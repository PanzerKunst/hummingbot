from dataclasses import dataclass
from decimal import Decimal

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
    ref: str
    filled_amount: Decimal = Decimal(0)
    exchange_order_id: str | None = None
    created_at: float | None = None
    last_filled_at: float | None = None
    last_filled_price: Decimal | None = None
    terminated_at: float | None = None
    close_type: CloseType | None = None
    trailing_stop_best_price: Decimal | None = None
