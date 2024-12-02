from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from hummingbot.core.data_type.common import OrderType
from scripts.pk.pk_trailing_stop import PkTrailingStop


@dataclass
class TripleBarrier:
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    time_limit: Optional[int] = None
    trailing_stop: Optional[PkTrailingStop] = None
    open_order_type: OrderType = OrderType.LIMIT
    take_profit_order_type: OrderType = OrderType.MARKET
    stop_loss_order_type: OrderType = OrderType.MARKET
    time_limit_order_type: OrderType = OrderType.MARKET
