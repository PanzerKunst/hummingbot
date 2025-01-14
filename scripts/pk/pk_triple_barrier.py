from dataclasses import dataclass
from decimal import Decimal

from hummingbot.core.data_type.common import OrderType
from scripts.pk.pk_trailing_stop import PkTrailingStop


@dataclass
class TripleBarrier:
    stop_loss_delta: Decimal | None = None
    take_profit_delta: Decimal | None = None
    time_limit: int | None = None
    trailing_stop: PkTrailingStop | None = None
    open_order_type: OrderType = OrderType.LIMIT
    take_profit_order_type: OrderType = OrderType.MARKET
    stop_loss_order_type: OrderType = OrderType.MARKET
    time_limit_order_type: OrderType = OrderType.MARKET
