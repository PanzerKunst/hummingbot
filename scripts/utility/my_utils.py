from datetime import datetime
from decimal import Decimal
from enum import Enum

from hummingbot.connector.derivative.position import Position
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


def are_positions_equal(position_1: Position, position_2: Position) -> bool:
    return (position_1.position_side == position_2.position_side
            and position_1.trading_pair == position_2.trading_pair
            and position_1.amount == position_2.amount)


def calculate_delta_bps(price_a: Decimal, price_b: Decimal) -> Decimal:
    if price_b == 0:
        return Decimal("Infinity")

    delta_bps = (price_a - price_b) / price_b * 10000
    return delta_bps


def has_order_expired(executor: ExecutorInfo, time_limit: int, current_timestamp: int) -> bool:
    """
    :param time_limit: In seconds
    :param current_timestamp: In seconds
    """
    delta = current_timestamp - executor.timestamp
    return delta > time_limit


def timestamp_to_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat()


class Trend(Enum):
    UP = "UP"
    DOWN = "DOWN"
