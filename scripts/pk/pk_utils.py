from datetime import datetime
from decimal import Decimal
from enum import Enum

import pandas as pd

from hummingbot.connector.derivative.position import Position
from hummingbot.core.data_type.common import TradeType
from scripts.pk.tracked_order_details import TrackedOrderDetails


def average(*args) -> Decimal:
    result = sum(args) / len(args) if args else 0
    return Decimal(result)


def are_positions_equal(position_1: Position, position_2: Position) -> bool:
    return (position_1.position_side == position_2.position_side
            and position_1.trading_pair == position_2.trading_pair
            and position_1.amount == position_2.amount)


def calculate_delta_bps(price_a: Decimal, price_b: Decimal) -> Decimal:
    if price_b == 0:
        return Decimal("Infinity")

    delta_bps = (price_a - price_b) / price_b * 10000
    return delta_bps


def compute_recent_price_delta_pct(low_series: pd.Series, high_series: pd.Series, nb_candles_to_consider: int, nb_excluded: int = 0) -> Decimal:
    start_index = nb_candles_to_consider + nb_excluded
    end_index = nb_excluded

    last_lows = low_series.iloc[-start_index:-end_index]
    lowest_price = Decimal(last_lows.min())

    last_highs = high_series.iloc[-start_index:-end_index]
    highest_price = Decimal(last_highs.max())

    return (highest_price - lowest_price) / highest_price * 100


def has_current_price_reached_stop_loss(tracked_order: TrackedOrderDetails, current_price: Decimal) -> bool:
    stop_loss = tracked_order.triple_barrier_config.stop_loss

    if not stop_loss:
        return False

    side = tracked_order.side
    entry_price = tracked_order.entry_price

    if side == TradeType.SELL:
        return current_price > entry_price * (1 + stop_loss)

    return current_price < entry_price * (1 - stop_loss)


def has_current_price_reached_take_profit(tracked_order: TrackedOrderDetails, current_price: Decimal) -> bool:
    take_profit = tracked_order.triple_barrier_config.take_profit

    if not take_profit:
        return False

    side = tracked_order.side
    entry_price = tracked_order.entry_price

    if side == TradeType.SELL:
        return current_price < entry_price * (1 - take_profit)

    return current_price > entry_price * (1 + take_profit)


def has_current_price_activated_trailing_stop(tracked_order: TrackedOrderDetails, current_price: Decimal) -> bool:
    trailing_stop = tracked_order.triple_barrier_config.trailing_stop

    if not trailing_stop:
        return False

    if tracked_order.trailing_stop_best_price:  # Already activated
        return False

    activation_price = trailing_stop.activation_price

    if tracked_order.side == TradeType.SELL:
        return current_price < activation_price

    return current_price > activation_price


def should_close_trailing_stop(tracked_order: TrackedOrderDetails, current_price: Decimal) -> bool:
    trailing_stop = tracked_order.triple_barrier_config.trailing_stop

    if not trailing_stop:
        return False

    if not tracked_order.trailing_stop_best_price:
        return False

    trailing_delta = trailing_stop.trailing_delta

    if tracked_order.side == TradeType.SELL:
        return current_price > tracked_order.trailing_stop_best_price * (1 + trailing_delta)

    return current_price < tracked_order.trailing_stop_best_price * (1 - trailing_delta)


def get_take_profit_price(side: TradeType, entry_price: Decimal, take_profit_pct: Decimal) -> Decimal:
    if side == TradeType.SELL:
        return entry_price * (1 - take_profit_pct / 100)

    return entry_price * (1 + take_profit_pct / 100)


def has_unfilled_order_expired(tracked_order: TrackedOrderDetails, expiration_min: int, current_timestamp: float) -> bool:
    created_at = tracked_order.created_at

    return created_at + expiration_min * 60 < current_timestamp


def has_filled_order_reached_time_limit(tracked_order: TrackedOrderDetails, current_timestamp: float) -> bool:
    time_limit = tracked_order.triple_barrier_config.time_limit

    if not time_limit:
        return False

    filled_at = tracked_order.last_filled_at

    return filled_at + time_limit < current_timestamp


def timestamp_to_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat()


class Trend(Enum):
    UP = "UP"
    DOWN = "DOWN"
