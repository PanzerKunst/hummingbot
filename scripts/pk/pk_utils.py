from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List

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


def compute_delta_bps(price_a: Decimal, price_b: Decimal) -> Decimal:
    if price_b == 0:
        return Decimal("Infinity")

    delta_bps = (price_a - price_b) / price_b * 10000
    return delta_bps


def compute_recent_price_delta_pct(low_series: pd.Series, high_series: pd.Series, nb_candles_to_consider: int, nb_excluded: int = 0) -> Decimal:
    start_index = nb_candles_to_consider + nb_excluded
    end_index = nb_excluded

    last_lows = low_series.iloc[-start_index:-end_index] if end_index > 0 else low_series.tail(start_index)
    lowest_price = Decimal(last_lows.min())

    last_highs = high_series.iloc[-start_index:-end_index] if end_index > 0 else high_series.tail(start_index)
    highest_price = Decimal(last_highs.max())

    return (highest_price - lowest_price) / highest_price * 100


def compute_sell_orders_pnl_pct(filled_sell_orders: List[TrackedOrderDetails], current_price: Decimal) -> Decimal:
    worst_filled_price = min(filled_sell_orders, key=lambda order: order.last_filled_price).last_filled_price
    return (worst_filled_price - current_price) / worst_filled_price * 100


def compute_buy_orders_pnl_pct(filled_buy_orders: List[TrackedOrderDetails], current_price: Decimal) -> Decimal:
    worst_filled_price = max(filled_buy_orders, key=lambda order: order.last_filled_price).last_filled_price
    return (current_price - worst_filled_price) / worst_filled_price * 100


def has_current_price_reached_stop_loss(tracked_order: TrackedOrderDetails, current_price: Decimal) -> bool:
    stop_loss = tracked_order.triple_barrier.stop_loss

    if not stop_loss:
        return False

    side = tracked_order.side
    ref_price = tracked_order.last_filled_price or tracked_order.entry_price

    if side == TradeType.SELL:
        return current_price > ref_price * (1 + stop_loss)

    return current_price < ref_price * (1 - stop_loss)


def has_current_price_reached_take_profit(tracked_order: TrackedOrderDetails, current_price: Decimal) -> bool:
    take_profit = tracked_order.triple_barrier.take_profit

    if not take_profit:
        return False

    side = tracked_order.side
    ref_price = tracked_order.last_filled_price or tracked_order.entry_price

    if side == TradeType.SELL:
        return current_price < ref_price * (1 - take_profit)

    return current_price > ref_price * (1 + take_profit)


def update_trailing_stop(tracked_order: TrackedOrderDetails, current_price: Decimal):
    trailing_stop = tracked_order.triple_barrier.trailing_stop

    if not trailing_stop:
        return

    activation_price = get_take_profit_price(tracked_order.side, tracked_order.last_filled_price, trailing_stop.activation_delta)
    price_to_compare = tracked_order.trailing_stop_best_price or activation_price

    if tracked_order.side == TradeType.SELL:
        if current_price < price_to_compare:
            tracked_order.trailing_stop_best_price = current_price
        return

    if current_price > price_to_compare:
        tracked_order.trailing_stop_best_price = current_price


def should_close_trailing_stop(tracked_order: TrackedOrderDetails, current_price: Decimal) -> bool:
    trailing_stop = tracked_order.triple_barrier.trailing_stop

    if not trailing_stop or not tracked_order.trailing_stop_best_price:
        return False

    trailing_delta = trailing_stop.trailing_delta

    if tracked_order.side == TradeType.SELL:
        return current_price > tracked_order.trailing_stop_best_price * (1 + trailing_delta)

    return current_price < tracked_order.trailing_stop_best_price * (1 - trailing_delta)


def get_take_profit_price(side: TradeType, ref_price: Decimal, take_profit_delta: Decimal) -> Decimal:
    if side == TradeType.SELL:
        return ref_price * (1 - take_profit_delta)

    return ref_price * (1 + take_profit_delta)


def has_unfilled_order_expired(tracked_order: TrackedOrderDetails, expiration_min: int, current_timestamp: float) -> bool:
    created_at = tracked_order.created_at

    return created_at + expiration_min * 60 < current_timestamp


def has_filled_order_reached_time_limit(tracked_order: TrackedOrderDetails, current_timestamp: float) -> bool:
    time_limit = tracked_order.triple_barrier.time_limit

    if not time_limit:
        return False

    filled_at = tracked_order.last_filled_at

    return filled_at + time_limit < current_timestamp


def was_an_order_recently_opened(tracked_orders: List[TrackedOrderDetails], seconds: int, current_timestamp: float) -> bool:
    if len(tracked_orders) == 0:
        return False

    most_recent_created_at = max(tracked_orders, key=lambda order: order.created_at).created_at

    return most_recent_created_at + seconds > current_timestamp


def timestamp_to_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat()


# TODO: return int instead of Decimal
def compute_rsi_pullback_difference(rsi: Decimal) -> Decimal:
    """
    When `rsi > 50`:
    i:3 rsi:75 result:72 (rsi-i)
    i:4 rsi:78 result:74 (rsi-i)
    i:5 rsi:81 result:76 (rsi-i)
    i:6 rsi:84 result:78 (rsi-i)
    i:7 rsi:87 result:80 (rsi-i)
    i:8 rsi:90 result:82 (rsi-i)
    i:9 rsi:93 result:84 (rsi-i)

    When `rsi < 50`:
    i:3 rsi:25 result:28 (rsi+i)
    i:4 rsi:22 result:26 (rsi+i)
    i:5 rsi:18 result:24 (rsi+i)
    i:6 rsi:15 result:22 (rsi+i)
    i:7 rsi:12 result:20 (rsi+i)
    i:8 rsi:09 result:18 (rsi+i)
    i:9 rsi:06 result:16 (rsi+i)
    """
    if rsi > 50:
        if rsi < 75:
            return Decimal(2.0)

        decrement = ((rsi - 75) // 3) + 3
        return decrement

    if rsi > 25:
        return Decimal(2.0)

    increment = ((25 - rsi) // 3) + 3
    return increment


class Trend(Enum):
    UP = "UP"
    DOWN = "DOWN"
