from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Tuple

import pandas as pd

from hummingbot.connector.derivative.position import Position
from hummingbot.core.data_type.common import TradeType
from scripts.pk.take_profit_order import TakeProfitOrder
from scripts.pk.tracked_order import TrackedOrder


def average(*args) -> Decimal:
    result = sum(args) / len(args) if args else 0
    return Decimal(result)


def are_positions_equal(position_1: Position, position_2: Position) -> bool:
    return (position_1.position_side == position_2.position_side
            and position_1.trading_pair == position_2.trading_pair
            and position_1.amount == position_2.amount)


def calc_recent_price_delta_pct(low_series: pd.Series, high_series: pd.Series, nb_candles_to_consider: int, nb_excluded: int = 0) -> Decimal:
    start_index = nb_candles_to_consider + nb_excluded
    end_index = nb_excluded

    last_lows = low_series.iloc[-start_index:-end_index] if end_index > 0 else low_series.tail(start_index)
    lowest_price = Decimal(last_lows.min())

    last_highs = high_series.iloc[-start_index:-end_index] if end_index > 0 else high_series.tail(start_index)
    highest_price = Decimal(last_highs.max())

    return (highest_price - lowest_price) / highest_price * 100


def calc_avg_filled_price(filled_orders: List[TrackedOrder]) -> Decimal:
    total_value: Decimal = sum(
        (
            o.last_filled_price * (o.filled_amount - o.filled_tp_amount)
            for o in filled_orders
        ),
        Decimal("0")
    )

    total_amount: Decimal = sum(
        (
            (o.filled_amount - o.filled_tp_amount)
            for o in filled_orders
        ),
        Decimal("0")
    )

    if total_amount == 0:
        return Decimal("0")

    return total_value / total_amount


def calc_sell_orders_pnl_pct(filled_sell_orders: List[TrackedOrder], current_price: Decimal) -> Decimal:
    avg_price: Decimal = calc_avg_filled_price(filled_sell_orders)
    return (avg_price - current_price) / avg_price * 100


def calc_buy_orders_pnl_pct(filled_buy_orders: List[TrackedOrder], current_price: Decimal) -> Decimal:
    avg_price: Decimal = calc_avg_filled_price(filled_buy_orders)
    return (current_price - avg_price) / avg_price * 100


def calc_stop_loss_price(side: TradeType, ref_price: Decimal, stop_loss_delta: Decimal) -> Decimal:
    if side == TradeType.SELL:
        return ref_price * (1 + stop_loss_delta)

    return ref_price * (1 - stop_loss_delta)


def calc_take_profit_price(side: TradeType, ref_price: Decimal, take_profit_delta: Decimal) -> Decimal:
    if side == TradeType.SELL:
        return ref_price * (1 - take_profit_delta)

    return ref_price * (1 + take_profit_delta)


def calc_avg_position_price(filled_orders: List[TrackedOrder]) -> Decimal:
    if len(filled_orders) == 0:
        return Decimal("0")

    total_amount = sum(order.filled_amount for order in filled_orders)
    total_cost = sum(order.filled_amount * order.last_filled_price for order in filled_orders)

    return Decimal(total_cost / total_amount)


def has_current_price_reached_stop_loss(tracked_order: TrackedOrder, current_price: Decimal) -> bool:
    stop_loss_delta: Decimal | None = tracked_order.triple_barrier.stop_loss_delta

    if stop_loss_delta is None:
        return False

    side: TradeType = tracked_order.side
    ref_price: Decimal = tracked_order.last_filled_price or tracked_order.entry_price
    stop_loss_price: Decimal = calc_stop_loss_price(side, ref_price, stop_loss_delta)

    if side == TradeType.SELL:
        return current_price > stop_loss_price

    return current_price < stop_loss_price


def has_unfilled_order_expired(order: TrackedOrder | TakeProfitOrder, expiration: int, current_timestamp: float) -> bool:
    created_at = order.created_at

    return created_at + expiration < current_timestamp


def has_filled_order_reached_time_limit(tracked_order: TrackedOrder, current_timestamp: float) -> bool:
    time_limit: int | None = tracked_order.triple_barrier.time_limit

    if time_limit is None:
        return False

    filled_at = tracked_order.last_filled_at

    return current_timestamp > filled_at + time_limit


def was_an_order_recently_opened(tracked_orders: List[TrackedOrder], seconds: int, current_timestamp: float) -> bool:
    if len(tracked_orders) == 0:
        return False

    most_recent_created_at = max(tracked_orders, key=lambda order: order.created_at).created_at

    return most_recent_created_at + seconds > current_timestamp


def combine_filled_orders(filled_orders: List[TrackedOrder]) -> TrackedOrder:
    non_terminated_filled_orders = [order for order in filled_orders if order.terminated_at is None]
    combined_filled_amount: Decimal = Decimal("0")

    for order in non_terminated_filled_orders:
        if order.side == TradeType.SELL:
            combined_filled_amount -= order.filled_amount

        else:
            combined_filled_amount += order.filled_amount

    first_order = filled_orders[0]

    return TrackedOrder(
        connector_name=first_order.connector_name,
        trading_pair=first_order.trading_pair,
        side=TradeType.SELL if combined_filled_amount < 0 else TradeType.BUY,
        order_id="combined",
        amount=abs(combined_filled_amount),
        entry_price=first_order.last_filled_price,
        triple_barrier=first_order.triple_barrier,
        tag=first_order.tag,
        created_at=first_order.created_at,
        filled_amount=abs(combined_filled_amount)
    )


def timestamp_to_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat()


def iso_to_timestamp(iso_date: str) -> float:
    return datetime.strptime(iso_date, "%Y-%m-%d").timestamp()


def normalize_timestamp_to_midnight(timestamp: float) -> float:
    dt = datetime.fromtimestamp(timestamp)
    return datetime(dt.year, dt.month, dt.day).timestamp()


def bucket_minute(tracked_order: TrackedOrder) -> Tuple[int, int, int, int, int]:
    """
    Convert an order’s created_at timestamp into a (year, month, day, hour, minute) tuple.
    """
    dt: datetime = datetime.fromtimestamp(tracked_order.created_at, tz=timezone.utc)
    return dt.year, dt.month, dt.day, dt.hour, dt.minute


# TODO: return int instead of Decimal
def calc_rsi_pullback_difference(rsi: Decimal) -> Decimal:
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
