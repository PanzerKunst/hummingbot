from decimal import Decimal

from hummingbot.connector.derivative.position import Position


def are_positions_equal(position_1: Position, position_2: Position) -> bool:
    return (position_1.position_side == position_2.position_side
            and position_1.trading_pair == position_2.trading_pair
            and position_1.amount == position_2.amount)


def calculate_delta_bps(price_a: Decimal, price_b: Decimal) -> Decimal:
    delta_bps = (price_a - price_b) / price_b * 10000
    return delta_bps
