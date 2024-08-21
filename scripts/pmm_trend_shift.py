import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class PMMTrendShift(ScriptStrategyBase):
    bid_spread = 0.001
    ask_spread = 0.001
    order_refresh_time = 20
    order_amount = 0.01  # base
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice

    create_timestamp = 0

    # Candles params
    candle_exchange = "binance"
    candles_length = 30
    candles_interval = "1s"

    # Spread params
    # Define spreads dynamically as (NATR over candles_length) * spread_scalar
    bid_spread_scalar = 120  # Buy orders filled every 120 * 1s = 2min
    ask_spread_scalar = 60  # Sell orders filled every 1min

    # Max range of shift for Price
    max_shift_spread = max(bid_spread, ask_spread)  # For illiquid pairs
    # max_shift_spread = 0.000001  # For highly liquid pairs

    # Price shift params
    orig_price = 1
    reference_price = 1
    price_multiplier = 1
    trend_scalar = -1

    # Initializes candles
    candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=candles_interval
    ))

    markets = {exchange: {trading_pair}}

    # Start the candles when the script starts
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()
        self.connector = self.connectors[self.exchange]

    def on_stop(self):
        self.candles.stop()

    def on_tick(self):
        # Check if it's time to place orders
        is_time_to_create = self.create_timestamp <= datetime.now().timestamp() - self.order_refresh_time

        if is_time_to_create:
            self.cancel_all_orders()
            self.update_multipliers()
            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = datetime.now().timestamp()

    def cancel_all_orders(self):
        active_orders: List[LimitOrder] = self.get_active_orders(self.exchange)
        self.connector.batch_order_cancel(active_orders)

    def create_proposal(self) -> List[OrderCandidate]:
        ref_price = self.connector.get_price_by_type(self.trading_pair, self.price_source)

        # Make sure that the order spreads aren't better than the best bid/ask orders on the book
        default_buy_price = ref_price * Decimal(1 - self.bid_spread)
        best_bid = self.connector.get_price(self.trading_pair, False)
        buy_price = min(default_buy_price, best_bid)

        default_sell_price = ref_price * Decimal(1 + self.ask_spread)
        best_ask = self.connector.get_price(self.trading_pair, True)
        sell_price = max(default_sell_price, best_ask)

        buy_order = OrderCandidate(
            self.trading_pair,
            True,
            OrderType.LIMIT,
            TradeType.BUY,
            self.connector.quantize_order_amount(self.trading_pair, Decimal(self.order_amount)),
            self.connector.quantize_order_price(self.trading_pair, buy_price)
        )

        sell_order = OrderCandidate(
            self.trading_pair,
            True,
            OrderType.LIMIT,
            TradeType.SELL,
            self.connector.quantize_order_amount(self.trading_pair, Decimal(self.order_amount)),
            self.connector.quantize_order_price(self.trading_pair, sell_price)
        )

        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        return self.connector.budget_checker.adjust_candidates(proposal, True)

    def place_orders(self, orders: List[OrderCandidate]):
        for order in orders:
            # if order.amount == 0:
            # continue
            if order.order_side == TradeType.BUY:
                self.buy(self.exchange, self.trading_pair, order.amount, order.order_type, order.price)
            else:
                self.sell(self.exchange, self.trading_pair, order.amount, order.order_type, order.price)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = f"{event.trade_type.name} {event.trading_pair} order got filled at price {event.price}"

        self.logger().info(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def get_candles_with_features(self):
        candles_df = self.candles.candles_df
        candles_df.ta.natr(length=self.candles_length, scalar=1, append=True)
        candles_df["bid_spread_bps"] = candles_df[f"NATR_{self.candles_length}"] * self.bid_spread_scalar * 10000
        candles_df["ask_spread_bps"] = candles_df[f"NATR_{self.candles_length}"] * self.ask_spread_scalar * 10000
        candles_df.ta.rsi(length=self.candles_length, append=True)
        return candles_df

    def update_multipliers(self):
        candles_df = self.get_candles_with_features()
        self.bid_spread = candles_df[f"NATR_{self.candles_length}"].iloc[-1] * self.bid_spread_scalar
        self.ask_spread = candles_df[f"NATR_{self.candles_length}"].iloc[-1] * self.ask_spread_scalar

        # Trend Shift
        rsi = candles_df[f"RSI_{self.candles_length}"].iloc[-1]
        self.price_multiplier: float = (rsi - 50) / 50 * self.max_shift_spread * self.trend_scalar

        # Define shifted reference price
        self.orig_price: Decimal = self.connector.get_price_by_type(self.trading_pair, self.price_source)
        self.reference_price = self.orig_price * Decimal(1 + self.price_multiplier)

    def format_status(self) -> str:
        """
        Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self.get_market_trading_pair_tuples()))

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        ref_price: Decimal = self.connector.get_price_by_type(self.trading_pair, self.price_source)

        best_bid = self.connector.get_price(self.trading_pair, False)
        best_bid_spread = (ref_price - best_bid) / ref_price

        best_ask = self.connector.get_price(self.trading_pair, True)
        best_ask_spread = (best_ask - ref_price) / ref_price

        lines.extend(["", "  Spreads:"])
        lines.extend([f"  Ask Spread (bps): {self.ask_spread * 10000:.4f} | Best Ask Spread (bps): {best_ask_spread * 10000:.4f}"])
        lines.extend([f"  Bid Spread (bps): {self.bid_spread * 10000:.4f} | Best Bid Spread (bps): {best_bid_spread * 10000:.4f}"])

        trend_price_shift = Decimal(self.price_multiplier) * self.reference_price
        lines.extend(["", "  Price Shifts:"])
        lines.extend([f"  Trend Scalar: {self.trend_scalar:.1f} | Trend Multiplier (bps): {self.price_multiplier * 10000:.4f} | Trend Price Shift: {trend_price_shift:.4f}"])
        lines.extend([f"  Orig Price: {self.orig_price:.4f} | Reference Price: {self.reference_price:.4f}"])

        candles_df = self.get_candles_with_features()
        lines.extend(["", f"  Candles: {self.candles.name} | Interval: {self.candles.interval}", ""])
        lines.extend(["    " + line for line in candles_df.tail(10).iloc[::-1].to_string(index=False).split("\n")])

        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
