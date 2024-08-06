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


class SimplePMM1(ScriptStrategyBase):
    bid_spread_pct = 0.1
    ask_spread_pct = 0.1
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

    # Initializes candles
    candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=candles_interval,
        max_records=candles_length
    ))

    markets = {exchange: {trading_pair}}

    # Start the candles when the script starts
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()

    def on_stop(self):
        self.candles.stop()

    def on_tick(self):
        # Check if it's time to place orders
        is_time_to_create = self.create_timestamp <= datetime.now().timestamp() - self.order_refresh_time

        if is_time_to_create:
            self.cancel_all_orders()
            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = datetime.now().timestamp()

    def cancel_all_orders(self):
        active_orders: List[LimitOrder] = self.get_active_orders(self.exchange)
        connector = self.connectors[self.exchange]
        connector.batch_order_cancel(active_orders)

    def create_proposal(self) -> List[OrderCandidate]:
        connector = self.connectors[self.exchange]

        ref_price = connector.get_price_by_type(self.trading_pair, self.price_source)
        buy_price = ref_price * Decimal(1 - self.bid_spread_pct / 100)
        sell_price = ref_price * Decimal(1 + self.ask_spread_pct / 100)

        buy_order = OrderCandidate(
            self.trading_pair,
            True,
            OrderType.LIMIT,
            TradeType.BUY,
            connector.quantize_order_amount(self.trading_pair, Decimal(self.order_amount)),
            connector.quantize_order_price(self.trading_pair, buy_price)
        )

        sell_order = OrderCandidate(
            self.trading_pair,
            True,
            OrderType.LIMIT,
            TradeType.SELL,
            connector.quantize_order_amount(self.trading_pair, Decimal(self.order_amount)),
            connector.quantize_order_price(self.trading_pair, sell_price)
        )

        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        connector = self.connectors[self.exchange]

        return connector.budget_checker.adjust_candidates(proposal, True)

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
        return self.candles.candles_df

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

        candles_df = self.get_candles_with_features()
        lines.extend([f"  Candles: {self.candles.name} | Interval: {self.candles.interval}", ""])
        lines.extend(["    " + line for line in candles_df.to_string(index=False).split("\n")])

        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
