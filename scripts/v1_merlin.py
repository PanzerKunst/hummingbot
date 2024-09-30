import os
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType, PositionMode, PositionAction
from hummingbot.core.event.events import OrderFilledEvent, OrderCancelledEvent, BuyOrderCreatedEvent, SellOrderCreatedEvent
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from scripts.utility.tracked_order_details import TrackedOrderDetails


class MerlinConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []
    markets: Dict[str, Set[str]] = {}
    config_update_interval: int = Field(10, client_data=ClientFieldData(is_updatable=True))

    connectors_and_pairs: Dict[str, Tuple[str, Decimal]] = {
        # connector_name: (trading_pair, price_adjustment_for_normalization)
        # "bitget_perpetual": "WIF-USDT",  Fails to connect
        "gate_io_perpetual": ("WIF-USDT", Decimal(0)),
        # "hyperliquid_perpetual": ("WIF-USD", Decimal(-0.0025)),  Balance zero for now
        "kucoin_perpetual": ("WIF-USDT", Decimal(-0.001)),
        "okx_perpetual": ("WIF-USDT", Decimal(0))
    }

    leverage: int = 20  # Max 5 on Hyperliquid
    total_amount_quote: int = Field(5, client_data=ClientFieldData(is_updatable=True))
    cooldown_time_min: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    # unfilled_order_expiration: int = 1 TODO: try alternative with LIMIT orders at mid-price

    # Triple Barrier
    stop_loss_pct: float = Field(0.5, client_data=ClientFieldData(is_updatable=True))
    take_profit_pct: float = Field(10.0, client_data=ClientFieldData(is_updatable=True))
    filled_order_expiration_min: int = Field(1000, client_data=ClientFieldData(is_updatable=True))

    # Trading algo
    min_ask_bid_price_delta_to_open_bps: int = Field(5, client_data=ClientFieldData(is_updatable=True))  # 0.2% TODO: 20
    max_mid_price_delta_to_close_bps: int = Field(2, client_data=ClientFieldData(is_updatable=True))  # 0.02%

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=Decimal(self.stop_loss_pct / 100),
            # take_profit=Decimal(self.take_profit_pct / 100),
            time_limit=self.filled_order_expiration_min * 60,
            open_order_type=OrderType.MARKET,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,  # Only market orders are supported for time_limit and stop_loss
            time_limit_order_type=OrderType.MARKET  # Only market orders are supported for time_limit and stop_loss
        )


# Generate config file: create --script-config v1_merlin
# Start the bot: start --script v1_merlin.py --conf conf_v1_merlin_WIF.yml
# Quickstart script: -p=a -f v1_merlin.py -c conf_v1_merlin_WIF.yml


class Merlin(StrategyV2Base):
    @classmethod
    def init_markets(cls, config: MerlinConfig):
        markets = {}
        for connector_name, (trading_pair, _) in config.connectors_and_pairs.items():
            markets[connector_name] = [trading_pair]
        cls.markets = markets

    def __init__(self, connectors: Dict[str, ConnectorBase], config: MerlinConfig):
        super().__init__(connectors, config)
        self.config = config

        self.tracked_orders: List[TrackedOrderDetails] = []

        self.best_asks_and_bids: Dict[str, Tuple[Decimal, Decimal]] = {}
        self.best_arbitrage = None

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(PositionMode.ONEWAY)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_best_asks_and_bids()

        # Loop through the connectors, and find where:
        # - Best-bid is above best-ask (best-bid - best-ask > 0)
        # - That difference is the largest

        self.best_arbitrage = self.find_best_arbitrage()

        if self.best_arbitrage:
            best_ask_connector_name, best_ask_price, best_bid_connector_name, best_bid_price = self.best_arbitrage
            delta_bps = (best_bid_price - best_ask_price) / best_bid_price * 10000

            if delta_bps > self.config.min_ask_bid_price_delta_to_open_bps:
                # If that difference is above the threshold:
                # - Make a MARKET Short order at the best-bid
                # - And a MARKET Long order at the best-ask

                active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()
                active_orders = active_sell_orders + active_buy_orders

                # TODO: remove
                self.logger().info(f"{delta_bps:.2f} > self.config.min_ask_bid_price_delta_to_open_bps")
                self.logger().info(f"{best_ask_connector_name}, {best_ask_price}, {best_bid_connector_name}, {best_bid_price}")
                self.logger().info(f"len(active_orders):{len(active_orders)}")

                if self.can_create_order(active_sell_orders):
                    sell_executor_config = self.get_executor_config(best_bid_connector_name, TradeType.SELL, best_bid_price)
                    self.create_order(sell_executor_config)

                if self.can_create_order(active_buy_orders):
                    buy_executor_config = self.get_executor_config(best_ask_connector_name, TradeType.BUY, best_ask_price)
                    self.create_order(buy_executor_config)

                # TODO: try alternative with LIMIT orders at mid-price

        return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()

        if len(active_sell_orders) > 0 and len(active_buy_orders) == 0:
            active_sell_order = active_sell_orders[0]

            # TODO: remove
            self.logger().error(f"ERROR: len(active_sell_orders) > 0 and len(active_buy_orders) == 0 | active_sell_order.filled_at: {active_sell_order.filled_at}")

            if active_sell_order.filled_at:
                self.close_tracked_order(active_sell_order)

        if len(active_buy_orders) > 0 and len(active_sell_orders) == 0:
            active_buy_order = active_buy_orders[0]

            # TODO: remove
            self.logger().error(f"ERROR: len(active_buy_orders) > 0 and len(active_sell_orders) == 0 | active_buy_order.filled_at: {active_buy_order.filled_at}")

            if active_buy_order.filled_at:
                self.close_tracked_order(active_buy_order)

        if len(active_sell_orders) > 0 and len(active_buy_orders) > 0:
            active_sell_order = active_sell_orders[0]
            active_buy_order = active_buy_orders[0]

            # If mid-prices are too close to each-other, we close both positions at MARKET price
            mid_price_where_shorting = self.get_mid_price_custom(active_sell_order.connector_name)
            mid_price_where_longing = self.get_mid_price_custom(active_buy_order.connector_name)
            mid_price_delta_bps = (mid_price_where_shorting - mid_price_where_longing) / mid_price_where_shorting * 10000

            self.logger().info(f"mid_price_delta_bps for closing: {mid_price_delta_bps:.2f}")

            if mid_price_delta_bps < self.config.max_mid_price_delta_to_close_bps:
                self.close_tracked_order(active_sell_order, mid_price_where_shorting)
                self.close_tracked_order(active_buy_order, mid_price_where_longing)

                # TODO: remove
                self.logger().info(f"Canceled both orders | self.tracked_orders: {self.tracked_orders}")

        return []

    def format_status(self) -> str:
        original_status = super().format_status()
        custom_status = []

        if self.ready_to_trade:
            custom_status.append("\nOrder books:")
            order_books_df = pd.DataFrame()

            for connector_name, ask_and_bid in self.best_asks_and_bids.items():
                best_ask, best_bid = ask_and_bid
                row = pd.DataFrame([[connector_name, best_ask, best_bid]])
                order_books_df = pd.concat([order_books_df, row], ignore_index=True)

            if not order_books_df.empty:
                order_books_df.columns = ["Connector", "Best ask", "Best bid"]
                custom_status.append(format_df_for_printout(df=order_books_df, table_format="psql"))

            custom_status.append("\nBest arbitrage:")

            if self.best_arbitrage:
                best_ask_connector_name, best_ask_price, best_bid_connector_name, best_bid_price = self.best_arbitrage
                delta_bps = (best_bid_price - best_ask_price) / best_bid_price * 10000
                custom_status.append(f"Short {best_bid_connector_name} at {best_bid_price} | Long {best_ask_connector_name} at {best_ask_price} | delta_bps:{delta_bps:.2f}")
            else:
                custom_status.append("None")

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_order(self, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if self.get_position_quote_amount() == 0 or len(active_tracked_orders) > 0:
            return False

        last_canceled_order = self.find_last_cancelled_tracked_order()

        if not last_canceled_order:
            return True

        if last_canceled_order.cancelled_at + self.config.cooldown_time_min * 60 > self.market_data_provider.time():
            self.logger().info("Cooldown not passed yet")
            return False

        return True

    def did_create_sell_order(self, created_event: SellOrderCreatedEvent):
        position = created_event.position

        if not position or position != PositionAction.OPEN.value:
            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == created_event.order_id:
                tracked_order.exchange_order_id = created_event.exchange_order_id,
                tracked_order.created_at = created_event.creation_timestamp
                break

        # TODO: remove
        self.logger().info(f"did_create_sell_order | self.tracked_orders: {self.tracked_orders}")

    def did_create_buy_order(self, created_event: BuyOrderCreatedEvent):
        position = created_event.position

        if not position or position != PositionAction.OPEN.value:
            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == created_event.order_id:
                tracked_order.exchange_order_id = created_event.exchange_order_id,
                tracked_order.created_at = created_event.creation_timestamp
                break

        # TODO: remove
        self.logger().info(f"did_create_buy_order | self.tracked_orders: {self.tracked_orders}")

    def did_fill_order(self, filled_event: OrderFilledEvent):
        position = filled_event.position

        if not position or position != PositionAction.OPEN.value:
            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == filled_event.order_id:
                tracked_order.filled_at = filled_event.timestamp
                break

        # TODO: remove
        self.logger().info(f"did_fill_order | self.tracked_orders: {self.tracked_orders}")

    # def get_mid_price(self, connector_name):
    #     trading_pair, _ = self.config.connectors_and_pairs.get(connector_name)
    #     return self.market_data_provider.get_price_by_type(connector_name, trading_pair, PriceType.MidPrice)

    def get_position_quote_amount(self) -> Decimal:
        # If balance = 100 USDT with leverage 20x, the quote position should be 500
        return Decimal(self.config.total_amount_quote * self.config.leverage / 4)

    def get_best_ask(self, connector_name: str) -> Decimal:
        return self._get_best_ask_or_bid(connector_name, PriceType.BestAsk)

    def get_best_bid(self, connector_name: str) -> Decimal:
        return self._get_best_ask_or_bid(connector_name, PriceType.BestBid)

    def _get_best_ask_or_bid(self, connector_name: str, price_type: PriceType) -> Decimal:
        trading_pair, _ = self.config.connectors_and_pairs.get(connector_name)
        return self.market_data_provider.get_price_by_type(connector_name, trading_pair, price_type)

    def get_executor_config(self, connector_name: str, side: TradeType, ref_price: Decimal) -> PositionExecutorConfig:
        trading_pair, _ = self.config.connectors_and_pairs.get(connector_name)

        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=connector_name,
            trading_pair=trading_pair,
            side=side,
            entry_price=ref_price,
            amount=self.get_position_quote_amount() / ref_price,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage
        )

    def find_tracked_order_of_id(self, order_id: str) -> Optional[TrackedOrderDetails]:
        orders_of_that_id = [order for order in self.tracked_orders if order.order_id == order_id]
        return None if len(orders_of_that_id) == 0 else orders_of_that_id[0]

    def find_last_cancelled_tracked_order(self) -> Optional[TrackedOrderDetails]:
        cancelled_orders = [order for order in self.tracked_orders if order.cancelled_at]

        if len(cancelled_orders) == 0:
            return None

        return max(cancelled_orders, key=lambda order: order.cancelled_at)

    def get_active_tracked_orders(self) -> List[TrackedOrderDetails]:
        return [order for order in self.tracked_orders if order.created_at and not order.cancelled_at]

    def get_active_tracked_orders_by_side(self) -> Tuple[List[TrackedOrderDetails], List[TrackedOrderDetails]]:
        active_orders = self.get_active_tracked_orders()
        active_sell_orders = [order for order in active_orders if order.side == TradeType.SELL]
        active_buy_orders = [order for order in active_orders if order.side == TradeType.BUY]
        return active_sell_orders, active_buy_orders

    def create_order(self, executor_config: PositionExecutorConfig):
        connector_name = executor_config.connector_name
        trading_pair = executor_config.trading_pair
        amount = executor_config.amount
        entry_price = executor_config.entry_price

        if executor_config.side == TradeType.SELL:
            order_id = self.sell(connector_name, trading_pair, amount, OrderType.MARKET, entry_price)

            self.tracked_orders.append(TrackedOrderDetails(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.SELL,
                order_id=order_id,
                position=PositionAction.OPEN.value,
                amount=amount
            ))

        else:
            order_id = self.buy(connector_name, trading_pair, amount, OrderType.MARKET, entry_price)

            self.tracked_orders.append(TrackedOrderDetails(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.BUY,
                order_id=order_id,
                position=PositionAction.OPEN.value,
                amount = amount
            ))

        # TODO: remove
        self.logger().info(f"create_order | self.tracked_orders: {self.tracked_orders}")

    # `self.cancel()` only works for unfilled orders
    def close_tracked_order(self, tracked_order: TrackedOrderDetails, current_price: Decimal):
        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair
        amount = tracked_order.amount

        # TODO: remove
        self.logger().info(f"cancel_order | tracked_order: {tracked_order}")

        if tracked_order.side == TradeType.SELL:
            self.buy(
                connector_name,
                trading_pair,
                amount,
                OrderType.MARKET,
                current_price,
                PositionAction.CLOSE
            )
        else:
            self.buy(
                connector_name,
                trading_pair,
                amount,
                OrderType.MARKET,
                current_price,
                PositionAction.CLOSE
            )

        for order in self.tracked_orders:
            if order.order_id == tracked_order.order_id:
                order.cancelled_at = self.market_data_provider.time()
                break

    #
    # Custom functions specific to this controller
    #

    def update_best_asks_and_bids(self):
        for connector_name, (_, adjustment) in self.config.connectors_and_pairs.items():
            best_ask = self.get_best_ask(connector_name) + adjustment
            best_bid = self.get_best_bid(connector_name) + adjustment
            self.best_asks_and_bids[connector_name] = best_ask, best_bid

    def find_best_arbitrage(self) -> Optional[Tuple[str, Decimal, str, Decimal]]:
        best_spread = 0
        best_ask_connector_name: Optional[str] = None
        best_bid_connector_name: Optional[str] = None
        best_ask_price = 0
        best_bid_price = 0

        for ask_connector_name, _ in self.best_asks_and_bids.items():
            for bid_connector_name, _ in self.best_asks_and_bids.items():
                if ask_connector_name != bid_connector_name:
                    best_ask, _ = self.best_asks_and_bids.get(ask_connector_name)
                    _, best_bid = self.best_asks_and_bids.get(bid_connector_name)
                    spread = best_bid - best_ask

                    if spread > best_spread:
                        best_spread = spread
                        best_ask_connector_name = ask_connector_name
                        best_bid_connector_name = bid_connector_name
                        best_ask_price = best_ask
                        best_bid_price = best_bid

        if best_ask_connector_name and best_bid_connector_name:
            return best_ask_connector_name, best_ask_price, best_bid_connector_name, best_bid_price
        else:
            return None  # No arbitrage opportunity found

    def get_mid_price_custom(self, connector_name: str) -> Decimal:
        best_ask, best_bid = self.best_asks_and_bids.get(connector_name)
        return (best_ask + best_bid) * Decimal(0.5)
