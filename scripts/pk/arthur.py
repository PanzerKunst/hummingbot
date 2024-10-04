import os
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionMode, TradeType, PositionAction, PriceType
from hummingbot.core.event.events import SellOrderCreatedEvent, BuyOrderCreatedEvent, OrderFilledEvent
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase, StrategyV2Base
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig, PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from scripts.utility.tracked_order_details import TrackedOrderDetails

from scripts.pk.pk_strategy import PkStrategy


class ArthurConfig(StrategyV2ConfigBase):
    # Standard attributes - avoid renaming
    markets: Dict[str, Set[str]] = {}
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor
    controllers_config: List[str] = []
    config_update_interval: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    connector_name: str = "okx_perpetual"
    trading_pair: str = "POPCAT-USDT"
    total_amount_quote: int = Field(5, client_data=ClientFieldData(is_updatable=True))
    leverage: int = 20
    position_mode: PositionMode = PositionMode.HEDGE

    cooldown_time_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = Field(7, client_data=ClientFieldData(is_updatable=True))

    # Triple Barrier
    stop_loss_pct: Decimal = Field(0.7, client_data=ClientFieldData(is_updatable=True))
    take_profit_pct: Decimal = Field(0.7, client_data=ClientFieldData(is_updatable=True))
    filled_order_expiration_min: int = Field(1000, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    bbands_length_for_trend: int = Field(6, client_data=ClientFieldData(is_updatable=True))
    bbands_std_dev_for_trend: Decimal = Field(2.0, client_data=ClientFieldData(is_updatable=True))
    bbands_length_for_volatility: int = Field(2, client_data=ClientFieldData(is_updatable=True))
    bbands_std_dev_for_volatility: Decimal = Field(3.0, client_data=ClientFieldData(is_updatable=True))
    high_volatility_threshold: Decimal = Field(3.0, client_data=ClientFieldData(is_updatable=True))
    rsi_length: int = Field(12, client_data=ClientFieldData(is_updatable=True))

    # Candles
    candles_connector: str = "okx_perpetual"
    candles_interval: str = "1m"
    candles_length: int = 24

    # Maker orders settings
    default_spread_pct: Decimal = Field(0.5, client_data=ClientFieldData(is_updatable=True))

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=Decimal(self.stop_loss_pct / 100),
            take_profit=Decimal(self.take_profit_pct / 100),
            time_limit=self.filled_order_expiration_min * 60,
            open_order_type=OrderType.LIMIT,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,  # Only market orders are supported for time_limit and stop_loss
            time_limit_order_type=OrderType.MARKET  # Only market orders are supported for time_limit and stop_loss
        )


# Generate config file: create --script-config arthur.arthur
# Start the bot: start --script arthur.arthur.py --conf conf_arthur_POPCAT.yml
# Quickstart script: -p=a -f arthur.arthur.py -c conf_arthur_POPCAT.yml


class Arthur(StrategyV2Base):
    @classmethod
    def init_markets(cls, config: ArthurConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ArthurConfig):
        super().__init__(connectors, config)
        self.config = config

        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.trading_pair,
                interval=config.candles_interval,
                max_records=config.candles_length
            ))

        self.pk_strat = PkStrategy(self.logger())

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def update_processed_data(self):
        candles_config = self.config.candles_config[0]

        candles_df = self.market_data_provider.get_candles_df(connector_name=candles_config.connector,
                                                              trading_pair=candles_config.trading_pair,
                                                              interval=candles_config.interval,
                                                              max_records=candles_config.max_records)
        candles_df["timestamp_iso"] = pd.to_datetime(candles_df["timestamp"], unit="s")

        bbands_for_trend = candles_df.ta.bbands(length=self.config.bbands_length_for_trend, std=self.config.bbands_std_dev_for_trend)
        candles_df["bbp"] = bbands_for_trend[f"BBP_{self.config.bbands_length_for_trend}_{self.config.bbands_std_dev_for_trend}"]
        candles_df["normalized_bbp"] = candles_df["bbp"].apply(self.normalize_bbp)

        bbands_for_volatility = candles_df.ta.bbands(length=self.config.bbands_length_for_volatility, std=self.config.bbands_std_dev_for_volatility)
        candles_df["bbb_for_volatility"] = bbands_for_volatility[f"BBB_{self.config.bbands_length_for_volatility}_{self.config.bbands_std_dev_for_volatility}"]

        rsi = candles_df.ta.rsi(length=self.config.rsi_length)
        candles_df["normalized_rsi"] = rsi.apply(self.normalize_rsi)

        self.pk_strat.processed_data["features"] = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

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

                # It seems that due to order latency, doing the opposite of the intended logic works better

                if self.can_create_order(active_sell_orders, TradeType.SELL):
                    sell_executor_config = self.get_executor_config(best_ask_connector_name, TradeType.SELL, best_ask_price)
                    self.create_order(sell_executor_config)

                if self.can_create_order(active_buy_orders, TradeType.BUY):
                    buy_executor_config = self.get_executor_config(best_bid_connector_name, TradeType.BUY, best_bid_price)
                    self.create_order(buy_executor_config)

                # TODO: try alternative with LIMIT orders at mid-price




        mid_price = self.get_mid_price()

        unfilled_sell_executors, unfilled_buy_executors = self.get_unfilled_executors_by_side()

        if self.can_create_order(unfilled_sell_executors, TradeType.SELL):
            sell_price = self.adjust_sell_price(mid_price)
            sell_executor_config = self.get_executor_config(TradeType.SELL, sell_price)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=sell_executor_config))

        if self.can_create_order(unfilled_buy_executors, TradeType.BUY):
            buy_price = self.adjust_buy_price(mid_price)
            buy_executor_config = self.get_executor_config(TradeType.BUY, buy_price)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=buy_executor_config))









        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()

        if len(active_sell_orders) > 0 and len(active_buy_orders) == 0:
            active_sell_order = active_sell_orders[0]

            # TODO: remove
            self.logger().error(f"ERROR: len(active_sell_orders) > 0 and len(active_buy_orders) == 0 | active_sell_order.filled_at: {active_sell_order.filled_at}")

            if active_sell_order.filled_at:
                mid_price_where_shorting = self.get_mid_price_custom(active_sell_order.connector_name)
                self.close_tracked_order(active_sell_order, mid_price_where_shorting)

        if len(active_buy_orders) > 0 and len(active_sell_orders) == 0:
            active_buy_order = active_buy_orders[0]

            # TODO: remove
            self.logger().error(f"ERROR: len(active_buy_orders) > 0 and len(active_sell_orders) == 0 | active_buy_order.filled_at: {active_buy_order.filled_at}")

            if active_buy_order.filled_at:
                mid_price_where_longing = self.get_mid_price_custom(active_buy_order.connector_name)
                self.close_tracked_order(active_buy_order, mid_price_where_longing)

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

        return []  # Always return []

    def format_status(self) -> str:
        original_status = super().format_status()
        custom_status = []

        if self.ready_to_trade:
            features_df = self.processed_data.get("features", pd.DataFrame())

            if not features_df.empty:
                columns_to_display = [
                    "timestamp_iso",
                    "close",
                    "normalized_bbp",
                    "bbb_for_volatility",
                    "normalized_rsi"
                ]

                custom_status.append(format_df_for_printout(features_df[columns_to_display].tail(self.config.bbands_length_for_trend), table_format="psql", ))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_order(self, active_tracked_orders: List[TrackedOrderDetails], side: TradeType) -> bool:
        if self.get_position_quote_amount(side) == 0 or len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL and self.is_a_sell_order_being_created:
            self.logger().info(f"Another SELL order is being created, avoiding a duplicate")
            return False

        if side == TradeType.BUY and self.is_a_buy_order_being_created:
            self.logger().info(f"Another BUY order is being created, avoiding a duplicate")
            return False

        last_canceled_order = self.find_last_cancelled_tracked_order()

        if not last_canceled_order:
            return True

        if last_canceled_order.cancelled_at + self.config.cooldown_time_min * 60 > self.market_data_provider.time():
            self.logger().info("Cooldown not passed yet")
            return False

        return True

    def did_create_sell_order(self, created_event: SellOrderCreatedEvent):
        self.pk_strat.did_create_sell_order(created_event)

    def did_create_buy_order(self, created_event: BuyOrderCreatedEvent):
        self.pk_strat.did_create_buy_order(created_event)

    def did_fill_order(self, filled_event: OrderFilledEvent):
        self.pk_strat.did_fill_order(filled_event)

    def get_mid_price(self):
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)

    def get_position_quote_amount(self, side: TradeType) -> Decimal:
        amount_quote = Decimal(self.config.total_amount_quote)

        # If amount_quote = 100 USDT with leverage 20x, the quote position should be 500
        position_quote_amount = amount_quote * self.config.leverage / 4

        if side == TradeType.SELL:
            position_quote_amount = position_quote_amount * Decimal(0.67)  # Less, because closing a Short position on SL costs significantly more

        return position_quote_amount

    def get_best_ask(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestAsk)

    def get_best_bid(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestBid)

    def _get_best_ask_or_bid(self, price_type: PriceType) -> Decimal:
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, price_type)

    def get_executor_config(self, side: TradeType, ref_price: Decimal) -> PositionExecutorConfig:
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=side,
            entry_price=ref_price,
            amount=self.get_position_quote_amount(side) / ref_price,
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

    def get_unfilled_tracked_orders_by_side(self) -> Tuple[List[TrackedOrderDetails], List[TrackedOrderDetails]]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()
        unfilled_sell_orders = [order for order in active_sell_orders if not order.filled_at]
        unfilled_buy_orders = [order for order in active_buy_orders if not order.filled_at]
        return unfilled_sell_orders, unfilled_buy_orders

    def create_order(self, executor_config: PositionExecutorConfig):
        connector_name = executor_config.connector_name
        trading_pair = executor_config.trading_pair
        amount = executor_config.amount
        entry_price = executor_config.entry_price

        if executor_config.side == TradeType.SELL:
            self.is_a_sell_order_being_created = True

            order_id = self.sell(connector_name, trading_pair, amount, OrderType.MARKET, entry_price)

            self.tracked_orders.append(TrackedOrderDetails(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.SELL,
                order_id=order_id,
                position=PositionAction.OPEN.value,
                amount=amount
            ))

            self.is_a_sell_order_being_created = False

        else:
            self.is_a_buy_order_being_created = True

            order_id = self.buy(connector_name, trading_pair, amount, OrderType.MARKET, entry_price)

            self.tracked_orders.append(TrackedOrderDetails(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.BUY,
                order_id=order_id,
                position=PositionAction.OPEN.value,
                amount = amount
            ))

            self.is_a_buy_order_being_created = False

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
            self.sell(
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

    @staticmethod
    def normalize_bbp(bbp: float) -> Decimal:
        return Decimal(bbp - 0.5)

    @staticmethod
    def normalize_rsi(rsi: float) -> Decimal:
        return Decimal(rsi * 2 - 100)

    def get_latest_normalized_bbp(self) -> Decimal:
        bbp_series: pd.Series = self.processed_data["features"]["normalized_bbp"]
        bbp_previous_full_minute = Decimal(bbp_series.iloc[-2])
        bbp_current_incomplete_minute = Decimal(bbp_series.iloc[-1])

        return (
            max(bbp_previous_full_minute, bbp_current_incomplete_minute) if bbp_previous_full_minute > 0
            else min(bbp_previous_full_minute, bbp_current_incomplete_minute)
        )

    def get_latest_bbb(self) -> Decimal:
        bbb_series: pd.Series = self.processed_data["features"]["bbb_for_volatility"]
        bbb_previous_full_minute = Decimal(bbb_series.iloc[-2])
        bbb_current_incomplete_minute = Decimal(bbb_series.iloc[-1])
        return max(bbb_previous_full_minute, bbb_current_incomplete_minute)

    def get_avg_last_tree_bbb(self) -> Decimal:
        bbb_series: pd.Series = self.processed_data["features"]["bbb_for_volatility"]
        bbb_last_full_minute = Decimal(bbb_series.iloc[-2])
        bbb_before_that = Decimal(bbb_series.iloc[-3])
        bbb_even_before_that = Decimal(bbb_series.iloc[-4])
        return (bbb_last_full_minute + bbb_before_that + bbb_even_before_that) / 3

    def is_high_volatility(self) -> bool:
        # TODO: remove
        self.logger().info(f"is_high_volatility() | latest_bbb: {self.get_latest_bbb()}")

        return self.get_latest_bbb() > self.config.high_volatility_threshold

    def is_still_trending_up(self) -> bool:
        return self.get_latest_normalized_bbp() > -0.2

    def is_still_trending_down(self) -> bool:
        return self.get_latest_normalized_bbp() < 0.2

    def get_latest_normalized_rsi(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["features"]["normalized_rsi"]
        return Decimal(rsi_series.iloc[-1])
