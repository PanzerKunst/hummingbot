from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from scripts.pk.arthur_config import ArthurConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.tracked_order_details import TrackedOrderDetails


# Trend start and reversals, dependant on sudden price movements and RSI
# Generate config file: create --script-config arthur
# Start the bot: start --script arthur.py --conf conf_arthur_POPCAT.yml
# Quickstart script: -p=a -f arthur.py -c conf_arthur_POPCAT.yml


class ArthurStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ArthurConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ArthurConfig):
        super().__init__(connectors, config)

        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.trading_pair,
                interval=config.candles_interval,
                max_records=config.candles_length
            ))

        self.processed_data = pd.DataFrame()

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def get_triple_barrier_config(self, sl_tp_pct: Decimal) -> TripleBarrierConfig:
        # TODO: remove
        self.logger().info(f"get_triple_barrier_config() | sl_tp_pct:{sl_tp_pct}")

        return TripleBarrierConfig(
            stop_loss=Decimal(sl_tp_pct / 100),
            take_profit=Decimal(sl_tp_pct / 100),
            open_order_type=OrderType.LIMIT,
            take_profit_order_type=OrderType.MARKET,  # TODO: LIMIT
            stop_loss_order_type=OrderType.MARKET,
            time_limit=self.config.filled_order_expiration_min * 60
        )

    def update_processed_data(self):
        candles_config = self.config.candles_config[0]

        candles_df = self.market_data_provider.get_candles_df(connector_name=candles_config.connector,
                                                              trading_pair=candles_config.trading_pair,
                                                              interval=candles_config.interval,
                                                              max_records=candles_config.max_records)
        num_rows = candles_df.shape[0]

        if num_rows == 0:
            return

        candles_df["index"] = candles_df["timestamp"]
        candles_df.set_index("index", inplace=True)

        candles_df["timestamp_iso"] = pd.to_datetime(candles_df["timestamp"], unit="s")

        candles_df["RSI"] = candles_df.ta.rsi(length=self.config.rsi_length)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()

        if self.can_create_trend_start_order(TradeType.SELL, active_sell_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp_for_trend_start(TradeType.SELL)
            triple_barrier_config = self.get_triple_barrier_config(sl_tp_pct)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_create_trend_start_order(TradeType.BUY, active_buy_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp_for_trend_start(TradeType.BUY)
            triple_barrier_config = self.get_triple_barrier_config(sl_tp_pct)
            self.create_order(TradeType.BUY, entry_price, triple_barrier_config)

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        return []  # Always return []

    def format_status(self) -> str:
        original_status = super().format_status()
        custom_status = ["\n"]

        if self.ready_to_trade:
            if not self.processed_data.empty:
                columns_to_display = [
                    "timestamp_iso",
                    "close",
                    "high",
                    "low",
                    "RSI"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(self.config.rsi_length), table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_trend_start_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        delta_pct = self.compute_delta_pct(side)

        if delta_pct < self.config.trend_start_candle_height_threshold_pct:
            return False

        self.logger().info(f"can_create_trend_start_order({side}) | delta_pct: {delta_pct}")

        if side == TradeType.SELL:
            self.logger().info(
                f"can_create_trend_start_order({side}) | is_rsi_in_range_for_trend_start_order: {self.is_rsi_in_range_for_trend_start_order(TradeType.SELL)}")

        if side == TradeType.SELL and self.is_rsi_in_range_for_trend_start_order(TradeType.SELL):
            return True

        if side == TradeType.BUY:
            self.logger().info(
                f"can_create_trend_start_order({side}) | is_rsi_in_range_for_trend_start_order: {self.is_rsi_in_range_for_trend_start_order(TradeType.BUY)}")

        if side == TradeType.BUY and self.is_rsi_in_range_for_trend_start_order(TradeType.BUY):
            return True

        return False

    #
    # Custom functions specific to this controller
    #

    def is_rsi_in_range_for_trend_start_order(self, side: TradeType) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        rsi_last_full_candle = Decimal(rsi_series.iloc[-2])

        if side == TradeType.SELL:
            return rsi_last_full_candle > self.config.trend_start_min_rsi_sell

        return rsi_last_full_candle < self.config.trend_start_max_rsi_buy

    def compute_delta_pct(self, side: TradeType) -> Decimal:
        current_close_price = Decimal(self.processed_data["close"].iloc[-1])
        highest_price_4candles_before = Decimal(self.processed_data["high"].iloc[-5])
        lowest_price_4candles_before = Decimal(self.processed_data["low"].iloc[-5])

        delta_pct_sell = (highest_price_4candles_before - current_close_price) / current_close_price * 100
        delta_pct_buy = (current_close_price - lowest_price_4candles_before) / current_close_price * 100
        return delta_pct_sell if side == TradeType.SELL else delta_pct_buy

    def compute_sl_and_tp_for_trend_start(self, side: TradeType) -> Decimal:
        delta_pct = self.compute_delta_pct(side)
        return delta_pct * Decimal(0.7)
