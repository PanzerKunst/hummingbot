from collections import deque
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
        self.latest_close_prices = deque(maxlen=config.trend_reversal_nb_seconds_to_calculate_end_of_trend)

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
            time_limit=self.config.filled_order_expiration_min * 60,
            open_order_type=OrderType.LIMIT,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,
            time_limit_order_type=OrderType.LIMIT
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

        bbands_for_volatility = candles_df.ta.bbands(length=self.config.bbands_length_for_volatility, std=self.config.bbands_std_dev_for_volatility)
        candles_df["bbb_for_volatility"] = bbands_for_volatility[f"BBB_{self.config.bbands_length_for_volatility}_{self.config.bbands_std_dev_for_volatility}"]

        candles_df["rsi"] = candles_df.ta.rsi(length=self.config.rsi_length)
        candles_df["normalized_rsi"] = candles_df["rsi"].apply(self.normalize_rsi)

        self.processed_data = candles_df

    # TODO: Currently bugged
    # def update_indicators(self, df: pd.DataFrame):
    #     rows_to_add = []
    #
    #     for _, row in df.iterrows():
    #         timestamp = row["timestamp"]
    #         bbb_for_volatility = row["bbb_for_volatility"]
    #         normalized_rsi = row["normalized_rsi"]
    #
    #         if pd.notna(bbb_for_volatility) and pd.notna(normalized_rsi):
    #             if self._get_indicators_for_timestamp(timestamp) is None:  # Do not refactor to `if not [...]`
    #                 rows_to_add.append(row)
    #
    #     if len(rows_to_add) > 0:
    #         new_rows = pd.DataFrame(rows_to_add)
    #         self.processed_data = pd.concat([self.processed_data, new_rows], ignore_index=True)
    #         self.processed_data["index"] = self.processed_data["timestamp"]
    #         self.processed_data.set_index("index", inplace=True)
    #
    # def _get_indicators_for_timestamp(self, timestamp: float) -> Optional[pd.Series]:
    #     matching_row = self.processed_data.query(f"timestamp == {timestamp}")
    #
    #     if matching_row.empty:
    #         return None
    #     else:
    #         return matching_row.iloc[0]

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.save_latest_close_price()

        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()

        if self.can_create_trend_reversal_order(TradeType.SELL, active_sell_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp_for_trend_reversal()
            triple_barrier_config = self.get_triple_barrier_config(sl_tp_pct)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)
        elif self.can_create_trend_start_order(TradeType.SELL, active_sell_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp_for_trend_start()
            triple_barrier_config = self.get_triple_barrier_config(sl_tp_pct)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_create_trend_reversal_order(TradeType.BUY, active_buy_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp_for_trend_reversal()
            triple_barrier_config = self.get_triple_barrier_config(sl_tp_pct)
            self.create_order(TradeType.BUY, entry_price, triple_barrier_config)
        elif self.can_create_trend_start_order(TradeType.BUY, active_buy_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp_for_trend_start()
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
                    "bbb_for_volatility",
                    "rsi",
                    "normalized_rsi"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(self.config.rsi_length), table_format="psql"))

            latest_close_prices = list(self.latest_close_prices)
            latest_close_prices_df = pd.DataFrame(latest_close_prices, columns=["Close price"])
            custom_status.append(format_df_for_printout(latest_close_prices_df, table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_trend_reversal_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        close_series: pd.Series = self.processed_data["close"]
        current_close_price = Decimal(close_series.iloc[-1])
        previous_close_price = Decimal(close_series.iloc[-2])
        delta_pct = (current_close_price - previous_close_price) / current_close_price * 100

        if abs(delta_pct) < self.config.trend_reversal_candle_height_threshold_pct:
            return False

        self.logger().info(f"can_create_trend_reversal_order() | abs(delta_pct): {abs(delta_pct)}")

        if side == TradeType.SELL:
            self.logger().info(f"can_create_trend_reversal_order({side}) | latest_rsi: {self.denormalize_rsi(self.get_latest_normalized_rsi())} | has_price_stopped_climbing:{self.has_price_stopped_climbing()}")

        if (
            side == TradeType.SELL and
            self.get_latest_normalized_rsi() > self.normalize_rsi(self.config.trend_reversal_rsi_threshold_sell) and
            self.has_price_stopped_climbing()
        ):
            return True

        if side == TradeType.BUY:
            self.logger().info(f"can_create_trend_reversal_order({side}) | latest_rsi: {self.denormalize_rsi(self.get_latest_normalized_rsi())} | has_price_stopped_dropping:{self.has_price_stopped_dropping()}")

        if (
            side == TradeType.BUY and
            self.get_latest_normalized_rsi() < self.normalize_rsi(self.config.trend_reversal_rsi_threshold_buy) and
            self.has_price_stopped_dropping()
        ):
            return True

        return False

    def can_create_trend_start_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        close_series: pd.Series = self.processed_data["close"]
        close_price_latest_full_minute = Decimal(close_series.iloc[-2])
        close_price_previous_minute = Decimal(close_series.iloc[-3])

        delta_pct_sell = (close_price_previous_minute - close_price_latest_full_minute) / close_price_latest_full_minute * 100
        delta_pct_buy = (close_price_latest_full_minute - close_price_previous_minute) / close_price_latest_full_minute * 100
        delta_pct = delta_pct_sell if side == TradeType.SELL else delta_pct_buy

        if delta_pct > self.config.trend_start_candle_height_threshold_pct and self.has_price_remained_stable_recently():
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

    @staticmethod
    def normalize_rsi(rsi: float) -> Decimal:
        return Decimal(rsi * 2 - 100)

    @staticmethod
    def denormalize_rsi(normalized_rsi: Decimal) -> Decimal:
        return (normalized_rsi + 100) / 2

    def get_latest_close_price(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[-1])

    def get_latest_bbb(self) -> Decimal:
        bbb_series: pd.Series = self.processed_data["bbb_for_volatility"]
        bbb_current_incomplete_minute = Decimal(bbb_series.iloc[-1])
        bbb_previous_full_minute = Decimal(bbb_series.iloc[-2])
        return max(bbb_current_incomplete_minute, bbb_previous_full_minute)

    def get_latest_normalized_rsi(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["normalized_rsi"]
        return Decimal(rsi_series.iloc[-1])

    def is_high_volatility(self) -> bool:
        # TODO: remove
        self.logger().info(f"is_high_volatility() | latest_bbb: {self.get_latest_bbb()}")

        return self.get_latest_bbb() > self.config.high_volatility_threshold

    def save_latest_close_price(self):
        latest_close_price = self.get_latest_close_price()
        self.latest_close_prices.append(latest_close_price)

    def has_price_stopped_climbing(self):
        current_price = self.latest_close_prices[-1]
        oldest_price = self.latest_close_prices[0]

        return current_price < oldest_price

    def has_price_stopped_dropping(self):
        current_price = self.latest_close_prices[-1]
        oldest_price = self.latest_close_prices[0]

        return current_price > oldest_price

    def has_price_remained_stable_recently(self):
        close_series: pd.Series = self.processed_data["close"]
        close_price_2min_before = close_series.iloc[-3]
        close_price_3min_before = close_series.iloc[-4]
        close_price_4min_before = close_series.iloc[-5]
        delta_1_pct = (close_price_2min_before - close_price_3min_before) / close_price_2min_before * 100
        delta_2_pct = (close_price_2min_before - close_price_4min_before) / close_price_2min_before * 100

        return abs(delta_1_pct) < 0.1 and abs(delta_2_pct) < 0.2

    def is_rsi_in_range_for_trend_start_order(self, side: TradeType) -> bool:
        rsi_series: pd.Series = self.processed_data["normalized_rsi"]
        current_normalized_rsi = Decimal(rsi_series.iloc[-1])
        previous_normalized_rsi = Decimal(rsi_series.iloc[-2])

        if side == TradeType.SELL:
            return (
                self.denormalize_rsi(previous_normalized_rsi) < self.config.trend_start_rsi_max_threshold_sell and
                self.denormalize_rsi(current_normalized_rsi) > self.config.trend_start_rsi_min_threshold_sell
            )

        return (
            self.denormalize_rsi(previous_normalized_rsi) > self.config.trend_start_rsi_min_threshold_buy and
            self.denormalize_rsi(current_normalized_rsi) < self.config.trend_start_rsi_max_threshold_buy
        )

    def compute_sl_and_tp_for_trend_reversal(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        latest_close_price = close_series.iloc[-1]
        close_price_3min_before = close_series.iloc[-4]
        delta_pct = (latest_close_price - close_price_3min_before) / latest_close_price * 100

        self.logger().info(f"compute_sl_and_tp_for_trend_reversal() | latest_close_price:{latest_close_price} | close_price_3min_before:{close_price_3min_before}")

        return abs(delta_pct) * 0.5

    def compute_sl_and_tp_for_trend_start(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        latest_close_price = close_series.iloc[-1]
        previous_close_price = close_series.iloc[-2]
        delta_pct = (latest_close_price - previous_close_price) / latest_close_price * 100

        self.logger().info(f"compute_sl_and_tp_for_trend_start() | latest_close_price:{latest_close_price} | previous_close_price:{previous_close_price}")

        return abs(delta_pct) * 0.75
