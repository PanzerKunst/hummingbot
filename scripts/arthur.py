from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from scripts.pk.arthur_config import ArthurConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_utils import average
from scripts.pk.tracked_order_details import TrackedOrderDetails


# Trend start and reversals, dependant on sudden price movements and RSI
# Generate config file: create --script-config arthur
# Start the bot: start --script arthur.py --conf conf_arthur_BOME.yml
#                start --script arthur.py --conf conf_arthur_BONK.yml
#                start --script arthur.py --conf conf_arthur_CAT.yml
#                start --script arthur.py --conf conf_arthur_DOGS.yml
#                start --script arthur.py --conf conf_arthur_FLOKI.yml
#                start --script arthur.py --conf conf_arthur_GOAT.yml
#                start --script arthur.py --conf conf_arthur_MEW.yml
#                start --script arthur.py --conf conf_arthur_MOODENG.yml
#                start --script arthur.py --conf conf_arthur_NEIRO.yml
#                start --script arthur.py --conf conf_arthur_NEIROETH.yml
#                start --script arthur.py --conf conf_arthur_PEOPLE.yml
#                start --script arthur.py --conf conf_arthur_POPCAT.yml
#                start --script arthur.py --conf conf_arthur_TURBO.yml
# Quickstart script: -p=a -f arthur.py -c conf_arthur_GOAT.yml


class ArthurStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ArthurConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ArthurConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()

        self.latest_price_crash_timestamp: float = 0
        self.latest_price_spike_timestamp: float = 0

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    @staticmethod
    def get_triple_barrier_config(expiration: int, open_order_type: OrderType, stop_loss_pct: Optional[Decimal] = None) -> TripleBarrierConfig:
        stop_loss = stop_loss_pct / 100 if stop_loss_pct else None

        return TripleBarrierConfig(
            stop_loss=stop_loss,
            open_order_type=open_order_type,
            time_limit=expiration
        )

    def update_processed_data(self):
        connectors = [config.connector for config in self.config.candles_config]
        candles_dataframes: List[pd.DataFrame] = []

        for i, connector in enumerate(connectors):
            candles_config = self.config.candles_config[i]

            candles_df = self.market_data_provider.get_candles_df(connector_name=candles_config.connector,
                                                                  trading_pair=candles_config.trading_pair,
                                                                  interval=candles_config.interval,
                                                                  max_records=candles_config.max_records)
            num_rows = candles_df.shape[0]

            if num_rows == 0:
                continue

            candles_df["index"] = candles_df["timestamp"]
            candles_df.set_index("index", inplace=True)

            candles_df["RSI"] = candles_df.ta.rsi(length=self.config.rsi_length)

            candles_dataframes.append(candles_df)

        merged_df = self.merge_dataframes(candles_dataframes, connectors, ["open", "close", "high", "low", "RSI"], ["volume"])
        merged_df["timestamp_iso"] = pd.to_datetime(merged_df["timestamp"], unit="s")
        merged_df.dropna(inplace=True)

        self.processed_data = merged_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()

        if self.can_create_trend_start_order(TradeType.SELL, active_sell_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier_config = self.get_triple_barrier_config(self.config.filled_trend_start_order_expiration, OrderType.MARKET)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_create_trend_start_order(TradeType.BUY, active_buy_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier_config = self.get_triple_barrier_config(self.config.filled_trend_start_order_expiration, OrderType.MARKET)
            self.create_order(TradeType.BUY, entry_price, triple_barrier_config)

        if self.can_create_trend_reversal_order(TradeType.SELL, active_sell_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier_config = self.get_triple_barrier_config(self.config.filled_trend_reversal_order_expiration, OrderType.LIMIT, self.config.trend_reversal_order_stop_loss_pct)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_create_trend_reversal_order(TradeType.BUY, active_buy_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier_config = self.get_triple_barrier_config(self.config.filled_trend_reversal_order_expiration, OrderType.LIMIT, self.config.trend_reversal_order_stop_loss_pct)
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
                    "volume",
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

        if not self.is_recent_volume_enough():
            return False

        self.logger().info("recent_volume_is_enough")

        delta_pct = self.compute_delta_pct(side)

        if delta_pct < self.config.trend_start_price_change_threshold_pct:
            return False

        self.logger().info(f"delta_pct above threshold: {delta_pct}")

        if side == TradeType.SELL:
            self.latest_price_crash_timestamp = self.get_market_data_provider_time()
            is_rsi_in_range = self.is_rsi_in_range_for_trend_start_sell_order()
            self.logger().info(f"is_rsi_in_range: {is_rsi_in_range}")
            return is_rsi_in_range

        self.latest_price_spike_timestamp = self.get_market_data_provider_time()
        is_rsi_in_range = self.is_rsi_in_range_for_trend_start_buy_order()
        self.logger().info(f"is_rsi_in_range: {is_rsi_in_range}")
        return is_rsi_in_range

    def can_create_trend_reversal_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            # During the last 12min, there was a price spike
            if self.latest_price_spike_timestamp + 12 * 60 < self.get_market_data_provider_time():
                return False

            self.logger().info(f"can_create_trend_reversal_order({side}) > There was a price spike within the last 12min")

            # During the last 7min, RSI exceeded TH
            if not self.did_rsi_recently_jump():
                return False

            self.logger().info(f"can_create_trend_reversal_order({side}) > rsi_did_recently_jump")

            # Current RSI is now back to over 30 / below 70
            if not self.has_rsi_recovered_from_jump():
                return False

            self.logger().info(f"can_create_trend_reversal_order({side}) > rsi_has_recovered_from_jump")

            return True

        if self.latest_price_crash_timestamp + 12 * 60 < self.get_market_data_provider_time():
            return False

        self.logger().info(f"can_create_trend_reversal_order({side}) > There was a price crash within the last 12min")

        if not self.did_rsi_recently_crash():
            return False

        self.logger().info(f"can_create_trend_reversal_order({side}) > rsi_did_recently_crash")

        if not self.has_rsi_recovered_from_crash():
            return False

        self.logger().info(f"can_create_trend_reversal_order({side}) > rsi_has_recovered_from_crash")

        return True

    #
    # Custom functions specific to this controller
    #

    def merge_dataframes(self, dataframes: List[pd.DataFrame], suffixes: List[str], columns_to_avg: List[str], columns_to_sum: List[str]) -> pd.DataFrame:
        if len(dataframes) != len(suffixes):
            raise ValueError("The number of dataframes must match the number of suffixes")

        # Start by merging the first two DataFrames
        merged_df = pd.merge(
            dataframes[0],
            dataframes[1],
            on="timestamp",
            suffixes=(f"_{suffixes[0]}", f"_{suffixes[1]}")
        )

        # Merge any additional DataFrames
        for i in range(2, len(dataframes)):
            df_with_suffix = self._add_suffix(dataframes[i], suffixes[i])
            merged_df = pd.merge(merged_df, df_with_suffix, on="timestamp")

        for col in columns_to_avg:
            columns_to_avg_list = [f"{col}_{suffix}" for suffix in suffixes]
            merged_df[col] = merged_df[columns_to_avg_list].mean(axis=1)

        for col in columns_to_sum:
            columns_to_sum_list = [f"{col}_{suffix}" for suffix in suffixes]
            merged_df[col] = merged_df[columns_to_sum_list].sum(axis=1)

        # Drop the original columns after averaging
        columns_to_drop = [f"{col}_{suffix}" for col in columns_to_avg+columns_to_sum for suffix in suffixes]
        merged_df = merged_df.drop(columns=columns_to_drop)

        return merged_df

    @staticmethod
    def _add_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return df.rename(columns={col: f"{col}_{suffix}" for col in df.columns if col != "timestamp"})

    def is_rsi_in_range_for_trend_start_sell_order(self) -> bool:
        if self.get_recent_rsi_avg() > 70:
            self.logger().info("is_rsi_in_range_for_trend_start_sell_order: too risky as the price has been trending up for a while")
            return False

        rsi_series: pd.Series = self.processed_data["RSI"]
        rsi_2candles_before = Decimal(rsi_series.iloc[-3])

        return rsi_2candles_before > self.config.trend_start_sell_latest_complete_candle_min_rsi

    def is_rsi_in_range_for_trend_start_buy_order(self) -> bool:
        if self.get_recent_rsi_avg() < 30:
            self.logger().info("is_rsi_in_range_for_trend_start_buy_order: too risky as the price has been trending down for a while")
            return False

        rsi_series: pd.Series = self.processed_data["RSI"]
        rsi_2candles_before = Decimal(rsi_series.iloc[-3])

        return rsi_2candles_before < self.config.trend_start_buy_latest_complete_candle_max_rsi

    def get_recent_rsi_avg(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["RSI"]
        rsi_3candles_before = Decimal(rsi_series.iloc[-4])
        rsi_4candles_before = Decimal(rsi_series.iloc[-5])
        rsi_5candles_before = Decimal(rsi_series.iloc[-6])
        rsi_6candles_before = Decimal(rsi_series.iloc[-7])
        rsi_7candles_before = Decimal(rsi_series.iloc[-8])

        return average(rsi_3candles_before, rsi_4candles_before, rsi_5candles_before, rsi_6candles_before, rsi_7candles_before)

    def did_rsi_recently_jump(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        recent_rsis = rsi_series.iloc[-8:-1]  # 7 items, last one excluded

        self.logger().info(f"did_rsi_recently_jump() | max(recent_rsis):{max(recent_rsis)}")

        return max(recent_rsis) > self.config.trend_reversal_sell_min_rsi

    def did_rsi_recently_crash(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        recent_rsis = rsi_series.iloc[-8:-1]  # 7 items, last one excluded

        self.logger().info(f"did_rsi_recently_crash() | min(recent_rsis):{min(recent_rsis)}")

        return min(recent_rsis) < self.config.trend_reversal_buy_max_rsi

    def has_rsi_recovered_from_jump(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        current_rsi = rsi_series.iloc[-1]

        # TODO: remove
        self.logger().info(f"has_rsi_recovered_from_jump() | current_rsi:{current_rsi}")

        return current_rsi < 70

    def has_rsi_recovered_from_crash(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        current_rsi = rsi_series.iloc[-1]

        # TODO: remove
        self.logger().info(f"has_rsi_recovered_from_crash() | current_rsi:{current_rsi}")

        return current_rsi > 30

    def is_recent_volume_enough(self) -> bool:
        volume_series: pd.Series = self.processed_data["volume"]
        current_volume = volume_series.iloc[-1]
        volume_latest_complete_candle = volume_series.iloc[-2]

        if current_volume < volume_latest_complete_candle:
            return False

        recent_volumes = [current_volume, volume_latest_complete_candle]
        older_volumes = volume_series.iloc[-7:-2]  # 5 items, last one excluded

        return sum(recent_volumes) > sum(older_volumes) * self.config.trend_start_volume_mul

    def compute_delta_pct(self, side: TradeType) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        current_close_price = Decimal(close_series.iloc[-1])

        high_series: pd.Series = self.processed_data["high"]
        high_latest_complete_candle = Decimal(high_series.iloc[-2])

        delta_pct_sell = (high_latest_complete_candle - current_close_price) / current_close_price * 100

        if side == TradeType.SELL:
            return delta_pct_sell

        low_series: pd.Series = self.processed_data["low"]
        low_latest_complete_candle = Decimal(low_series.iloc[-2])

        delta_pct_buy = (current_close_price - low_latest_complete_candle) / current_close_price * 100

        return delta_pct_buy
