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
from scripts.pk.galahad_config import GalahadConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_utils import compute_recent_price_delta_pct
from scripts.pk.tracked_order_details import TrackedOrderDetails


# Follows MACD and Parabolic SAR signals
# Generate config file: create --script-config galahad
# Start the bot: start --script galahad.py --conf conf_galahad_NEIRO.yml
# Quickstart script: -p=a -f galahad.py -c conf_galahad_NEIRO.yml


class GalahadStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: GalahadConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: GalahadConfig):
        super().__init__(connectors, config)

        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_pair,
                interval=config.candles_interval,
                max_records=config.candles_length
            ))

        self.processed_data = pd.DataFrame()

        self.nb_seconds_macd_turned_positive = 0
        self.nb_seconds_macd_turned_negative = 0

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
            stop_loss_order_type=OrderType.MARKET,
            take_profit_order_type=OrderType.MARKET,  # TODO: LIMIT
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

        macd_df = candles_df.ta.macd(fast=self.config.macd_short, slow=self.config.macd_long, signal=self.config.macd_signal)
        candles_df["MACD"] = macd_df[f"MACD_{self.config.macd_short}_{self.config.macd_long}_{self.config.macd_signal}"]
        candles_df["MACDs"] = macd_df[f"MACDs_{self.config.macd_short}_{self.config.macd_long}_{self.config.macd_signal}"]
        candles_df["MACDh"] = macd_df[f"MACDh_{self.config.macd_short}_{self.config.macd_long}_{self.config.macd_signal}"]

        psar_df = candles_df.ta.psar(af=self.config.psar_af, max_af=self.config.psar_max_af)
        candles_df["PSARl"] = psar_df[f"PSARl_{self.config.psar_af}_{self.config.psar_max_af}"]
        candles_df["PSARs"] = psar_df[f"PSARs_{self.config.psar_af}_{self.config.psar_max_af}"]

        bbands_df = candles_df.ta.bbands(length=self.config.bbands_length, std=self.config.bbands_std_dev)
        candles_df["BBB"] = bbands_df[f"BBB_{self.config.bbands_length}_{self.config.bbands_std_dev}"]

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

        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp()
            triple_barrier_config = self.get_triple_barrier_config(sl_tp_pct)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_create(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp()
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
                    "RSI",
                    "MACDh",
                    "PSARl",
                    "PSARs",
                    "BBB"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(self.config.candles_length), table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        rsi_series: pd.Series = self.processed_data["RSI"]
        current_rsi = Decimal(rsi_series.iloc[-1])

        if side == TradeType.SELL:
            if self.is_rsi_below_bottom_edge(current_rsi):
                return False

            return self.has_macdh_turned_bearish() and self.has_psar_turned_bearish() and self.is_volatile_enough()

        if self.is_rsi_above_top_edge(current_rsi):
            return False

        return self.has_macdh_turned_bullish() and self.has_psar_turned_bullish() and self.is_volatile_enough()

    #
    # Custom functions specific to this controller
    #

    def is_rsi_above_top_edge(self, rsi: Decimal) -> bool:
        return rsi > self.config.rsi_top_edge

    def is_rsi_below_bottom_edge(self, rsi: Decimal) -> bool:
        return rsi < self.config.rsi_bottom_edge

    def has_macdh_turned_bullish(self) -> bool:
        macdh_series: pd.Series = self.processed_data["MACDh"]
        current_macd = Decimal(macdh_series.iloc[-1])
        macd_latest_complete_candle = Decimal(macdh_series.iloc[-2])

        has_macdh_turned_positive = current_macd > 0 and macd_latest_complete_candle < 0

        if not has_macdh_turned_positive and self.nb_seconds_macd_turned_positive > 0:
            self.logger().info("Reseting nb_seconds_macd_turned_positive to zero")
            self.nb_seconds_macd_turned_positive = 0

        if has_macdh_turned_positive:
            self.nb_seconds_macd_turned_positive += 1
            self.logger().info(f"Incremented nb_seconds_macd_turned_positive to {self.nb_seconds_macd_turned_positive}")

        # TODO: remove
        if self.nb_seconds_macd_turned_positive > 9 and self.is_macd_increasing_enough(current_macd, macd_latest_complete_candle):
            delta = (current_macd - macd_latest_complete_candle) / abs(current_macd)
            self.logger().info(f"has_macdh_turned_bullish | delta:{delta}")

        return has_macdh_turned_positive and self.is_macd_increasing_enough(current_macd, macd_latest_complete_candle)

    def has_macdh_turned_bearish(self) -> bool:
        macdh_series: pd.Series = self.processed_data["MACDh"]
        current_macd = Decimal(macdh_series.iloc[-1])
        macd_latest_complete_candle = Decimal(macdh_series.iloc[-2])

        has_macdh_turned_negative = current_macd < 0 and macd_latest_complete_candle > 0

        if not has_macdh_turned_negative and self.nb_seconds_macd_turned_negative > 0:
            self.logger().info("Reseting nb_seconds_macd_turned_negative to zero")
            self.nb_seconds_macd_turned_negative = 0

        if has_macdh_turned_negative:
            self.nb_seconds_macd_turned_negative += 1
            self.logger().info(f"Incremented nb_seconds_macd_turned_negative to {self.nb_seconds_macd_turned_negative}")

        # TODO: remove
        if self.nb_seconds_macd_turned_negative > 9 and self.is_macd_decreasing_enough(current_macd, macd_latest_complete_candle):
            delta = (macd_latest_complete_candle - current_macd) / abs(current_macd)
            self.logger().info(f"has_macdh_turned_bearish | delta:{delta}")

        return has_macdh_turned_negative and self.is_macd_decreasing_enough(current_macd, macd_latest_complete_candle)

    def is_macd_increasing_enough(self, current_macd: Decimal, macd_latest_complete_candle: Decimal) -> bool:
        delta = current_macd - macd_latest_complete_candle
        self.logger().info(f"is_macd_increasing_enough | delta:{delta}")

        if delta < 0:  # TODO
            self.logger().info("Not enough")
            return False

        close_series: pd.Series = self.processed_data["close"]
        current_close = Decimal(close_series.iloc[-1])

        high_series: pd.Series = self.processed_data["high"]
        high_latest_complete_candle = Decimal(high_series.iloc[-2])

        self.logger().info(f"is_macd_increasing_enough() | current_macd:{current_macd} | macd_latest_complete_candle:{macd_latest_complete_candle} | current_close:{current_close} | high_latest_complete_candle:{high_latest_complete_candle}")

        return current_close > high_latest_complete_candle

    def is_macd_decreasing_enough(self, current_macd: Decimal, macd_latest_complete_candle: Decimal) -> bool:
        delta = macd_latest_complete_candle - current_macd
        self.logger().info(f"is_macd_decreasing_enough | delta:{delta}")

        if delta < 0:  # TODO
            self.logger().info("Not enough")
            return False

        close_series: pd.Series = self.processed_data["close"]
        current_close = Decimal(close_series.iloc[-1])

        low_series: pd.Series = self.processed_data["low"]
        low_latest_complete_candle = Decimal(low_series.iloc[-2])

        self.logger().info(f"is_macd_decreasing_enough() | current_macd:{current_macd} | macd_latest_complete_candle:{macd_latest_complete_candle} | current_close:{current_close} | low_latest_complete_candle:{low_latest_complete_candle}")

        return current_close < low_latest_complete_candle

    def has_psar_turned_bullish(self) -> bool:
        psarl_series: pd.Series = self.processed_data["PSARl"]
        psarl_latest_complete_candle = Decimal(psarl_series.iloc[-2])
        psarl_1candle_before = Decimal(psarl_series.iloc[-3])

        # TODO: remove
        result = not pd.isna(psarl_latest_complete_candle) and pd.isna(psarl_1candle_before)
        if result:
            self.logger().info("psar_has_turned_bullish")

        return result

    def has_psar_turned_bearish(self) -> bool:
        psars_series: pd.Series = self.processed_data["PSARs"]
        psars_latest_complete_candle = Decimal(psars_series.iloc[-2])
        psars_1candle_before = Decimal(psars_series.iloc[-3])

        # TODO: remove
        result = not pd.isna(psars_latest_complete_candle) and pd.isna(psars_1candle_before)
        if result:
            self.logger().info("psar_has_turned_bearish")

        return result

    def is_volatile_enough(self) -> bool:
        delta_pct = self.compute_delta_pct()

        if delta_pct < 1:
            self.logger().info(f"Not volatile enough | delta_pct:{delta_pct}")
            return False

        bbb_series: pd.Series = self.processed_data["BBB"]
        current_bbb = Decimal(bbb_series.iloc[-1])
        bbb_latest_complete_candle = Decimal(bbb_series.iloc[-2])
        bbb_1candle_before = Decimal(bbb_series.iloc[-3])
        bbb_2candles_before = Decimal(bbb_series.iloc[-4])

        max_bbb = max(bbb_latest_complete_candle, bbb_1candle_before, bbb_2candles_before)

        self.logger().info(f"is_volatile_enough | current_bbb:{current_bbb} | max_bbb:{max_bbb}")

        if current_bbb > self.config.min_bbb_instant_volatility:
            return True

        return max_bbb > self.config.min_bbb_past_volatility

    def compute_delta_pct(self) -> Decimal:
        low_series: pd.Series = self.processed_data["low"]
        high_series: pd.Series = self.processed_data["high"]

        return compute_recent_price_delta_pct(low_series, high_series, 20)

    def compute_sl_and_tp(self) -> Decimal:
        return self.compute_delta_pct() * Decimal(0.5)
