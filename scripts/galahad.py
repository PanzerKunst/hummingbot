from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig, TrailingStop
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from scripts.pk.galahad_config import GalahadConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_utils import compute_recent_price_delta_pct, average, get_take_profit_price
from scripts.pk.tracked_order_details import TrackedOrderDetails


# Follows MACD and Parabolic SAR signals
# Generate config file: create --script-config galahad
# Start the bot: start --script galahad.py --conf conf_galahad_MEW.yml
#                start --script galahad.py --conf conf_galahad_NEIRO.yml
# Quickstart script: -p=a -f galahad.py -c conf_galahad_NEIRO.yml


class GalahadStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: GalahadConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: GalahadConfig):
        super().__init__(connectors, config)

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

    def get_triple_barrier_config(self, side: TradeType, entry_price: Decimal) -> TripleBarrierConfig:
        trailing_stop = TrailingStop(
            activation_price=get_take_profit_price(side, entry_price, self.config.trailing_stop_activation_pct),
            trailing_delta=self.config.trailing_stop_close_delta_pct / 100
        )

        return TripleBarrierConfig(
            stop_loss=self.config.stop_loss_pct / 100,
            trailing_stop=trailing_stop,
            open_order_type=OrderType.MARKET,
            stop_loss_order_type=OrderType.MARKET
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

        psar_df = candles_df.ta.psar(af0=self.config.psar_start, af=self.config.psar_increment, max_af=self.config.psar_max)
        candles_df["PSARl"] = psar_df[f"PSARl_{self.config.psar_increment}_{self.config.psar_max}"]
        candles_df["PSARs"] = psar_df[f"PSARs_{self.config.psar_increment}_{self.config.psar_max}"]

        # candles_df.dropna(inplace=True) not done here, as PSARl and PSARs are supposed to have NaN

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
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier_config = self.get_triple_barrier_config(TradeType.SELL, entry_price)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_create(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier_config = self.get_triple_barrier_config(TradeType.BUY, entry_price)
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
                    "PSARs"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(20), table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            is_rsi_in_range = self.is_rsi_in_range_for_sell_order()

            if not is_rsi_in_range:
                return False

            self.logger().info(f"is_rsi_in_range: {is_rsi_in_range}")

            return self.has_macdh_turned_bearish() and self.is_psar_bearish() and self.is_trend_negative_enough()

        is_rsi_in_range = self.is_rsi_in_range_for_buy_order()

        if not is_rsi_in_range:
            return False

        return self.has_macdh_turned_bullish() and self.is_psar_bullish() and self.is_trend_positive_enough()

    #
    # Custom functions specific to this controller
    #

    def is_rsi_in_range_for_sell_order(self) -> bool:
        if self.get_recent_rsi_avg() > 70:
            self.logger().info("is_rsi_in_range_for_sell_order: too risky as the price has been trending up for a while")
            return False

        rsi_series: pd.Series = self.processed_data["RSI"]
        rsi_latest_complete_candle = Decimal(rsi_series.iloc[-2])

        return rsi_latest_complete_candle > self.config.trend_start_sell_latest_complete_candle_min_rsi

    def is_rsi_in_range_for_buy_order(self) -> bool:
        if self.get_recent_rsi_avg() < 30:
            self.logger().info("is_rsi_in_range_for_buy_order: too risky as the price has been trending down for a while")
            return False

        rsi_series: pd.Series = self.processed_data["RSI"]
        rsi_latest_complete_candle = Decimal(rsi_series.iloc[-2])

        return rsi_latest_complete_candle < self.config.trend_start_buy_latest_complete_candle_max_rsi

    def get_recent_rsi_avg(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["RSI"]
        rsi_3candles_before = Decimal(rsi_series.iloc[-4])
        rsi_4candles_before = Decimal(rsi_series.iloc[-5])
        rsi_5candles_before = Decimal(rsi_series.iloc[-6])
        rsi_6candles_before = Decimal(rsi_series.iloc[-7])
        rsi_7candles_before = Decimal(rsi_series.iloc[-8])

        return average(rsi_3candles_before, rsi_4candles_before, rsi_5candles_before, rsi_6candles_before, rsi_7candles_before)

    def has_macdh_turned_bullish(self) -> bool:
        macdh_series: pd.Series = self.processed_data["MACDh"]
        macd_latest_complete_candle = Decimal(macdh_series.iloc[-2])
        macd_1candle_before = Decimal(macdh_series.iloc[-3])

        return macd_latest_complete_candle > 0 and macd_1candle_before < 0

    def has_macdh_turned_bearish(self) -> bool:
        macdh_series: pd.Series = self.processed_data["MACDh"]
        macd_latest_complete_candle = Decimal(macdh_series.iloc[-2])
        macd_1candle_before = Decimal(macdh_series.iloc[-3])

        return macd_latest_complete_candle < 0 and macd_1candle_before > 0

    def is_trend_positive_enough(self) -> bool:
        close_series: pd.Series = self.processed_data["close"]
        current_close = Decimal(close_series.iloc[-1])

        high_series: pd.Series = self.processed_data["high"]
        high_latest_complete_candle = Decimal(high_series.iloc[-2])

        self.logger().info(f"is_macd_increasing_enough() | current_close:{current_close} | high_latest_complete_candle:{high_latest_complete_candle}")

        if current_close < high_latest_complete_candle:
            return False

        low_series: pd.Series = self.processed_data["low"]
        low_latest_complete_candle = Decimal(low_series.iloc[-2])
        low_1candle_before = Decimal(low_series.iloc[-3])

        delta_pct = (current_close - min(low_latest_complete_candle, low_1candle_before)) / current_close * 100
        self.logger().info(f"is_macd_increasing_enough() | delta_pct:{delta_pct}")

        return delta_pct > self.config.trend_start_price_change_threshold_pct

    def is_trend_negative_enough(self) -> bool:
        close_series: pd.Series = self.processed_data["close"]
        current_close = Decimal(close_series.iloc[-1])

        low_series: pd.Series = self.processed_data["low"]
        low_latest_complete_candle = Decimal(low_series.iloc[-2])

        self.logger().info(f"is_macd_decreasing_enough() | current_close:{current_close} | low_latest_complete_candle:{low_latest_complete_candle}")

        if current_close > low_latest_complete_candle:
            return False

        high_series: pd.Series = self.processed_data["high"]
        high_latest_complete_candle = Decimal(high_series.iloc[-2])
        high_1candle_before = Decimal(high_series.iloc[-3])

        delta_pct = (max(high_latest_complete_candle, high_1candle_before) - current_close) / current_close * 100
        self.logger().info(f"is_macd_decreasing_enough() | delta_pct:{delta_pct}")

        return delta_pct > self.config.trend_start_price_change_threshold_pct

    def is_psar_bullish(self) -> bool:
        psarl_series: pd.Series = self.processed_data["PSARl"]
        current_psarl = Decimal(psarl_series.iloc[-1])

        # TODO: remove
        result = not pd.isna(current_psarl)
        if result:
            self.logger().info("psar_is_bullish")

        return result

    def is_psar_bearish(self) -> bool:
        psars_series: pd.Series = self.processed_data["PSARs"]
        current_psars = Decimal(psars_series.iloc[-1])

        # TODO: remove
        result = not pd.isna(current_psars)
        if result:
            self.logger().info("psar_is_bearish")

        return result
