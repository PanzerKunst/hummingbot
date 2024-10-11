from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig, TrailingStop
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from scripts.pk.galahad_config import GalahadConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_utils import get_take_profit_price
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
            trailing_delta=self.config.trailing_stop_close_delta_bps / 10000
        )

        return TripleBarrierConfig(
            stop_loss=Decimal(self.config.stop_loss_pct / 100),
            trailing_stop=trailing_stop,
            open_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,
            take_profit_order_type=OrderType.MARKET
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
            triple_barrier_config = self.get_triple_barrier_config(TradeType.SELL, entry_price)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_create(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
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

        rsi_series: pd.Series = self.processed_data["RSI"]
        current_rsi = Decimal(rsi_series.iloc[-1])

        if side == TradeType.SELL:
            if self.is_rsi_above_top_edge(current_rsi):
                self.logger().info(f"rsi_is_above_top_edge: {current_rsi}")
                return self.has_macdh_turned_bearish() and self.has_psar_turned_bearish()
            else:
                return self.has_macdh_turned_bearish() and self.has_psar_turned_bearish() and self.has_price_recently_dropped()

        if self.is_rsi_below_bottom_edge(current_rsi):
            self.logger().info(f"rsi_is_below_bottom_edge: {current_rsi}")
            return self.has_macdh_turned_bullish() and self.has_psar_turned_bullish()
        else:
            return self.has_macdh_turned_bullish() and self.has_psar_turned_bullish() and self.has_price_recently_climbed()

    #
    # Custom functions specific to this controller
    #

    def is_rsi_above_top_edge(self, rsi: Decimal) -> bool:
        return rsi > self.config.rsi_top_edge

    def is_rsi_below_bottom_edge(self, rsi: Decimal) -> bool:
        return rsi < self.config.rsi_bottom_edge

    def has_macdh_turned_bullish(self) -> bool:
        macdh_series: pd.Series = self.processed_data["MACDh"]
        macd_latest_complete_candle = Decimal(macdh_series.iloc[-2])
        macd_1candle_before = Decimal(macdh_series.iloc[-3])
        delta = (macd_latest_complete_candle - macd_1candle_before) / macd_latest_complete_candle

        # TODO: remove
        if macd_1candle_before < 0 and macd_latest_complete_candle > 0:
            self.logger().info(f"has_macdh_turned_bullish | delta:{delta}")

        return macd_1candle_before < 0 and macd_latest_complete_candle > 0

    def has_macdh_turned_bearish(self) -> bool:
        macdh_series: pd.Series = self.processed_data["MACDh"]
        macd_latest_complete_candle = Decimal(macdh_series.iloc[-2])
        macd_1candle_before = Decimal(macdh_series.iloc[-3])
        delta = (macd_1candle_before - macd_latest_complete_candle) / macd_latest_complete_candle

        # TODO: remove
        if macd_1candle_before > 0 and macd_latest_complete_candle < 0:
            self.logger().info(f"has_macdh_turned_bearish | delta:{delta}")

        return macd_1candle_before > 0 and macd_latest_complete_candle < 0

    def has_price_recently_dropped(self) -> bool:
        close_series: pd.Series = self.processed_data["close"]
        current_close = Decimal(close_series.iloc[-1])

        high_series: pd.Series = self.processed_data["high"]
        high_2candles_before = Decimal(high_series.iloc[-3])

        delta_pct = (high_2candles_before - current_close) / current_close * 100

        self.logger().info(f"has_price_recently_dropped() | delta_pct:{delta_pct}")

        return delta_pct > self.config.significant_price_change_pct

    def has_price_recently_climbed(self) -> bool:
        close_series: pd.Series = self.processed_data["close"]
        current_close = Decimal(close_series.iloc[-1])

        low_series: pd.Series = self.processed_data["low"]
        low_2candles_before = Decimal(low_series.iloc[-3])

        delta_pct = (current_close - low_2candles_before) / current_close * 100

        self.logger().info(f"has_price_recently_climbed() | delta_pct:{delta_pct}")

        return delta_pct > self.config.significant_price_change_pct

    def has_psar_turned_bullish(self) -> bool:
        psarl_series: pd.Series = self.processed_data["PSARl"]
        psarl_latest_complete_candle = Decimal(psarl_series.iloc[-2])
        psarl_1candle_before = Decimal(psarl_series.iloc[-3])

        # TODO: remove
        result = psarl_latest_complete_candle > 0 and pd.isna(psarl_1candle_before)
        if result:
            self.logger().info("psar_has_turned_bullish")

        return result

    def has_psar_turned_bearish(self) -> bool:
        psars_series: pd.Series = self.processed_data["PSARs"]
        psars_latest_complete_candle = Decimal(psars_series.iloc[-2])
        psars_1candle_before = Decimal(psars_series.iloc[-3])

        # TODO: remove
        result = psars_latest_complete_candle > 0 and pd.isna(psars_1candle_before)
        if result:
            self.logger().info("psar_has_turned_bearish")

        return result
