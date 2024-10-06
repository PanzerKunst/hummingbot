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
        self.latest_normalized_rsis = deque(maxlen=config.nb_seconds_to_calculate_end_of_rsi_trend)

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def get_triple_barrier_config(self) -> TripleBarrierConfig:
        sl_tp_pct = self.compute_sl_and_tp()

        # TODO: remove
        self.logger().info(f"get_triple_barrier_config() | sl_tp_pct: {sl_tp_pct}")

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

        rsi = candles_df.ta.rsi(length=self.config.rsi_length)
        candles_df["normalized_rsi"] = rsi.apply(self.normalize_rsi)

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

        self.save_latest_normalized_rsi()

        # if self.is_high_volatility():
        #     return []

        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()

        if self.can_create(TradeType.SELL, active_sell_orders):
            entry_price: Decimal = self.get_mid_price() * Decimal(1 + self.config.delta_with_mid_price_bps / 10000)
            self.create_order(TradeType.SELL, entry_price)

        if self.can_create(TradeType.BUY, active_buy_orders):
            entry_price: Decimal = self.get_mid_price() * Decimal(1 - self.config.delta_with_mid_price_bps / 10000)
            self.create_order(TradeType.BUY, entry_price)

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        # if self.is_high_volatility():
        #     self.logger().info(f"##### is_high_volatility -> Stopping unfilled executors #####")
        #     unfilled_sell_orders, unfilled_buy_orders = self.get_unfilled_tracked_orders_by_side()
        #
        #     for unfilled_order in unfilled_sell_orders + unfilled_buy_orders:
        #         self.cancel_tracked_order(unfilled_order)

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
                    "normalized_rsi"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(self.config.rsi_length), table_format="psql"))

            latest_normalized_rsis = list(self.latest_normalized_rsis)
            latest_normalized_rsis_df = pd.DataFrame(latest_normalized_rsis, columns=["Normalized RSI"])
            custom_status.append(format_df_for_printout(latest_normalized_rsis_df, table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if (
            side == TradeType.SELL and
            self.get_latest_normalized_rsi() > self.normalize_rsi(self.config.rsi_threshold_sell) and
            self.has_rsi_stopped_increasing()
        ):
            return True

        if (
            side == TradeType.BUY and
            self.get_latest_normalized_rsi() < self.normalize_rsi(self.config.rsi_threshold_buy) and
            self.has_rsi_stopped_decreasing()
        ):
            return True

        return False

    #
    # Custom functions specific to this controller
    #

    @staticmethod
    def normalize_rsi(rsi: float) -> Decimal:
        return Decimal(rsi * 2 - 100)

    def get_latest_bbb(self) -> Decimal:
        bbb_series: pd.Series = self.processed_data["bbb_for_volatility"]
        bbb_current_incomplete_minute = Decimal(bbb_series.iloc[-1])
        bbb_previous_full_minute = Decimal(bbb_series.iloc[-2])
        return max(bbb_current_incomplete_minute, bbb_previous_full_minute)

    def is_high_volatility(self) -> bool:
        # TODO: remove
        self.logger().info(f"is_high_volatility() | latest_bbb: {self.get_latest_bbb()}")

        return self.get_latest_bbb() > self.config.high_volatility_threshold

    def get_latest_normalized_rsi(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["normalized_rsi"]
        return Decimal(rsi_series.iloc[-1])

    def save_latest_normalized_rsi(self):
        latest_normalized_rsi = self.get_latest_normalized_rsi()
        self.latest_normalized_rsis.append(latest_normalized_rsi)

    def has_rsi_stopped_increasing(self):
        current_rsi = self.latest_normalized_rsis[-1]
        oldest_rsi = self.latest_normalized_rsis[0]

        return current_rsi < oldest_rsi

    def has_rsi_stopped_decreasing(self):
        current_rsi = self.latest_normalized_rsis[-1]
        oldest_rsi = self.latest_normalized_rsis[0]

        return current_rsi > oldest_rsi

    def compute_sl_and_tp(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        latest_close_price = close_series.iloc[-1]
        close_price_3min_before = close_series.iloc[-4]
        delta_pct = (latest_close_price - close_price_3min_before) / latest_close_price * 100

        self.logger().info(f"compute_sl_and_tp() | latest_close_price:{latest_close_price} | close_price_3min_before:{close_price_3min_before}")

        return abs(delta_pct) / 2
