from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.keltner_tf_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Generate config file: create --script-config keltner_tf
# Start the bot: start --script keltner_tf.py --conf conf_keltner_tf_GOAT.yml
# Quickstart script: -p=a -f keltner_tf.py -c conf_keltner_tf_GOAT.yml

ORDER_REF_KELTNER_TF: str = "KeltnerTF"


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()
        self.reset_tf_context()

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def get_triple_barrier(self) -> TripleBarrier:
        return TripleBarrier(
            time_limit=60
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

        sma_20_df = candles_df.ta.sma(length=20)

        atr_20_df = candles_df.ta.atr(length=20)

        kc_mult: float = 2.0
        candles_df["KC_u"] = sma_20_df + atr_20_df * kc_mult
        candles_df["KC_l"] = sma_20_df - atr_20_df * kc_mult

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        self.create_actions_proposal_trend_following()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()
        self.stop_actions_proposal_trend_following()

        return []  # Always return []

    def format_status(self) -> str:
        original_status = super().format_status()
        custom_status = ["\n"]

        if self.ready_to_trade:
            if not self.processed_data.empty:
                columns_to_display = [
                    "timestamp_iso",
                    "low",
                    "high",
                    "close",
                    "volume",
                    "KC_u",
                    "KC_l"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display], table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Keltner Trend Following start/stop action proposals
    #

    def create_actions_proposal_trend_following(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_KELTNER_TF)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_trend_following_order(TradeType.BUY, active_orders):
            triple_barrier = self.get_triple_barrier()
            self.create_order(TradeType.BUY, self.get_current_close(), triple_barrier, self.config.amount_quote, ORDER_REF_KELTNER_TF)

    def can_create_trend_following_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_KELTNER_TF, 0):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if self.has_price_crossed_upper_band():
            self.logger().info(f"can_create_trend_following_order() > Opening Keltner TF Buy at {self.get_current_close()}")
            self.reset_tf_context()
            return True

        return False

    def stop_actions_proposal_trend_following(self):
        _, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_KELTNER_TF)

        if len(filled_buy_orders) > 0:
            if self.has_price_crossed_lower_band():
                self.logger().info(f"stop_actions_proposal_trend_following() > Closing Keltner TF Buy at {self.get_current_close()}")
                self.close_filled_orders(filled_buy_orders, OrderType.LIMIT, CloseType.COMPLETED)

    #
    # Getters on `self.processed_data[]`
    #

    def get_current_close(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[-1])

    def get_current_open(self) -> Decimal:
        open_series: pd.Series = self.processed_data["open"]
        return Decimal(open_series.iloc[-1])

    def get_current_low(self) -> Decimal:
        low_series: pd.Series = self.processed_data["low"]
        return Decimal(low_series.iloc[-1])

    def get_current_high(self) -> Decimal:
        high_series: pd.Series = self.processed_data["high"]
        return Decimal(high_series.iloc[-1])

    def get_current_upper_band(self) -> Decimal:
        upper_band_series: pd.Series = self.processed_data["KC_u"]
        return Decimal(upper_band_series.iloc[-1])

    def get_current_lower_band(self) -> Decimal:
        lower_band_series: pd.Series = self.processed_data["KC_l"]
        return Decimal(lower_band_series.iloc[-1])

    #
    # Keltner TF context
    #

    def reset_tf_context(self):
        self.tf_price_over_upper_band_counter: int = 0
        self.logger().info("Keltner TF context is reset")

    #
    # Keltner TF functions
    #

    def has_price_crossed_upper_band(self) -> bool:
        current_price = self.get_current_close()
        current_upper_band = self.get_current_upper_band()

        if current_price < current_upper_band:
            self.tf_price_over_upper_band_counter = 0
            return False

        self.tf_price_over_upper_band_counter += 1
        self.logger().info(f"has_crossed_over_upper_band() | incremented self.tf_price_over_upper_band_counter to:{self.tf_price_over_upper_band_counter}")

        return self.tf_price_over_upper_band_counter > 39

    def has_price_crossed_lower_band(self) -> bool:
        current_price = self.get_current_close()
        current_lower_band = self.get_current_lower_band()

        if current_price < current_lower_band:
            self.logger().info(f"has_price_crossed_lower_band() | current_price:{current_price} | current_lower_band:{current_lower_band}")

        return current_price < current_lower_band
