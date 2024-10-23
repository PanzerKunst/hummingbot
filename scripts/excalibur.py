from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig, TrailingStop
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.pk.excalibur_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_utils import get_take_profit_price
from scripts.pk.tracked_order_details import TrackedOrderDetails


# Trends via comparing 2 SMAs
# Generate config file: create --script-config excalibur
# Start the bot: start --script excalibur.py --conf conf_excalibur_BOME.yml
#                start --script excalibur.py --conf conf_excalibur_POPCAT.yml
#                start --script excalibur.py --conf conf_excalibur_SOL.yml
# Quickstart script: -p=a -f excalibur.py -c conf_excalibur_POPCAT.yml


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()

        self.is_ready_to_sell = False
        self.is_ready_to_buy = False

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
            stop_loss=Decimal(self.config.stop_loss_pct / 100),
            trailing_stop=trailing_stop,
            open_order_type=OrderType.LIMIT,
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

        candles_df["SMA_short"] = candles_df.ta.sma(length=self.config.sma_short)
        candles_df["SMA_long"] = candles_df.ta.sma(length=self.config.sma_long)

        candles_df["RSI_for_open"] = candles_df.ta.rsi(length=self.config.rsi_length_for_open_order)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_sma_cross_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier_config = self.get_triple_barrier_config(TradeType.SELL, entry_price)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_create_sma_cross_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier_config = self.get_triple_barrier_config(TradeType.BUY, entry_price)
            self.create_order(TradeType.BUY, entry_price, triple_barrier_config)

        if self.can_recreate_order_after_tp(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier_config = self.get_triple_barrier_config(TradeType.SELL, entry_price)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_recreate_order_after_tp(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier_config = self.get_triple_barrier_config(TradeType.BUY, entry_price)
            self.create_order(TradeType.BUY, entry_price, triple_barrier_config)

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side()

        if len(filled_sell_orders) == 1:
            filled_order = filled_sell_orders[0]

            if self.did_short_sma_cross_over_long():
                self.logger().info("stop_actions_proposal() > Short SMA crossed over long")
                self.close_filled_order(filled_order, OrderType.MARKET, CloseType.COMPLETED)

            else:
                did_rsi_crash_and_recover = self.did_rsi_crash_and_recover()

                if did_rsi_crash_and_recover:
                    self.logger().info(f"stop_actions_proposal(SELL) > rsi_did_crash_and_recover")

                if did_rsi_crash_and_recover and self.has_position_been_open_long_enough(filled_order):
                    self.close_filled_order(filled_order, OrderType.MARKET, CloseType.TAKE_PROFIT)

        if len(filled_buy_orders) == 1:
            filled_order = filled_buy_orders[0]

            if self.did_short_sma_cross_under_long():
                self.logger().info("stop_actions_proposal() > Short SMA crossed under long")
                self.close_filled_order(filled_order, OrderType.MARKET, CloseType.COMPLETED)

            else:
                did_rsi_spike_and_recover = self.did_rsi_spike_and_recover()

                if did_rsi_spike_and_recover:
                    self.logger().info(f"stop_actions_proposal(BUY) > rsi_did_spike_and_recover")

                if did_rsi_spike_and_recover and self.has_position_been_open_long_enough(filled_order):
                    self.close_filled_order(filled_order, OrderType.MARKET, CloseType.TAKE_PROFIT)

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
                    "SMA_short",
                    "SMA_long",
                    "RSI_for_open"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(20), table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_sma_cross_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.is_ready_to_sell and self.is_short_rsi_good_for_sell():
                self.is_ready_to_sell = False
                return True

            if self.did_short_sma_cross_under_long():
                self.logger().info("can_create_sma_cross_order() > Short SMA crossed under long")
                self.is_ready_to_sell = True

            return False

        if self.is_ready_to_buy and self.is_short_rsi_good_for_buy():
            self.is_ready_to_buy = False
            return True

        if self.did_short_sma_cross_over_long():
            self.logger().info("can_create_sma_cross_order() > Short SMA crossed over long")
            self.is_ready_to_buy = True

        return False

    def can_recreate_order_after_tp(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]):
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        last_terminated_filled_order = self.find_last_terminated_filled_order(side)

        # If the last completed order on this side was a TP
        if not last_terminated_filled_order or last_terminated_filled_order.close_type != CloseType.TAKE_PROFIT:
            return False

        self.logger().info(f"can_recreate_order_after_tp({side}) > last_terminated_filled_order is TP")

        return True

    #
    # Custom functions specific to this controller
    #

    def get_latest_sma(self, short_or_long: str) -> float:
        return self._get_sma_at_index(short_or_long, -1)

    def get_previous_sma(self, short_or_long: str) -> float:
        return self._get_sma_at_index(short_or_long, -2)

    def _get_sma_at_index(self, short_or_long: str, index: int) -> float:
        return self.processed_data[f"SMA_{short_or_long}"].iloc[index]

    def did_short_sma_cross_under_long(self) -> bool:
        return not self.is_latest_short_sma_over_long() and self.is_previous_short_sma_over_long()

    def did_short_sma_cross_over_long(self) -> bool:
        return self.is_latest_short_sma_over_long() and not self.is_previous_short_sma_over_long()

    def is_latest_short_sma_over_long(self) -> bool:
        latest_short_minus_long: float = self.get_latest_sma("short") - self.get_latest_sma("long")
        return latest_short_minus_long > 0

    def is_previous_short_sma_over_long(self) -> bool:
        previous_short_minus_long: float = self.get_previous_sma("short") - self.get_previous_sma("long")
        return previous_short_minus_long > 0

    def did_rsi_crash_and_recover(self):
        rsi_series: pd.Series = self.processed_data["RSI"]
        current_rsi = Decimal(rsi_series.iloc[-1])
        older_rsis = rsi_series.iloc[-13:-1]  # 12 items, last one excluded

        return older_rsis.min() < self.config.take_profit_sell_rsi_threshold and current_rsi > 30

    def has_position_been_open_long_enough(self, tracked_order: TrackedOrderDetails) -> bool:
        return tracked_order.last_filled_at + self.config.filled_position_min_duration_min * 60 < self.get_market_data_provider_time()

    def did_rsi_spike_and_recover(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        current_rsi = Decimal(rsi_series.iloc[-1])
        older_rsis = rsi_series.iloc[-13:-1]

        return older_rsis.max() > self.config.take_profit_buy_rsi_threshold and current_rsi < 70

    def is_short_rsi_good_for_sell(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_for_open"]
        current_rsi = Decimal(rsi_series.iloc[-1])

        # TODO: remove
        self.logger().info(f"is_short_rsi_good_for_sell: {current_rsi}")

        return current_rsi > self.config.min_rsi_to_open_sell_order

    def is_short_rsi_good_for_buy(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_for_open"]
        current_rsi = Decimal(rsi_series.iloc[-1])

        # TODO: remove
        self.logger().info(f"is_short_rsi_good_for_buy: {current_rsi}")

        return current_rsi < self.config.max_rsi_to_open_buy_order
