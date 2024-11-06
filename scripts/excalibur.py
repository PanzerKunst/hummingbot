import asyncio
from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.pk.excalibur_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import was_an_order_recently_opened
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Trends via comparing 2 SMAs
# Generate config file: create --script-config excalibur
# Start the bot: start --script excalibur.py --conf conf_excalibur_GOAT.yml
#                start --script excalibur.py --conf conf_excalibur_GRASS.yml
#                start --script excalibur.py --conf conf_excalibur_MOODENG.yml
#                start --script excalibur.py --conf conf_excalibur_POPCAT.yml
# Quickstart script: -p=a -f excalibur.py -c conf_excalibur_POPCAT.yml

ORDER_REF_SMA_CROSS = "SmaCross"
ORDER_REF_MEAN_REVERSION = "MeanReversion"


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()
        self.reset_context_sma_cross()

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def get_triple_barrier(self, order_ref: str) -> TripleBarrier:
        if order_ref == ORDER_REF_SMA_CROSS:
            return TripleBarrier(
                open_order_type=OrderType.MARKET,
                stop_loss=self.config.sma_cross_stop_loss_pct / 100
            )

        return TripleBarrier(
            open_order_type=OrderType.MARKET,
            stop_loss=self.config.mean_reversion_stop_loss_pct / 100
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

        bb_df = candles_df.ta.bbands(length=self.config.bb_length, std=self.config.bb_std_dev)
        self.logger().error(f"bb_df.columns: {bb_df.columns}")

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        # TODO self.create_actions_proposal_sma_cross()
        # self.create_actions_proposal_mean_reversion()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        self.stop_actions_proposal_sma_cross()
        self.stop_actions_proposal_mean_reversion()

        return []  # Always return []

    def format_status(self) -> str:
        original_status = super().format_status()
        custom_status = ["\n"]

        if self.ready_to_trade:
            if not self.processed_data.empty:
                columns_to_display = [
                    "timestamp_iso",
                    "close",
                    "volume",
                    "RSI",
                    "SMA_short",
                    "SMA_long",
                    "BB_lower",
                    "BB_upper"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display], table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def create_actions_proposal_sma_cross(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_SMA_CROSS)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_sma_cross_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_SMA_CROSS)
            asyncio.get_running_loop().create_task(self.create_twap_market_orders(TradeType.SELL, entry_price, triple_barrier, ORDER_REF_SMA_CROSS))

        if self.can_create_sma_cross_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_SMA_CROSS)
            asyncio.get_running_loop().create_task(self.create_twap_market_orders(TradeType.BUY, entry_price, triple_barrier, ORDER_REF_SMA_CROSS))

    def can_create_sma_cross_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, ORDER_REF_SMA_CROSS, 0):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.did_short_sma_cross_under_long():
                self.logger().info("can_create_sma_cross_order() > Short SMA crossed under long")
                return not self.is_rsi_too_low_to_open_short() and not self.did_price_suddenly_rise_to_short_sma()

            return False

        if self.did_short_sma_cross_over_long():
            self.logger().info("can_create_sma_cross_order() > Short SMA crossed over long")
            return not self.is_rsi_too_high_to_open_long() and not self.did_price_suddenly_drop_to_short_sma()

        return False

    def stop_actions_proposal_sma_cross(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_SMA_CROSS)

        if len(filled_sell_orders) > 0:
            if self.did_short_sma_cross_over_long():
                self.logger().info("stop_actions_proposal_sma_cross(SELL) > Short SMA crossed over long")
                self.close_sma_cross_orders(filled_sell_orders, CloseType.COMPLETED)

            else:
                if self.should_close_sma_cross_orders_when_price_crosses_indicator:
                    if self.is_current_price_over_short_sma() or self.is_current_price_over_upper_bb():
                        self.logger().info("stop_actions_proposal_sma_cross(SELL) > current_price_is_over_short_sma or current_price_is_over_upper_bb")
                        self.close_sma_cross_orders(filled_sell_orders, CloseType.TAKE_PROFIT)

                elif self.should_short_orders_activate_trailing_stop(filled_sell_orders):
                    self.logger().info("stop_actions_proposal_sma_cross(SELL) > short_orders_should_activate_trailing_stop. Setting self.should_close_when_price_hits_sma to TRUE.")
                    self.should_close_sma_cross_orders_when_price_crosses_indicator = True

        if len(filled_buy_orders) > 0:
            if self.did_short_sma_cross_under_long():
                self.logger().info("stop_actions_proposal_sma_cross(BUY) > Short SMA crossed under long")
                self.close_sma_cross_orders(filled_buy_orders, CloseType.COMPLETED)

            else:
                if self.should_close_sma_cross_orders_when_price_crosses_indicator:
                    if self.is_current_price_under_short_sma() or self.is_current_price_under_lower_bb():
                        self.logger().info("stop_actions_proposal_sma_cross(BUY) > current_price_is_under_short_sma or current_price_is_under_lower_bb")
                        self.close_sma_cross_orders(filled_buy_orders, CloseType.TAKE_PROFIT)

                elif self.should_long_orders_activate_trailing_stop(filled_buy_orders):
                    self.logger().info("stop_actions_proposal_sma_cross(BUY) > long_orders_should_activate_trailing_stop. Setting self.should_close_when_price_hits_sma to TRUE.")
                    self.should_close_sma_cross_orders_when_price_crosses_indicator = True

    #
    # Custom functions specific to this controller
    #

    def create_actions_proposal_mean_reversion(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MEAN_REVERSION)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_mean_reversion_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MEAN_REVERSION)
            asyncio.get_running_loop().create_task(self.create_twap_market_orders(TradeType.SELL, entry_price, triple_barrier, ORDER_REF_MEAN_REVERSION))

        if self.can_create_mean_reversion_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MEAN_REVERSION)
            asyncio.get_running_loop().create_task(self.create_twap_market_orders(TradeType.BUY, entry_price, triple_barrier, ORDER_REF_MEAN_REVERSION))

    def can_create_mean_reversion_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        # No cooldown for MR orders, as having one could result in missed trades
        if not self.can_create_order(side, ORDER_REF_MEAN_REVERSION, 0):
            return False

        if was_an_order_recently_opened(active_tracked_orders, 5 * 60, self.get_market_data_provider_time()):
            self.logger().info("can_create_mean_reversion_order() > Recently opened an order - not doing it again")
            return False

        if side == TradeType.SELL:
            if self.did_price_drop_back_into_bb() and self.did_rsi_spike_and_recover():
                self.logger().info("can_create_mean_reversion_order() > Price just dropped back into BB")
                return True

            return False

        if self.did_price_rise_back_into_bb() and self.did_rsi_crash_and_recover():
            self.logger().info("can_create_mean_reversion_order() > Price just rose back into BB")
            return True

        return False

    def stop_actions_proposal_mean_reversion(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MEAN_REVERSION)

        if len(filled_sell_orders) > 0:
            if self.is_current_price_under_short_sma() or self.is_current_price_under_lower_bb():
                self.logger().info("stop_actions_proposal_mean_reversion(SELL) > current_price_is_under_short_sma or current_price_is_under_lower_bb")
                self.market_close_orders(filled_sell_orders, CloseType.TAKE_PROFIT)

        if len(filled_buy_orders) > 0:
            if self.is_current_price_over_short_sma() or self.is_current_price_over_upper_bb():
                self.logger().info("stop_actions_proposal_mean_reversion(BUY) > current_price_is_over_short_sma or current_price_is_over_upper_bb")
                self.market_close_orders(filled_buy_orders, CloseType.TAKE_PROFIT)

    def get_latest_close(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[-2])

    def get_current_rsi(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["RSI"]
        return Decimal(rsi_series.iloc[-1])

    def get_latest_sma(self, short_or_long: str) -> Decimal:
        return self._get_sma_at_index(short_or_long, -2)

    def get_previous_sma(self, short_or_long: str) -> Decimal:
        return self._get_sma_at_index(short_or_long, -3)

    def _get_sma_at_index(self, short_or_long: str, index: int) -> Decimal:
        sma_series: pd.Series = self.processed_data[f"SMA_{short_or_long}"]
        return Decimal(sma_series.iloc[index])

    def did_short_sma_cross_under_long(self) -> bool:
        return not self.is_latest_short_sma_over_long() and self.is_previous_short_sma_over_long()

    def did_short_sma_cross_over_long(self) -> bool:
        return self.is_latest_short_sma_over_long() and not self.is_previous_short_sma_over_long()

    def is_latest_short_sma_over_long(self) -> bool:
        latest_short_minus_long: Decimal = self.get_latest_sma("short") - self.get_latest_sma("long")
        return latest_short_minus_long > 0

    def is_previous_short_sma_over_long(self) -> bool:
        previous_short_minus_long: Decimal = self.get_previous_sma("short") - self.get_previous_sma("long")
        return previous_short_minus_long > 0

    def is_current_price_over_short_sma(self) -> bool:
        return self.get_latest_close() > self.get_latest_sma("short")

    def is_current_price_under_short_sma(self) -> bool:
        return not self.is_current_price_over_short_sma()

    def did_rsi_crash_and_recover(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        older_rsis = rsi_series.iloc[-6:-1]  # 5 items, last one excluded
        min_rsi = Decimal(older_rsis.min())

        if min_rsi > 27:
            return False

        current_rsi = self.get_current_rsi()

        if current_rsi - min_rsi < 3:
            return False

        self.logger().info(f"did_rsi_crash_and_recover() | current_rsi:{current_rsi} | min_rsi:{min_rsi}")

        return True

    def did_rsi_spike_and_recover(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        older_rsis = rsi_series.iloc[-6:-1]  # 5 items, last one excluded
        max_rsi = Decimal(older_rsis.max())

        if max_rsi < 73:
            return False

        current_rsi = self.get_current_rsi()

        if max_rsi - current_rsi < 3:
            return False

        self.logger().info(f"did_rsi_spike_and_recover() | current_rsi:{current_rsi} | max_rsi:{max_rsi}")

        return True

    def is_rsi_too_low_to_open_short(self) -> bool:
        current_rsi = self.get_current_rsi()

        self.logger().info(f"is_rsi_too_low_to_open_short() | current_rsi:{current_rsi}")

        return current_rsi < 37.5

    def is_rsi_too_high_to_open_long(self) -> bool:
        current_rsi = self.get_current_rsi()

        self.logger().info(f"is_rsi_too_high_to_open_long() | current_rsi:{current_rsi}")

        return current_rsi > 62.5

    def should_short_orders_activate_trailing_stop(self, filled_sell_orders: List[TrackedOrderDetails]) -> bool:
        pnl_pct: Decimal = self.compute_short_orders_pnl_pct(filled_sell_orders)
        return pnl_pct > self.config.sma_cross_trailing_stop_activation_pct

    def should_long_orders_activate_trailing_stop(self, filled_buy_orders: List[TrackedOrderDetails]) -> bool:
        pnl_pct: Decimal = self.compute_long_orders_pnl_pct(filled_buy_orders)
        return pnl_pct > self.config.sma_cross_trailing_stop_activation_pct

    def did_price_suddenly_rise_to_short_sma(self) -> bool:
        close_series: pd.Series = self.processed_data["close"]
        recent_prices = close_series.iloc[-17:-2]  # 15 items, last one excluded
        min_price: Decimal = Decimal(recent_prices.min())

        price_delta_pct: Decimal = (self.get_latest_close() - min_price) * 100

        self.logger().info(f"did_price_suddenly_rise_to_short_sma() | self.get_latest_close():{self.get_latest_close()} | min_price:{min_price} | price_delta_pct:{price_delta_pct}")

        # The percentage difference between min_price and current_price is over x%
        return price_delta_pct > self.config.min_price_delta_pct_for_sudden_reversal_to_short_sma

    def did_price_suddenly_drop_to_short_sma(self) -> bool:
        close_series: pd.Series = self.processed_data["close"]
        recent_prices = close_series.iloc[-17:-2]  # 15 items, last one excluded
        max_price: Decimal = Decimal(recent_prices.max())

        price_delta_pct: Decimal = (max_price - self.get_latest_close()) * 100

        self.logger().info(f"did_price_suddenly_drop_to_short_sma() | self.get_latest_close():{self.get_latest_close()} | max_price:{max_price} | price_delta_pct:{price_delta_pct}")

        return price_delta_pct > self.config.min_price_delta_pct_for_sudden_reversal_to_short_sma

    def did_price_drop_back_into_bb(self) -> bool:
        bb_upper_series: pd.Series = self.processed_data["BB_upper"]
        current_bb_upper = bb_upper_series.iloc[-1]
        bb_upper_latest_complete_candle = bb_upper_series.iloc[-2]

        close_series: pd.Series = self.processed_data["close"]
        current_close = close_series.iloc[-1]

        high_series: pd.Series = self.processed_data["high"]
        high_latest_complete_candle = high_series.iloc[-2]

        return high_latest_complete_candle > bb_upper_latest_complete_candle and current_close < current_bb_upper

    def did_price_rise_back_into_bb(self) -> bool:
        bb_lower_series: pd.Series = self.processed_data["BB_lower"]
        current_bb_lower = bb_lower_series.iloc[-1]
        bb_lower_latest_complete_candle = bb_lower_series.iloc[-2]

        close_series: pd.Series = self.processed_data["close"]
        current_close = close_series.iloc[-1]

        low_series: pd.Series = self.processed_data["low"]
        low_latest_complete_candle = low_series.iloc[-2]

        return low_latest_complete_candle < bb_lower_latest_complete_candle and current_close > current_bb_lower

    # Since based on real-time data, use only to close positions
    def is_current_price_under_lower_bb(self) -> bool:
        bb_lower_series: pd.Series = self.processed_data["BB_lower"]
        current_bb_lower = bb_lower_series.iloc[-1]

        close_series: pd.Series = self.processed_data["close"]
        current_close = close_series.iloc[-1]

        return current_close < current_bb_lower

    def is_current_price_over_upper_bb(self) -> bool:
        bb_upper_series: pd.Series = self.processed_data["BB_upper"]
        current_bb_upper = bb_upper_series.iloc[-1]

        close_series: pd.Series = self.processed_data["close"]
        current_close = close_series.iloc[-1]

        return current_close > current_bb_upper

    def compute_short_orders_pnl_pct(self, filled_sell_orders: List[TrackedOrderDetails]) -> Decimal:
        worst_filled_price = min(filled_sell_orders, key=lambda order: order.last_filled_price).last_filled_price
        return (worst_filled_price - self.get_latest_close()) / worst_filled_price * 100

    def compute_long_orders_pnl_pct(self, filled_buy_orders: List[TrackedOrderDetails]) -> Decimal:
        worst_filled_price = max(filled_buy_orders, key=lambda order: order.last_filled_price).last_filled_price
        return (self.get_latest_close() - worst_filled_price) / worst_filled_price * 100

    def close_sma_cross_orders(self, filled_orders: List[TrackedOrderDetails], close_type: CloseType):
        self.market_close_orders(filled_orders, close_type)
        self.reset_context_sma_cross()

    def reset_context_sma_cross(self):
        self.should_close_sma_cross_orders_when_price_crosses_indicator = False
