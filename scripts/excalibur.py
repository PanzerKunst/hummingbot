import asyncio
from decimal import Decimal
from typing import Dict, List

import pandas as pd
from pandas_ta import stoch

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.pk.excalibur_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Trend following via comparing 2 SMAs, and mean reversion based on RSI & SMA
# Generate config file: create --script-config excalibur
# Start the bot: start --script excalibur.py --conf conf_excalibur_GOAT.yml
#                start --script excalibur.py --conf conf_excalibur_MOODENG.yml
#                start --script excalibur.py --conf conf_excalibur_POPCAT.yml
# Quickstart script: -p=a -f excalibur.py -c conf_excalibur_POPCAT.yml

ORDER_REF_SMA_CROSS = "SmaCross"
ORDER_REF_MAJOR_MR = "MajorMR"
ORDER_REF_MINOR_MR = "MinorMR"


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
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
        candles_df["RSI_mr"] = candles_df.ta.rsi(length=self.config.rsi_mr_length)

        candles_df["SMA_short"] = candles_df.ta.sma(length=self.config.sma_short)
        candles_df["SMA_long"] = candles_df.ta.sma(length=self.config.sma_long)

        # Calling the lower-level function, because the one in core.py has a bug in the argument names
        stoch_df = stoch(
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            k=self.config.stoch_k_length,
            d=self.config.stoch_d_smoothing,
            smooth_k=self.config.stoch_k_smoothing
        )

        self.logger().info(f"stoch_df.columns:{stoch_df.columns}")

        candles_df["STOCH_k"] = stoch_df[f"STOCHk_{self.config.stoch_k_length}_{self.config.stoch_d_smoothing}_{self.config.stoch_k_smoothing}"]

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        # TODO self.create_actions_proposal_sma_cross()
        # TODO self.create_actions_proposal_major_mr()
        # TODO self.create_actions_proposal_minor_mr()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        self.stop_actions_proposal_sma_cross()
        self.stop_actions_proposal_major_mr()
        self.stop_actions_proposal_minor_mr()

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
                    "RSI_mr",
                    "SMA_short",
                    "SMA_long",
                    "STOCH_k"
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
                return self.is_price_close_enough_to_short_sma() and not self.is_rsi_too_low_to_open_short() and not self.did_price_suddenly_rise_to_short_sma()

            return False

        if self.did_short_sma_cross_over_long():
            self.logger().info("can_create_sma_cross_order() > Short SMA crossed over long")
            return self.is_price_close_enough_to_short_sma() and not self.is_rsi_too_high_to_open_long() and not self.did_price_suddenly_drop_to_short_sma()

        return False

    def stop_actions_proposal_sma_cross(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_SMA_CROSS)

        if len(filled_sell_orders) > 0:
            if self.did_short_sma_cross_over_long():
                self.logger().info("stop_actions_proposal_sma_cross(SELL) > Short SMA crossed over long")
                self.market_close_orders(filled_sell_orders, CloseType.COMPLETED)

        if len(filled_buy_orders) > 0:
            if self.did_short_sma_cross_under_long():
                self.logger().info("stop_actions_proposal_sma_cross(BUY) > Short SMA crossed under long")
                self.market_close_orders(filled_buy_orders, CloseType.COMPLETED)

    #
    # Custom functions specific to this controller
    #

    def create_actions_proposal_major_mr(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MAJOR_MR)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_major_mr_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MAJOR_MR)
            self.create_order(TradeType.SELL, entry_price, triple_barrier, ORDER_REF_MAJOR_MR)

        if self.can_create_major_mr_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MAJOR_MR)
            self.create_order(TradeType.BUY, entry_price, triple_barrier, ORDER_REF_MAJOR_MR)

    def can_create_major_mr_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, ORDER_REF_MAJOR_MR, 3):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.did_major_rsi_spike_and_recovery_happen():
                self.logger().info("can_create_major_mr_order() > RSI spike and recovery")
                return True

            return False

        if self.did_major_rsi_crash_and_recovery_happen():
            self.logger().info("can_create_major_mr_order() > RSI crash and recovery")
            return True

        return False

    def stop_actions_proposal_major_mr(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MAJOR_MR)

        if len(filled_sell_orders) > 0:
            if self.is_current_price_under_short_sma():
                self.logger().info("stop_actions_proposal_major_mr(SELL) > current_price_is_under_short_sma")
                self.market_close_orders(filled_sell_orders, CloseType.TAKE_PROFIT)

        if len(filled_buy_orders) > 0:
            if self.is_current_price_over_short_sma():
                self.logger().info("stop_actions_proposal_major_mr(BUY) > current_price_is_over_short_sma")
                self.market_close_orders(filled_buy_orders, CloseType.TAKE_PROFIT)

    def create_actions_proposal_minor_mr(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MINOR_MR)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_minor_mr_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MINOR_MR)
            self.create_order(TradeType.SELL, entry_price, triple_barrier, ORDER_REF_MINOR_MR)

        if self.can_create_minor_mr_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MINOR_MR)
            self.create_order(TradeType.BUY, entry_price, triple_barrier, ORDER_REF_MINOR_MR)

    def can_create_minor_mr_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, ORDER_REF_MINOR_MR, 3):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.does_srsi_indicate_to_open_minor_mr_short() and self.did_minor_rsi_spike_happen():
                self.logger().info("can_create_minor_mr_order() > Opening Minor Short MR")
                return True

            return False

        if self.does_srsi_indicate_to_open_minor_mr_long() and self.did_minor_rsi_crash_happen():
            self.logger().info("can_create_minor_mr_order() > Opening Minor Long MR")
            return True

        return False

    def stop_actions_proposal_minor_mr(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MINOR_MR)

        if len(filled_sell_orders) > 0:
            if self.should_close_minor_mr_short():
                self.logger().info("stop_actions_proposal_minor_mr(SELL) > should_close_mr_short")
                self.market_close_orders(filled_sell_orders, CloseType.TAKE_PROFIT)

        if len(filled_buy_orders) > 0:
            if self.should_close_minor_mr_long():
                self.logger().info("stop_actions_proposal_minor_mr(BUY) > should_close_minor_mr_long")
                self.market_close_orders(filled_buy_orders, CloseType.TAKE_PROFIT)

    def get_latest_close(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[-2])

    def get_current_rsi(self, default_or_mr: str) -> Decimal:
        column_name = "RSI" if default_or_mr == "default" else "RSI_mr"
        rsi_series: pd.Series = self.processed_data[column_name]
        return Decimal(rsi_series.iloc[-1])

    def get_latest_sma(self, short_or_long: str) -> Decimal:
        return self._get_sma_at_index(short_or_long, -2)

    def get_previous_sma(self, short_or_long: str) -> Decimal:
        return self._get_sma_at_index(short_or_long, -3)

    def _get_sma_at_index(self, short_or_long: str, index: int) -> Decimal:
        sma_series: pd.Series = self.processed_data[f"SMA_{short_or_long}"]
        return Decimal(sma_series.iloc[index])

    def get_current_srsi(self, k_or_d: str) -> Decimal:
        return self._get_srsi_at_index(k_or_d, -1)

    def get_latest_srsi(self, k_or_d: str) -> Decimal:
        return self._get_srsi_at_index(k_or_d, -2)

    def _get_srsi_at_index(self, k_or_d: str, index: int) -> Decimal:
        srsi_series: pd.Series = self.processed_data[f"SRSI_{k_or_d}"]
        return Decimal(srsi_series.iloc[index])

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

    def did_major_rsi_spike_and_recovery_happen(self) -> bool:
        current_rsi = self.get_current_rsi("mr")
        rsi_recovery_second_threshold: Decimal = self.config.rsi_major_spike_recovery_threshold - Decimal(0.5)

        if not (rsi_recovery_second_threshold < current_rsi < self.config.rsi_major_spike_recovery_threshold):
            return False

        rsi_series: pd.Series = self.processed_data["RSI_mr"].reset_index(drop=True)
        recent_rsis = rsi_series.iloc[-16:-1]  # 15 items, last one excluded

        peak_rsi = Decimal(recent_rsis.max())

        if peak_rsi < self.config.rsi_major_spike_peak_threshold:
            return False

        peak_rsi_index = recent_rsis.idxmax()
        bottom_rsi = Decimal(recent_rsis.iloc[0:peak_rsi_index].min())
        intro_delta: Decimal = peak_rsi - bottom_rsi

        self.logger().info(f"did_major_rsi_spike_and_recovery_happen() | bottom_rsi:{bottom_rsi} | peak_rsi:{peak_rsi} | current_rsi:{current_rsi} | intro_delta:{intro_delta}")

        return intro_delta > 15

    def did_major_rsi_crash_and_recovery_happen(self) -> bool:
        current_rsi = self.get_current_rsi("mr")
        rsi_recovery_second_threshold: Decimal = self.config.rsi_major_crash_recovery_threshold + Decimal(0.5)

        if not (self.config.rsi_major_crash_recovery_threshold < current_rsi < rsi_recovery_second_threshold):
            return False

        rsi_series: pd.Series = self.processed_data["RSI_mr"].reset_index(drop=True)
        recent_rsis = rsi_series.iloc[-16:-1]  # 15 items, last one excluded

        bottom_rsi = Decimal(recent_rsis.min())

        if bottom_rsi > self.config.rsi_major_crash_bottom_threshold:
            return False

        bottom_rsi_index = recent_rsis.idxmin()
        peak_rsi = Decimal(recent_rsis.iloc[0:bottom_rsi_index].max())
        intro_delta: Decimal = peak_rsi - bottom_rsi

        self.logger().info(f"did_major_rsi_crash_and_recovery_happen() | peak_rsi:{peak_rsi} | bottom_rsi:{bottom_rsi} | current_rsi:{current_rsi} | intro_delta:{intro_delta}")

        return intro_delta > 15

    def is_price_close_enough_to_short_sma(self):
        latest_close = self.get_latest_close()
        delta_pct: Decimal = (latest_close - self.get_latest_sma("short")) / latest_close * 100

        self.logger().info(f"is_price_close_enough_to_short_sma() | latest_close:{latest_close} | latest_short_sma:{self.get_latest_sma('short')} | delta_pct:{delta_pct}")

        return abs(delta_pct) < self.config.max_price_delta_pct_with_short_sma_to_open

    def is_rsi_too_low_to_open_short(self) -> bool:
        current_rsi = self.get_current_rsi("default")

        self.logger().info(f"is_rsi_too_low_to_open_short() | current_rsi:{current_rsi}")

        if current_rsi < 37.5:
            return True

        rsi_series: pd.Series = self.processed_data["RSI"]
        recent_rsis = rsi_series.iloc[-11:-1]  # 10 items, last one excluded

        min_rsi = Decimal(recent_rsis.min())

        self.logger().info(f"is_rsi_too_low_to_open_short() | min_rsi:{min_rsi}")

        return min_rsi < 30

    def is_rsi_too_high_to_open_long(self) -> bool:
        current_rsi = self.get_current_rsi("default")

        self.logger().info(f"is_rsi_too_high_to_open_long() | current_rsi:{current_rsi}")

        if current_rsi > 62.5:
            return True

        rsi_series: pd.Series = self.processed_data["RSI"]
        recent_rsis = rsi_series.iloc[-11:-1]  # 10 items, last one excluded

        max_rsi = Decimal(recent_rsis.max())

        self.logger().info(f"is_rsi_too_high_to_open_long() | max_rsi:{max_rsi}")

        return max_rsi > 70

    def did_price_suddenly_rise_to_short_sma(self) -> bool:
        latest_close = self.get_latest_close()

        close_series: pd.Series = self.processed_data["close"]
        recent_prices = close_series.iloc[-22:-2]  # 20 items, last one excluded
        min_price: Decimal = Decimal(recent_prices.min())

        price_delta_pct: Decimal = (latest_close - min_price) / latest_close * 100

        self.logger().info(f"did_price_suddenly_rise_to_short_sma() | latest_close:{latest_close} | min_price:{min_price} | price_delta_pct:{price_delta_pct}")

        # The percentage difference between min_price and current_price is over x%
        return price_delta_pct > self.config.min_price_delta_pct_for_sudden_reversal_to_short_sma

    def did_price_suddenly_drop_to_short_sma(self) -> bool:
        latest_close = self.get_latest_close()

        close_series: pd.Series = self.processed_data["close"]
        recent_prices = close_series.iloc[-22:-2]  # 20 items, last one excluded
        max_price: Decimal = Decimal(recent_prices.max())

        price_delta_pct: Decimal = (max_price - latest_close) / latest_close * 100

        self.logger().info(f"did_price_suddenly_drop_to_short_sma() | latest_close:{latest_close} | max_price:{max_price} | price_delta_pct:{price_delta_pct}")

        return price_delta_pct > self.config.min_price_delta_pct_for_sudden_reversal_to_short_sma

    def does_srsi_indicate_to_open_minor_mr_short(self) -> bool:
        latest_srsi_k = self.get_latest_srsi("k")

        if latest_srsi_k < 80:
            return False

        srsi_k_series: pd.Series = self.processed_data["SRSI_k"]
        recent_srsi_ks = srsi_k_series.iloc[-7:-2]  # 5 items, last one excluded
        peak_srsi_k: Decimal = Decimal(recent_srsi_ks.max())

        srsi_delta: Decimal = peak_srsi_k - latest_srsi_k

        self.logger().info(f"does_srsi_indicate_to_open_minor_mr_short() | peak_srsi_k:{peak_srsi_k} | latest_srsi_k:{latest_srsi_k} | srsi_delta:{srsi_delta}")

        return srsi_delta > 2

    def does_srsi_indicate_to_open_minor_mr_long(self) -> bool:
        latest_srsi_k = self.get_latest_srsi("k")

        if latest_srsi_k > 20:
            return False

        srsi_k_series: pd.Series = self.processed_data["SRSI_k"]
        recent_srsi_ks = srsi_k_series.iloc[-7:-2]  # 5 items, last one excluded
        bottom_srsi_k: Decimal = Decimal(recent_srsi_ks.min())

        srsi_delta: Decimal = latest_srsi_k - bottom_srsi_k

        self.logger().info(f"does_srsi_indicate_to_open_minor_mr_long() | bottom_srsi_k:{bottom_srsi_k} | latest_srsi_k:{latest_srsi_k} | srsi_delta:{srsi_delta}")

        return srsi_delta > 2

    def should_close_minor_mr_short(self) -> bool:
        current_srsi_k = self.get_current_srsi("k")

        if current_srsi_k > 20:
            return False

        srsi_k_series: pd.Series = self.processed_data["SRSI_k"]
        recent_srsi_ks = srsi_k_series.iloc[-7:-2]  # 5 items, last one excluded
        bottom_srsi_k: Decimal = Decimal(recent_srsi_ks.min())

        srsi_delta: Decimal = current_srsi_k - bottom_srsi_k

        self.logger().info(f"should_close_minor_mr_short() | bottom_srsi_k:{bottom_srsi_k} | current_srsi_k:{current_srsi_k} | srsi_delta:{srsi_delta}")

        return srsi_delta > 2

    def should_close_minor_mr_long(self) -> bool:
        current_srsi_k = self.get_current_srsi("k")

        if current_srsi_k < 80:
            return False

        srsi_k_series: pd.Series = self.processed_data["SRSI_k"]
        recent_srsi_ks = srsi_k_series.iloc[-7:-2]  # 5 items, last one excluded
        peak_srsi_k: Decimal = Decimal(recent_srsi_ks.max())

        srsi_delta: Decimal = peak_srsi_k - current_srsi_k

        self.logger().info(f"should_close_minor_mr_long() | peak_srsi_k:{peak_srsi_k} | current_srsi_k:{current_srsi_k} | srsi_delta:{srsi_delta}")

        return srsi_delta > 2

    def did_minor_rsi_spike_happen(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_mr"].reset_index(drop=True)
        recent_rsis = rsi_series.iloc[-16:-1]  # 15 items, last one excluded

        peak_rsi = Decimal(recent_rsis.max())

        if peak_rsi < 60:
            return False

        peak_rsi_index = recent_rsis.idxmax()
        bottom_rsi = Decimal(recent_rsis.iloc[0:peak_rsi_index].min())
        intro_delta: Decimal = peak_rsi - bottom_rsi

        current_rsi = self.get_current_rsi("mr")
        outro_delta: Decimal = peak_rsi - current_rsi

        self.logger().info(f"did_minor_rsi_spike_happen() | bottom_rsi:{bottom_rsi} | peak_rsi:{peak_rsi} | current_rsi:{current_rsi} | intro_delta:{intro_delta} | outro_delta:{outro_delta}")

        return intro_delta > 10 and outro_delta > Decimal(1.5)

    def did_minor_rsi_crash_happen(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_mr"].reset_index(drop=True)
        recent_rsis = rsi_series.iloc[-16:-1]  # 15 items, last one excluded

        bottom_rsi = Decimal(recent_rsis.min())

        if bottom_rsi > 40:
            return False

        bottom_rsi_index = recent_rsis.idxmin()
        peak_rsi = Decimal(recent_rsis.iloc[0:bottom_rsi_index].max())
        intro_delta: Decimal = peak_rsi - bottom_rsi

        current_rsi = self.get_current_rsi("mr")
        outro_delta: Decimal = current_rsi - bottom_rsi

        self.logger().info(f"did_minor_rsi_crash_happen() | peak_rsi:{peak_rsi} | bottom_rsi:{bottom_rsi} | current_rsi:{current_rsi} | intro_delta:{intro_delta} | outro_delta:{outro_delta}")

        return intro_delta > 10 and outro_delta > Decimal(1.5)
