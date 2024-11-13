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
ORDER_REF_MR = "MeanReversion"


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
            open_order_type=OrderType.MARKET
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

        candles_df["STOCH_k"] = stoch_df[f"STOCHk_{self.config.stoch_k_length}_{self.config.stoch_d_smoothing}_{self.config.stoch_k_smoothing}"]

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        self.create_actions_proposal_sma_cross()
        self.create_actions_proposal_mr()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        self.stop_actions_proposal_sma_cross()
        self.stop_actions_proposal_mr()

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

    def create_actions_proposal_mr(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MR)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_mr_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MR)
            self.create_order(TradeType.SELL, entry_price, triple_barrier, ORDER_REF_MR)

        if self.can_create_mr_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MR)
            self.create_order(TradeType.BUY, entry_price, triple_barrier, ORDER_REF_MR)

    def can_create_mr_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, ORDER_REF_MR, 0):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.did_rsi_spike() and self.is_stoch_good_to_open_mr_short():
                self.logger().info("can_create_mr_order() > Opening Short MR")
                return True

            return False

        if self.did_rsi_crash() and self.is_stoch_good_to_open_mr_long():
            self.logger().info("can_create_mr_order() > Opening Long MR")
            return True

        return False

    def stop_actions_proposal_mr(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MR)

        if len(filled_sell_orders) > 0:
            if self.should_close_mr_short():
                self.logger().info("stop_actions_proposal_mr() > should_close_mr_short")
                self.market_close_orders(filled_sell_orders, CloseType.TAKE_PROFIT)

        if len(filled_buy_orders) > 0:
            if self.should_close_mr_long():
                self.logger().info("stop_actions_proposal_mr() > should_close_mr_long")
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

    def get_current_stoch(self) -> Decimal:
        stoch_series: pd.Series = self.processed_data["STOCH_k"]
        return Decimal(stoch_series.iloc[-1])

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

    def did_rsi_spike(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_mr"].reset_index(drop=True)
        recent_rsis = rsi_series.iloc[-16:-1]  # 15 items, last one excluded

        peak_rsi = Decimal(recent_rsis.max())

        if peak_rsi < self.config.rsi_spike_peak_threshold:
            return False

        current_rsi = self.get_current_rsi("mr")
        min_acceptable_rsi: Decimal = peak_rsi - 2

        if current_rsi < min_acceptable_rsi:
            return False

        peak_rsi_index = recent_rsis.idxmax()
        bottom_rsi = Decimal(recent_rsis.iloc[0:peak_rsi_index].min())
        start_delta: Decimal = peak_rsi - bottom_rsi

        if start_delta < 12:
            return False

        self.logger().info(f"did_rsi_spike() | bottom_rsi:{bottom_rsi} | peak_rsi:{peak_rsi} | current_rsi:{current_rsi} | start_delta:{start_delta}")

        return current_rsi < min_acceptable_rsi + Decimal(0.5)

    def did_rsi_crash(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_mr"].reset_index(drop=True)
        recent_rsis = rsi_series.iloc[-16:-1]  # 15 items, last one excluded

        bottom_rsi = Decimal(recent_rsis.min())

        if bottom_rsi > self.config.rsi_crash_bottom_threshold:
            return False

        current_rsi = self.get_current_rsi("mr")
        max_acceptable_rsi: Decimal = bottom_rsi + 2

        if current_rsi > max_acceptable_rsi:
            return False

        bottom_rsi_index = recent_rsis.idxmin()
        peak_rsi = Decimal(recent_rsis.iloc[0:bottom_rsi_index].max())
        start_delta: Decimal = peak_rsi - bottom_rsi

        if start_delta < 12:
            return False

        self.logger().info(f"did_rsi_crash() | peak_rsi:{peak_rsi} | bottom_rsi:{bottom_rsi} | current_rsi:{current_rsi} | start_delta:{start_delta}")

        return current_rsi > max_acceptable_rsi - Decimal(0.5)

    def is_stoch_good_to_open_mr_short(self) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_k"]
        recent_stochs = stoch_series.iloc[-6:-1]  # 5 items, last one excluded
        peak_stoch: Decimal = Decimal(recent_stochs.max())

        if peak_stoch < 90:
            return False

        current_stoch = self.get_current_stoch()
        max_acceptable_stoch: Decimal = peak_stoch - 2

        self.logger().info(f"is_stoch_good_to_open_mr_short() | peak_stoch:{peak_stoch} | current_stoch:{current_stoch}")

        return current_stoch < max_acceptable_stoch

    def is_stoch_good_to_open_mr_long(self) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_k"]
        recent_stochs = stoch_series.iloc[-6:-1]  # 5 items, last one excluded
        bottom_stoch: Decimal = Decimal(recent_stochs.min())

        if bottom_stoch > 10:
            return False

        current_stoch = self.get_current_stoch()
        min_acceptable_stoch: Decimal = bottom_stoch + 2

        self.logger().info(f"is_stoch_good_to_open_mr_long() | bottom_stoch:{bottom_stoch} | current_stoch:{current_stoch}")

        return current_stoch > min_acceptable_stoch

    def should_close_mr_short(self) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_k"]
        recent_stochs = stoch_series.iloc[-6:-1]  # 5 items, last one excluded
        bottom_stoch: Decimal = Decimal(recent_stochs.min())

        if bottom_stoch > 18:
            return False

        current_stoch = self.get_current_stoch()
        min_acceptable_stoch: Decimal = bottom_stoch + 2

        self.logger().info(f"should_close_mr_short() | bottom_stoch:{bottom_stoch} | current_stoch:{current_stoch}")

        return current_stoch > min_acceptable_stoch

    def should_close_mr_long(self) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_k"]
        recent_stochs = stoch_series.iloc[-6:-1]  # 5 items, last one excluded
        peak_stoch: Decimal = Decimal(recent_stochs.max())

        if peak_stoch < 82:
            return False

        current_stoch = self.get_current_stoch()
        max_acceptable_stoch: Decimal = peak_stoch - 2

        self.logger().info(f"should_close_mr_long() | peak_stoch:{peak_stoch} | current_stoch:{current_stoch}")

        return current_stoch < max_acceptable_stoch
