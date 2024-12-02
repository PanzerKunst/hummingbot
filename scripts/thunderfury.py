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
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import was_an_order_recently_opened
from scripts.pk.tracked_order_details import TrackedOrderDetails
from scripts.thunderfury_config import ExcaliburConfig

# Mean reversion based on price, RSI & Stochastic
# Generate config file: create --script-config thunderfury
# Start the bot: start --script thunderfury.py --conf conf_thunderfury_GOAT.yml
#                start --script thunderfury.py --conf conf_thunderfury_CHILLGUY.yml
#                start --script thunderfury.py --conf conf_thunderfury_FLOKI.yml
#                start --script thunderfury.py --conf conf_thunderfury_MOODENG.yml
#                start --script thunderfury.py --conf conf_thunderfury_NEIRO.yml
#                start --script thunderfury.py --conf conf_thunderfury_PNUT.yml
#                start --script thunderfury.py --conf conf_thunderfury_POPCAT.yml
# Quickstart script: -p=a -f thunderfury.py -c conf_thunderfury_GOAT.yml

ORDER_REF_REV = "Rev"


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()

        self.last_price_spike_or_crash_pct: Decimal = Decimal(0.0)

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
        stop_loss_pct: Decimal = self.last_price_spike_or_crash_pct / 6

        return TripleBarrier(
            open_order_type=OrderType.MARKET,
            stop_loss=stop_loss_pct / 100
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

        candles_df["RSI_40"] = candles_df.ta.rsi(length=40)

        # Calling the lower-level function, because the one in core.py has a bug in the argument names
        stoch_40_df = stoch(
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            k=40,
            d=6,
            smooth_k=8
        )

        candles_df["STOCH_40_k"] = stoch_40_df["STOCHk_40_6_8"]

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        self.create_actions_proposal_rev()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        self.stop_actions_proposal_rev()

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
                    "RSI_40",
                    "STOCH_40_k"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display], table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Reversion start/stop action proposals
    #

    def create_actions_proposal_rev(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_REV)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_rev_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier()
            self.create_order(TradeType.SELL, entry_price, triple_barrier, self.config.amount_quote, ORDER_REF_REV)

        if self.can_create_rev_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier()
            self.create_order(TradeType.BUY, entry_price, triple_barrier, self.config.amount_quote, ORDER_REF_REV)

    def can_create_rev_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_REV, 8):
            return False

        if len(active_tracked_orders) > 0:
            return False

        candle_count_for_rev: int = 6

        if side == TradeType.SELL:
            if self.is_price_spiking(candle_count_for_rev) and self.has_rsi_peaked(candle_count_for_rev):
                self.logger().info("can_create_rev_order() > Opening Sell reversion")
                return True

            return False

        if self.is_price_crashing(candle_count_for_rev) and self.has_rsi_bottomed(candle_count_for_rev):
            self.logger().info("can_create_rev_order() > Opening Buy reversion")
            return True

        return False

    def stop_actions_proposal_rev(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_REV)

        if len(filled_sell_orders) > 0:
            if self.should_close_rev_sell_due_to_stoch_reversal(filled_sell_orders):
                self.logger().info("stop_actions_proposal_rev() > Closing Sell reversion")
                self.market_close_orders(filled_sell_orders, CloseType.COMPLETED)

        if len(filled_buy_orders) > 0:
            if self.should_close_rev_buy_due_to_stoch_reversal(filled_buy_orders):
                self.logger().info("stop_actions_proposal_rev() > Closing Buy reversion")
                self.market_close_orders(filled_buy_orders, CloseType.COMPLETED)

    #
    # Getters on `self.processed_data[]`
    #

    def get_current_close(self) -> Decimal:
        return self._get_close_at_index(-1)

    def get_latest_close(self) -> Decimal:
        return self._get_close_at_index(-2)

    def _get_close_at_index(self, index: int) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[index])

    def get_current_rsi(self, length: int) -> Decimal:
        rsi_series: pd.Series = self.processed_data[f"RSI_{length}"]
        return Decimal(rsi_series.iloc[-1])

    def get_current_stoch(self, length: int) -> Decimal:
        return self._get_stoch_at_index(length, -1)

    def get_latest_stoch(self, length: int) -> Decimal:
        return self._get_stoch_at_index(length, -2)

    def _get_stoch_at_index(self, length: int, index: int) -> Decimal:
        stoch_series: pd.Series = self.processed_data[f"STOCH_{length}_k"]
        return Decimal(stoch_series.iloc[index])

    #
    # Reversion functions
    #

    def is_price_spiking(self, candle_count: int) -> bool:
        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-candle_count:].reset_index(drop=True)

        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-candle_count:]

        peak_price = Decimal(recent_highs.max())
        peak_price_index = recent_highs.idxmax()

        if peak_price_index == 0:
            return False

        bottom_price = Decimal(recent_lows.iloc[0:peak_price_index].min())
        start_delta_pct: Decimal = (peak_price - bottom_price) / bottom_price * 100
        is_spiking = start_delta_pct > self.config.price_start_delta_pct_to_open

        if is_spiking:
            self.logger().info(f"is_price_spiking() | peak_price_index:{peak_price_index} | peak_price:{peak_price} | bottom_price:{bottom_price} | start_delta_pct:{start_delta_pct}")
            self.last_price_spike_or_crash_pct = start_delta_pct

        return is_spiking

    def is_price_crashing(self, candle_count: int) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-candle_count:].reset_index(drop=True)

        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-candle_count:]

        bottom_price = Decimal(recent_lows.min())
        bottom_price_index = recent_lows.idxmin()

        if bottom_price_index == 0:
            return False

        peak_price = Decimal(recent_highs.iloc[0:bottom_price_index].max())
        start_delta_pct: Decimal = (peak_price - bottom_price) / bottom_price * 100
        is_crashing = start_delta_pct > self.config.price_start_delta_pct_to_open

        if is_crashing:
            self.logger().info(f"is_price_crashing() | bottom_price_index:{bottom_price_index} | bottom_price:{bottom_price} | peak_price:{peak_price} | start_delta_pct:{start_delta_pct}")
            self.last_price_spike_or_crash_pct = start_delta_pct

        return is_crashing

    def has_rsi_peaked(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_40"]
        recent_rsis = rsi_series.iloc[-candle_count:]

        peak_rsi = Decimal(recent_rsis.max())

        # Avoids opening an opposite Sell Rev, when the price goes back up after a crash
        if peak_rsi < 63:
            return False

        # If peak_rsi > 72, we wait until it's back down to 70
        rsi_threshold: Decimal = 70 if peak_rsi > 72 else peak_rsi - 2
        current_rsi = self.get_current_rsi(40)

        if current_rsi > rsi_threshold:
            return False

        too_late_threshold: Decimal = rsi_threshold - 2
        has_peaked = current_rsi > too_late_threshold

        if has_peaked:
            self.logger().info(f"has_rsi_peaked() | peak_rsi:{peak_rsi} | current_rsi:{current_rsi} | rsi_threshold:{rsi_threshold}")

        return has_peaked

    def has_rsi_bottomed(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_40"]
        recent_rsis = rsi_series.iloc[-candle_count:]

        bottom_rsi = Decimal(recent_rsis.min())

        if bottom_rsi > 37:
            return False

        rsi_threshold: Decimal = 30 if bottom_rsi < 28 else bottom_rsi + 2
        current_rsi = self.get_current_rsi(40)

        if current_rsi < rsi_threshold:
            return False

        too_late_threshold: Decimal = rsi_threshold + 2
        has_bottomed = current_rsi < too_late_threshold

        if has_bottomed:
            self.logger().info(f"has_rsi_bottomed() | bottom_rsi:{bottom_rsi} | current_rsi:{current_rsi} | rsi_threshold:{rsi_threshold}")

        return has_bottomed

    def should_close_rev_sell_due_to_stoch_reversal(self, filled_sell_orders: List[TrackedOrderDetails]) -> bool:
        # Don't close if we just opened
        if was_an_order_recently_opened(filled_sell_orders, 8 * 60, self.get_market_data_provider_time()):
            return False

        stoch_series: pd.Series = self.processed_data["STOCH_40_k"]
        recent_stochs = stoch_series.iloc[-8:]
        bottom_stoch: Decimal = Decimal(recent_stochs.min())

        if bottom_stoch > 20:
            return False

        current_stoch = self.get_current_stoch(40)
        stoch_threshold: Decimal = bottom_stoch + 1

        self.logger().info(f"should_close_rev_sell_due_to_stoch_reversal() | bottom_stoch:{bottom_stoch} | current_stoch:{current_stoch}")

        return current_stoch > stoch_threshold

    def should_close_rev_buy_due_to_stoch_reversal(self, filled_buy_orders: List[TrackedOrderDetails]) -> bool:
        # Don't close if we just opened
        if was_an_order_recently_opened(filled_buy_orders, 8 * 60, self.get_market_data_provider_time()):
            return False

        stoch_series: pd.Series = self.processed_data["STOCH_40_k"]
        recent_stochs = stoch_series.iloc[-8:]
        peak_stoch: Decimal = Decimal(recent_stochs.max())

        if peak_stoch < 80:
            return False

        current_stoch = self.get_current_stoch(40)
        stoch_threshold: Decimal = peak_stoch - 1

        self.logger().info(f"should_close_rev_buy_due_to_stoch_reversal() | peak_stoch:{peak_stoch} | current_stoch:{current_stoch}")

        return current_stoch < stoch_threshold
