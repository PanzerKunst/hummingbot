from decimal import Decimal
from typing import Dict, List, Tuple

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
from scripts.pk.pk_utils import compute_rsi_pullback_difference
from scripts.pk.tracked_order_details import TrackedOrderDetails
from scripts.thunderfury_config import ExcaliburConfig

# Generate config file: create --script-config thunderfury
# Start the bot: start --script thunderfury.py --conf conf_thunderfury_GOAT.yml
#                start --script thunderfury.py --conf conf_thunderfury_BOME.yml
#                start --script thunderfury.py --conf conf_thunderfury_CHILLGUY.yml
#                start --script thunderfury.py --conf conf_thunderfury_FLOKI.yml
#                start --script thunderfury.py --conf conf_thunderfury_MOODENG.yml
#                start --script thunderfury.py --conf conf_thunderfury_NEIRO.yml
#                start --script thunderfury.py --conf conf_thunderfury_PNUT.yml
#                start --script thunderfury.py --conf conf_thunderfury_POPCAT.yml
#                start --script thunderfury.py --conf conf_thunderfury_WIF.yml
# Quickstart script: -p=a -f thunderfury.py -c conf_thunderfury_GOAT.yml

ORDER_REF_REV = "Rev"


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()
        self.reset_context()

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
        saved_price_spike_or_crash_pct, _ = self.saved_price_spike_or_crash_pct
        stop_loss_pct: Decimal = saved_price_spike_or_crash_pct / 2

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

        candles_df["SMA_8"] = candles_df.ta.sma(length=8)

        # Calling the lower-level function, because the one in core.py has a bug in the argument names
        stoch_10_df = stoch(
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            k=10,
            d=1,
            smooth_k=1
        )

        candles_df["STOCH_10_k"] = stoch_10_df["STOCHk_10_1_1"]

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        self.check_context(6)  # `candle_count_for_rev` + 1
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
                    "SMA_8",
                    "STOCH_10_k"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(30), table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Reversion start/stop action proposals
    #

    def create_actions_proposal_rev(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_REV)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_rev_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_mid_price() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier()
            self.create_order(TradeType.SELL, entry_price, triple_barrier, self.config.amount_quote, ORDER_REF_REV)

        if self.can_create_rev_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_mid_price() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier()
            self.create_order(TradeType.BUY, entry_price, triple_barrier, self.config.amount_quote, ORDER_REF_REV)

    def can_create_rev_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_REV, 5):
            return False

        if len(active_tracked_orders) > 0:
            return False

        candle_count_for_rev: int = 5

        if side == TradeType.SELL:
            if (
                self.is_price_spiking(candle_count_for_rev) and
                not self.is_price_spike_a_reversal(candle_count_for_rev) and
                self.has_rsi_peaked(candle_count_for_rev)
            ):
                # self.logger().info("can_create_rev_order() > Opening Sell reversion")
                return False  # Disabled for now

            return False

        if (
            self.is_price_crashing(candle_count_for_rev) and
            not self.is_price_crash_a_reversal(candle_count_for_rev) and
            self.has_rsi_bottomed(candle_count_for_rev)
        ):
            self.logger().info("can_create_rev_order() > Opening Buy reversion")
            return True

        return False

    def stop_actions_proposal_rev(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_REV)

        if len(filled_sell_orders) > 0:
            if self.is_price_under_ma() and self.has_stoch_reversed_for_sell():
                self.logger().info("stop_actions_proposal_rev() > Closing Sell reversion")
                self.market_close_orders(filled_sell_orders, CloseType.COMPLETED)

        if len(filled_buy_orders) > 0:
            if self.is_price_over_ma() and self.has_stoch_reversed_for_buy():
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

    def get_current_ma(self) -> Decimal:
        sma_series: pd.Series = self.processed_data["SMA_8"]
        return Decimal(sma_series.iloc[-1])

    def get_current_stoch(self, length: int) -> Decimal:
        return self._get_stoch_at_index(length, -1)

    def get_latest_stoch(self, length: int) -> Decimal:
        return self._get_stoch_at_index(length, -2)

    def _get_stoch_at_index(self, length: int, index: int) -> Decimal:
        stoch_series: pd.Series = self.processed_data[f"STOCH_{length}_k"]
        return Decimal(stoch_series.iloc[index])

    #
    # Context functions
    #

    def reset_context(self):
        self.save_bottom_price(Decimal("Infinity"), self.get_market_data_provider_time())
        self.save_peak_price(Decimal(0.0), self.get_market_data_provider_time())

        self.save_price_spike_or_crash_pct(Decimal(0.0), self.get_market_data_provider_time())

        self.save_bottom_rsi(Decimal(50.0), self.get_market_data_provider_time())
        self.save_peak_rsi(Decimal(50.0), self.get_market_data_provider_time())

        self.save_bottom_stoch(Decimal(50.0), self.get_market_data_provider_time())
        self.save_peak_stoch(Decimal(50.0), self.get_market_data_provider_time())

        self.rsi_reversal_counter: int = 0
        self.stoch_reversal_counter: int = 0

    def save_bottom_price(self, bottom_price: Decimal, timestamp: float):
        self.saved_bottom_price: Tuple[Decimal, float] = bottom_price, timestamp

    def save_peak_price(self, peak_price: Decimal, timestamp: float):
        self.saved_peak_price: Tuple[Decimal, float] = peak_price, timestamp

    def save_price_spike_or_crash_pct(self, price_spike_or_crash_pct: Decimal, timestamp: float):
        self.saved_price_spike_or_crash_pct: Tuple[Decimal, float] = price_spike_or_crash_pct, timestamp

    def save_bottom_rsi(self, bottom_rsi: Decimal, timestamp: float):
        self.saved_bottom_rsi: Tuple[Decimal, float] = bottom_rsi, timestamp

    def save_peak_rsi(self, peak_rsi: Decimal, timestamp: float):
        self.saved_peak_rsi: Tuple[Decimal, float] = peak_rsi, timestamp

    def save_bottom_stoch(self, bottom_stoch: Decimal, timestamp: float):
        self.saved_bottom_stoch: Tuple[Decimal, float] = bottom_stoch, timestamp

    def save_peak_stoch(self, peak_stoch: Decimal, timestamp: float):
        self.saved_peak_stoch: Tuple[Decimal, float] = peak_stoch, timestamp

    def check_context(self, lifetime_minutes: int):
        saved_bottom_price, saved_bottom_price_timestamp = self.saved_bottom_price
        saved_peak_price, saved_peak_price_timestamp = self.saved_peak_price

        saved_price_spike_or_crash_pct, saved_price_spike_or_crash_pct_timestamp = self.saved_price_spike_or_crash_pct

        saved_bottom_rsi, saved_bottom_rsi_timestamp = self.saved_bottom_rsi
        saved_peak_rsi, saved_peak_rsi_timestamp = self.saved_peak_rsi

        saved_bottom_stoch, saved_bottom_stoch_timestamp = self.saved_bottom_stoch
        saved_peak_stoch, saved_peak_stoch_timestamp = self.saved_peak_stoch

        all_timestamps: List[float] = [
            saved_bottom_price_timestamp,
            saved_peak_price_timestamp,
            saved_price_spike_or_crash_pct_timestamp,
            saved_bottom_rsi_timestamp,
            saved_peak_rsi_timestamp,
            saved_bottom_stoch_timestamp,
            saved_peak_stoch_timestamp
        ]

        last_acceptable_timestamp = self.get_market_data_provider_time() - lifetime_minutes * 60

        is_any_outdated: bool = any(timestamp < last_acceptable_timestamp for timestamp in all_timestamps)

        if is_any_outdated and not self.is_context_default():
            self.logger().info(f"check_context() | One of the context vars is outdated | {last_acceptable_timestamp} | all_timestamps:{all_timestamps}")
            self.reset_context()

    def is_context_default(self) -> bool:
        saved_bottom_price, _ = self.saved_bottom_price
        saved_peak_price, _ = self.saved_peak_price

        saved_price_spike_or_crash_pct, _ = self.saved_price_spike_or_crash_pct

        saved_bottom_rsi, _ = self.saved_bottom_rsi
        saved_peak_rsi, _ = self.saved_peak_rsi

        saved_bottom_stoch, _ = self.saved_bottom_stoch
        saved_peak_stoch, _ = self.saved_peak_stoch

        return (
            saved_bottom_price == Decimal("Infinity") and
            saved_peak_price == Decimal(0.0) and
            saved_price_spike_or_crash_pct == Decimal(0.0) and
            saved_bottom_rsi == Decimal(50.0) and
            saved_peak_rsi == Decimal(50.0) and
            saved_bottom_stoch == Decimal(50.0) and
            saved_peak_stoch == Decimal(50.0) and
            self.rsi_reversal_counter == 0 and
            self.stoch_reversal_counter == 0
        )

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

        timestamp_series: pd.Series = self.processed_data["timestamp"]
        recent_timestamps = timestamp_series.iloc[-candle_count:].reset_index(drop=True)
        saved_peak_price, _ = self.saved_peak_price

        if peak_price > saved_peak_price:
            peak_price_timestamp = recent_timestamps.iloc[peak_price_index]
            self.save_peak_price(peak_price, peak_price_timestamp)

        saved_peak_price, _ = self.saved_peak_price

        bottom_price = Decimal(recent_lows.iloc[0:peak_price_index].min())
        price_delta_pct: Decimal = (saved_peak_price - bottom_price) / bottom_price * 100
        is_spiking = price_delta_pct > self.config.min_price_delta_pct_to_open

        if is_spiking:
            self.logger().info(f"is_price_spiking() | current_price:{self.get_current_close()} | peak_price_index:{peak_price_index} | saved_peak_price:{saved_peak_price} | bottom_price:{bottom_price} | price_delta_pct:{price_delta_pct}")
            self.save_price_spike_or_crash_pct(price_delta_pct, self.get_market_data_provider_time())

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

        timestamp_series: pd.Series = self.processed_data["timestamp"]
        recent_timestamps = timestamp_series.iloc[-candle_count:].reset_index(drop=True)
        saved_bottom_price, _ = self.saved_bottom_price

        if bottom_price < saved_bottom_price:
            bottom_price_timestamp = recent_timestamps.iloc[bottom_price_index]
            self.save_bottom_price(bottom_price, bottom_price_timestamp)

        saved_bottom_price, _ = self.saved_bottom_price

        peak_price = Decimal(recent_highs.iloc[0:bottom_price_index].max())
        price_delta_pct: Decimal = (peak_price - saved_bottom_price) / saved_bottom_price * 100
        is_crashing = price_delta_pct > self.config.min_price_delta_pct_to_open

        if is_crashing:
            self.logger().info(f"is_price_crashing() | current_price:{self.get_current_close()} | bottom_price_index:{bottom_price_index} | saved_bottom_price:{saved_bottom_price} | peak_price:{peak_price} | price_delta_pct:{price_delta_pct}")
            self.save_price_spike_or_crash_pct(price_delta_pct, self.get_market_data_provider_time())

        return is_crashing

    def is_price_spike_a_reversal(self, candle_count: int) -> bool:
        candle_end_index: int = -candle_count
        candle_start_index: int = candle_end_index * 2

        high_series: pd.Series = self.processed_data["high"]
        previous_highs = high_series.iloc[candle_start_index:candle_end_index]  # last one excluded

        previous_peak = Decimal(previous_highs.max())
        saved_peak_price, _ = self.saved_peak_price
        delta_pct: Decimal = (saved_peak_price - previous_peak) / saved_peak_price * 100

        self.logger().info(f"is_price_spike_a_reversal() | saved_peak_price:{saved_peak_price} | previous_peak:{previous_peak} | delta_pct:{delta_pct}")

        return delta_pct < self.config.min_price_delta_pct_to_open * Decimal(0.75)

    def is_price_crash_a_reversal(self, candle_count: int) -> bool:
        candle_end_index: int = -candle_count
        candle_start_index: int = candle_end_index * 2

        low_series: pd.Series = self.processed_data["low"]
        previous_lows = low_series.iloc[candle_start_index:candle_end_index]

        previous_bottom = Decimal(previous_lows.min())
        saved_bottom_price, _ = self.saved_bottom_price
        delta_pct: Decimal = (previous_bottom - saved_bottom_price) / saved_bottom_price * 100

        self.logger().info(f"is_price_crash_a_reversal() | saved_bottom_price:{saved_bottom_price} | previous_bottom:{previous_bottom} | delta_pct:{delta_pct}")

        return delta_pct < self.config.min_price_delta_pct_to_open * Decimal(0.75)

    def has_rsi_peaked(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_40"]
        recent_rsis = rsi_series.iloc[-candle_count:].reset_index(drop=True)

        peak_rsi = Decimal(recent_rsis.max())
        peak_rsi_index = recent_rsis.idxmax()

        if peak_rsi_index == 0:
            return False

        if peak_rsi < 63:
            return False

        timestamp_series: pd.Series = self.processed_data["timestamp"]
        recent_timestamps = timestamp_series.iloc[-candle_count:].reset_index(drop=True)
        saved_peak_rsi, _ = self.saved_peak_rsi

        if peak_rsi > saved_peak_rsi:
            peak_rsi_timestamp = recent_timestamps.iloc[peak_rsi_index]
            self.save_peak_rsi(peak_rsi, peak_rsi_timestamp)

        saved_peak_rsi, _ = self.saved_peak_rsi

        rsi_decrement: Decimal = compute_rsi_pullback_difference(saved_peak_rsi)
        rsi_threshold: Decimal = saved_peak_rsi - rsi_decrement
        current_rsi = self.get_current_rsi(40)

        self.logger().info(f"has_rsi_peaked() | saved_peak_rsi:{saved_peak_rsi} | current_rsi:{current_rsi} | rsi_decrement:{rsi_decrement} | rsi_threshold:{rsi_threshold}")

        if current_rsi > rsi_threshold:
            self.rsi_reversal_counter = 0
            self.logger().info("has_rsi_peaked() | resetting self.rsi_reversal_counter to 0")
            return False

        self.rsi_reversal_counter += 1
        self.logger().info(f"has_rsi_peaked() | incremented self.rsi_reversal_counter to:{self.rsi_reversal_counter}")

        if self.rsi_reversal_counter < 5:
            return False

        too_late_threshold: Decimal = rsi_threshold - rsi_decrement

        return current_rsi > too_late_threshold

    def has_rsi_bottomed(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_40"]
        recent_rsis = rsi_series.iloc[-candle_count:].reset_index(drop=True)

        bottom_rsi = Decimal(recent_rsis.min())
        bottom_rsi_index = recent_rsis.idxmin()

        if bottom_rsi_index == 0:
            return False

        if bottom_rsi > 37:
            return False

        timestamp_series: pd.Series = self.processed_data["timestamp"]
        recent_timestamps = timestamp_series.iloc[-candle_count:].reset_index(drop=True)
        saved_bottom_rsi, _ = self.saved_bottom_rsi

        if bottom_rsi < saved_bottom_rsi:
            bottom_rsi_timestamp = recent_timestamps.iloc[bottom_rsi_index]
            self.save_bottom_rsi(bottom_rsi, bottom_rsi_timestamp)

        saved_bottom_rsi, _ = self.saved_bottom_rsi

        rsi_increment: Decimal = compute_rsi_pullback_difference(saved_bottom_rsi)
        rsi_threshold: Decimal = saved_bottom_rsi + rsi_increment
        current_rsi = self.get_current_rsi(40)

        self.logger().info(f"has_rsi_bottomed() | saved_bottom_rsi:{saved_bottom_rsi} | current_rsi:{current_rsi} | rsi_increment:{rsi_increment} | rsi_threshold:{rsi_threshold}")

        if current_rsi < rsi_threshold:
            self.rsi_reversal_counter = 0
            self.logger().info("has_rsi_bottomed() | resetting self.rsi_reversal_counter to 0")
            return False

        self.rsi_reversal_counter += 1
        self.logger().info(f"has_rsi_bottomed() | incremented self.rsi_reversal_counter to:{self.rsi_reversal_counter}")

        if self.rsi_reversal_counter < 5:
            return False

        too_late_threshold: Decimal = rsi_threshold + rsi_increment

        return current_rsi < too_late_threshold

    def is_price_under_ma(self) -> bool:
        current_price: Decimal = self.get_current_close()
        current_ma: Decimal = self.get_current_ma()

        self.logger().info(f"is_price_under_ma() | current_price:{current_price} | current_ma:{current_ma}")

        return current_price < current_ma

    def is_price_over_ma(self) -> bool:
        current_price: Decimal = self.get_current_close()
        current_ma: Decimal = self.get_current_ma()

        self.logger().info(f"is_price_over_ma() | current_price:{current_price} | current_ma:{current_ma}")

        return current_price > current_ma

    def has_stoch_reversed_for_sell(self) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_10_k"]
        recent_stochs = stoch_series.iloc[-5:].reset_index(drop=True)

        bottom_stoch: Decimal = Decimal(recent_stochs.min())
        bottom_stoch_index = recent_stochs.idxmin()

        if bottom_stoch_index == 0:
            return False

        if bottom_stoch > 50:
            return False

        timestamp_series: pd.Series = self.processed_data["timestamp"]
        recent_timestamps = timestamp_series.iloc[-5:].reset_index(drop=True)
        saved_bottom_stoch, _ = self.saved_bottom_stoch

        if bottom_stoch < saved_bottom_stoch:
            bottom_stoch_timestamp = recent_timestamps.iloc[bottom_stoch_index]
            self.save_bottom_stoch(bottom_stoch, bottom_stoch_timestamp)

        saved_bottom_stoch, _ = self.saved_bottom_stoch

        stoch_threshold: Decimal = saved_bottom_stoch + 3
        current_stoch = self.get_current_stoch(10)

        self.logger().info(f"has_stoch_reversed_for_sell() | saved_bottom_stoch:{saved_bottom_stoch} | current_stoch:{current_stoch} | stoch_threshold:{stoch_threshold} | current_price:{self.get_current_close()}")

        if current_stoch < stoch_threshold:
            self.stoch_reversal_counter = 0
            self.logger().info("has_stoch_reversed_for_sell() | resetting self.stoch_reversal_counter to 0")
            return False

        self.stoch_reversal_counter += 1
        self.logger().info(f"has_stoch_reversed_for_sell() | incremented self.stoch_reversal_counter to:{self.stoch_reversal_counter}")

        return self.stoch_reversal_counter > 4

    def has_stoch_reversed_for_buy(self) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_10_k"]
        recent_stochs = stoch_series.iloc[-5:].reset_index(drop=True)

        peak_stoch: Decimal = Decimal(recent_stochs.max())
        peak_stoch_index = recent_stochs.idxmax()

        if peak_stoch_index == 0:
            return False

        if peak_stoch < 50:
            return False

        timestamp_series: pd.Series = self.processed_data["timestamp"]
        recent_timestamps = timestamp_series.iloc[-5:].reset_index(drop=True)
        saved_peak_stoch, _ = self.saved_peak_stoch

        if peak_stoch > saved_peak_stoch:
            peak_stoch_timestamp = recent_timestamps.iloc[peak_stoch_index]
            self.save_peak_stoch(peak_stoch, peak_stoch_timestamp)

        saved_peak_stoch, _ = self.saved_peak_stoch

        stoch_threshold: Decimal = saved_peak_stoch - 3
        current_stoch = self.get_current_stoch(10)

        self.logger().info(f"has_stoch_reversed_for_buy() | saved_peak_stoch:{saved_peak_stoch} | current_stoch:{current_stoch} | stoch_threshold:{stoch_threshold} | current_price:{self.get_current_close()}")

        if current_stoch > stoch_threshold:
            self.stoch_reversal_counter = 0
            self.logger().info("has_stoch_reversed_for_buy() | resetting self.stoch_reversal_counter to 0")
            return False

        self.stoch_reversal_counter += 1
        self.logger().info(f"has_stoch_reversed_for_buy() | incremented self.stoch_reversal_counter to:{self.stoch_reversal_counter}")

        return self.stoch_reversal_counter > 4
