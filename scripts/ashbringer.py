import asyncio
from decimal import Decimal
from typing import Dict, List, Tuple

import pandas as pd
from pandas_ta import sma, stoch

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.ashbringer_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Generate config file: create --script-config ashbringer_config
# Start the bot: start --script ashbringer.py --conf conf_ashbringer_GOAT.yml
#                start --script ashbringer.py --conf conf_ashbringer_BOME.yml
#                start --script ashbringer.py --conf conf_ashbringer_CHILLGUY.yml
#                start --script ashbringer.py --conf conf_ashbringer_FLOKI.yml
#                start --script ashbringer.py --conf conf_ashbringer_MOODENG.yml
#                start --script ashbringer.py --conf conf_ashbringer_NEIRO.yml
#                start --script ashbringer.py --conf conf_ashbringer_PNUT.yml
#                start --script ashbringer.py --conf conf_ashbringer_POPCAT.yml
#                start --script ashbringer.py --conf conf_ashbringer_WIF.yml
# Quickstart script: -p=a -f ashbringer.py -c conf_ashbringer_GOAT.yml

ORDER_REF_TREND_REV: str = "TrendReversal"
ORDER_REF_MA_CHANNEL: str = "MaChannel"
CANDLE_COUNT_FOR_STOCH_REV_AND_MA_CHANNEL: int = 5


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

    def get_triple_barrier(self, ref: str) -> TripleBarrier:
        if ref == ORDER_REF_TREND_REV:
            return TripleBarrier(
                open_order_type=OrderType.MARKET,
                stop_loss=self.config.trend_rev_stop_loss_pct / 100
            )

        saved_price_spike_or_crash_pct, _ = self.saved_price_spike_or_crash_pct
        take_profit_pct: Decimal = saved_price_spike_or_crash_pct / 2
        stop_loss_pct: Decimal = take_profit_pct / 2

        return TripleBarrier(
            open_order_type=OrderType.MARKET,
            take_profit=take_profit_pct / 100,
            stop_loss=stop_loss_pct / 100,
            time_limit=12 * 60
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

        candles_df["RSI_20"] = candles_df.ta.rsi(length=20)
        candles_df["RSI_40"] = candles_df.ta.rsi(length=40)

        # Calling the lower-level function, because the one in core.py has a bug in the argument names
        stoch_40_df = stoch(
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            k=40,
            d=1,
            smooth_k=1
        )

        candles_df["STOCH_40_k"] = stoch_40_df["STOCHk_40_1_1"]

        candles_df["SMA_10_h"] = sma(close=candles_df["high"], length=10)
        candles_df["SMA_10_l"] = sma(close=candles_df["low"], length=10)

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        self.check_context(CANDLE_COUNT_FOR_STOCH_REV_AND_MA_CHANNEL + 1)
        self.create_actions_proposal_trend_reversal()
        self.create_actions_proposal_ma_channel()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()
        self.stop_actions_proposal_trend_reversal()
        self.stop_actions_proposal_ma_channel()

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
                    "RSI_20",
                    "RSI_40",
                    "STOCH_40_k",
                    "SMA_10_h",
                    "SMA_10_l"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display], table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Trend Reversal start/stop action proposals
    #

    def create_actions_proposal_trend_reversal(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_TREND_REV)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_trend_reversal_order(TradeType.BUY, active_orders):
            triple_barrier = self.get_triple_barrier(ORDER_REF_TREND_REV)

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.BUY, self.get_mid_price(), triple_barrier, self.config.amount_quote, ORDER_REF_TREND_REV)
            )

    def can_create_trend_reversal_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_TREND_REV, 0):
            return False

        if len(active_tracked_orders) > 0:
            return False

        history_candle_count: int = 25

        if (
            self.is_recent_rsi_low_enough(5) and
            self.are_candles_green(3) and
            self.is_price_crashing(history_candle_count) and
            self.is_price_bottom_recent(history_candle_count, 5)
        ):
            self.logger().info("can_create_ma_channel_order() > Opening Buy Trend Reversal")
            return True

        return False

    def stop_actions_proposal_trend_reversal(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_TREND_REV)

        if len(filled_buy_orders) > 0:
            if self.has_stoch_reversed_for_buy(CANDLE_COUNT_FOR_STOCH_REV_AND_MA_CHANNEL):
                self.logger().info("stop_actions_proposal_rev() > Closing Buy Trend Reversal")
                self.market_close_orders(filled_buy_orders, CloseType.COMPLETED)

    #
    # MA Channel start/stop action proposals
    #

    def create_actions_proposal_ma_channel(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MA_CHANNEL)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_ma_channel_order(TradeType.SELL, active_orders):
            triple_barrier = self.get_triple_barrier(ORDER_REF_MA_CHANNEL)

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.SELL, self.get_mid_price(), triple_barrier, self.config.amount_quote, ORDER_REF_MA_CHANNEL)
            )

        if self.can_create_ma_channel_order(TradeType.BUY, active_orders):
            triple_barrier = self.get_triple_barrier(ORDER_REF_MA_CHANNEL)

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.BUY, self.get_mid_price(), triple_barrier, self.config.amount_quote, ORDER_REF_MA_CHANNEL)
            )

    def can_create_ma_channel_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_MA_CHANNEL, 0):
            return False

        if len(active_tracked_orders) > 0:
            return False

        candle_index_candle_crossing_ma: int = -CANDLE_COUNT_FOR_STOCH_REV_AND_MA_CHANNEL
        candle_count_outside_ma: int = CANDLE_COUNT_FOR_STOCH_REV_AND_MA_CHANNEL - 2

        if side == TradeType.SELL:
            if (
                self.did_price_cross_under_mal(candle_index_candle_crossing_ma) and
                self.are_candles_fully_below_mal(candle_count_outside_ma) and
                self.are_candles_red(candle_count_outside_ma) and
                not self.is_recent_rsi_too_low(3) and
                self.is_price_continuing_down_trend()
            ):
                self.logger().info("can_create_ma_channel_order() > Opening Sell MA-C")
                self.save_ma_channel_sell_price_delta_pct(CANDLE_COUNT_FOR_STOCH_REV_AND_MA_CHANNEL)
                return True

            return False

        if (
                self.did_price_cross_over_mah(candle_index_candle_crossing_ma) and
                self.are_candles_fully_above_mah(candle_count_outside_ma) and
                self.are_candles_green(candle_count_outside_ma) and
                not self.is_recent_rsi_too_high(3) and
                self.is_price_continuing_up_trend()
        ):
            self.logger().info("can_create_ma_channel_order() > Opening Buy MA-C")
            self.save_ma_channel_buy_price_delta_pct(CANDLE_COUNT_FOR_STOCH_REV_AND_MA_CHANNEL)
            return True

        return False

    def stop_actions_proposal_ma_channel(self):
        pass

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

    def get_current_stoch(self) -> Decimal:
        stoch_series: pd.Series = self.processed_data["STOCH_40_k"]
        return Decimal(stoch_series.iloc[-1])

    def get_current_mah(self) -> Decimal:
        smah_series: pd.Series = self.processed_data["SMA_10_h"]
        return Decimal(smah_series.iloc[-1])

    def get_current_mal(self) -> Decimal:
        smal_series: pd.Series = self.processed_data["SMA_10_l"]
        return Decimal(smal_series.iloc[-1])

    #
    # Context functions
    #

    def reset_context(self):
        self.save_peak_stoch(Decimal(50.0), self.get_market_data_provider_time())

        self.save_price_spike_or_crash_pct(Decimal(0.0), self.get_market_data_provider_time())

        self.stoch_reversal_counter: int = 0
        self.trend_continuation_counter: int = 0

    def save_peak_stoch(self, peak_stoch: Decimal, timestamp: float):
        self.saved_peak_stoch: Tuple[Decimal, float] = peak_stoch, timestamp

    def save_price_spike_or_crash_pct(self, price_spike_or_crash_pct: Decimal, timestamp: float):
        self.saved_price_spike_or_crash_pct: Tuple[Decimal, float] = price_spike_or_crash_pct, timestamp

    def check_context(self, lifetime_minutes: int):
        saved_peak_stoch, saved_peak_stoch_timestamp = self.saved_peak_stoch

        saved_price_spike_or_crash_pct, saved_price_spike_or_crash_pct_timestamp = self.saved_price_spike_or_crash_pct

        all_timestamps: List[float] = [
            saved_peak_stoch_timestamp,
            saved_price_spike_or_crash_pct_timestamp
        ]

        last_acceptable_timestamp = self.get_market_data_provider_time() - lifetime_minutes * 60

        is_any_outdated: bool = any(timestamp < last_acceptable_timestamp for timestamp in all_timestamps)

        if is_any_outdated and not self.is_context_default():
            self.logger().info("check_context() | One of the context vars is outdated")
            self.reset_context()

    def is_context_default(self) -> bool:
        saved_peak_stoch, _ = self.saved_peak_stoch

        saved_price_spike_or_crash_pct, _ = self.saved_price_spike_or_crash_pct

        return (
            saved_peak_stoch == Decimal(50.0) and
            saved_price_spike_or_crash_pct == Decimal(0.0) and
            self.stoch_reversal_counter == 0 and
            self.trend_continuation_counter == 0
        )

    #
    # Functions common to both strategies
    #

    def are_candles_green(self, candle_count: int) -> bool:
        candle_start_index: int = -candle_count - 1

        open_series: pd.Series = self.processed_data["open"]
        recent_opens = open_series.iloc[candle_start_index:-1].reset_index(drop=True)

        close_series: pd.Series = self.processed_data["close"]
        recent_closes = close_series.iloc[candle_start_index:-1].reset_index(drop=True)

        return all(recent_closes[i] > recent_opens[i] for i in range(len(recent_opens)))

    def are_candles_red(self, candle_count: int) -> bool:
        candle_start_index: int = -candle_count - 1

        open_series: pd.Series = self.processed_data["open"]
        recent_opens = open_series.iloc[candle_start_index:-1].reset_index(drop=True)

        close_series: pd.Series = self.processed_data["close"]
        recent_closes = close_series.iloc[candle_start_index:-1].reset_index(drop=True)

        return all(recent_closes[i] < recent_opens[i] for i in range(len(recent_opens)))

    #
    # Trend Reversal functions
    #

    def is_recent_rsi_low_enough(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_20"]
        recent_rsis = rsi_series.iloc[-candle_count:].reset_index(drop=True)
        bottom_rsi: Decimal = Decimal(recent_rsis.min())

        if bottom_rsi < 29:
            self.logger().info(f"is_recent_rsi_low_enough() | bottom_rsi:{bottom_rsi}")

        return bottom_rsi < 29

    def is_price_crashing(self, candle_count: int) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-candle_count:].reset_index(drop=True)

        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-candle_count:].reset_index(drop=True)

        bottom_price = Decimal(recent_lows.min())
        bottom_price_index = recent_lows.idxmin()

        if bottom_price_index == 0:
            return False

        peak_price = Decimal(recent_highs.iloc[0:bottom_price_index].max())
        price_delta_pct: Decimal = (peak_price - bottom_price) / bottom_price * 100
        is_crashing = self.config.min_price_delta_pct_to_open_trend_rev < price_delta_pct

        if is_crashing:
            self.logger().info(f"is_price_crashing() | current_price:{self.get_current_close()} | bottom_price_index:{bottom_price_index} | bottom_price:{bottom_price} | peak_price:{peak_price} | price_delta_pct:{price_delta_pct}")

        return is_crashing

    def is_price_bottom_recent(self, history_candle_count: int, recent_candle_count: int) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        lows = low_series.iloc[-history_candle_count:].reset_index(drop=True)

        bottom_price_index = lows.idxmin()

        self.logger().info(f"is_price_bottom_recent() | bottom_price_index:{bottom_price_index}")

        return bottom_price_index > history_candle_count - recent_candle_count

    def has_stoch_reversed_for_buy(self, candle_count: int) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_40_k"]
        recent_stochs = stoch_series.iloc[-candle_count:].reset_index(drop=True)

        peak_stoch: Decimal = Decimal(recent_stochs.max())
        peak_stoch_index = recent_stochs.idxmax()

        if peak_stoch_index == 0:
            return False

        if peak_stoch < 75:
            return False

        timestamp_series: pd.Series = self.processed_data["timestamp"]
        recent_timestamps = timestamp_series.iloc[-candle_count:].reset_index(drop=True)
        saved_peak_stoch, _ = self.saved_peak_stoch

        if peak_stoch > saved_peak_stoch:
            peak_stoch_timestamp = recent_timestamps.iloc[peak_stoch_index]
            self.save_peak_stoch(peak_stoch, peak_stoch_timestamp)

        saved_peak_stoch, _ = self.saved_peak_stoch

        stoch_threshold: Decimal = saved_peak_stoch - 3
        current_stoch = self.get_current_stoch()

        self.logger().info(f"has_stoch_reversed_for_buy() | saved_peak_stoch:{saved_peak_stoch} | current_stoch:{current_stoch} | stoch_threshold:{stoch_threshold} | current_price:{self.get_current_close()}")

        if current_stoch > stoch_threshold:
            self.stoch_reversal_counter = 0
            self.logger().info("has_stoch_reversed_for_buy() | resetting self.stoch_reversal_counter to 0")
            return False

        self.stoch_reversal_counter += 1
        self.logger().info(f"has_stoch_reversed_for_buy() | incremented self.stoch_reversal_counter to:{self.stoch_reversal_counter}")

        return self.stoch_reversal_counter > 4

    #
    # MA Channel functions
    #

    def did_price_cross_under_mal(self, candle_index: int) -> bool:
        high_series: pd.Series = self.processed_data["high"]
        high: Decimal = Decimal(high_series.iloc[candle_index])

        close_series: pd.Series = self.processed_data["close"]
        close: Decimal = Decimal(close_series.iloc[candle_index])

        mal_series: pd.Series = self.processed_data["SMA_10_l"]
        mal: Decimal = Decimal(mal_series.iloc[candle_index])

        return close < mal < high

    def did_price_cross_over_mah(self, candle_index: int) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        low: Decimal = Decimal(low_series.iloc[candle_index])

        close_series: pd.Series = self.processed_data["close"]
        close: Decimal = Decimal(close_series.iloc[candle_index])

        mah_series: pd.Series = self.processed_data["SMA_10_h"]
        mah: Decimal = Decimal(mah_series.iloc[candle_index])

        return low < mah < close

    def are_candles_fully_below_mal(self, candle_count: int) -> bool:
        candle_start_index: int = -candle_count - 1

        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[candle_start_index:-1].reset_index(drop=True)

        mal_series: pd.Series = self.processed_data["SMA_10_l"]
        recent_mals = mal_series.iloc[candle_start_index:-1].reset_index(drop=True)

        return all(recent_highs[i] < recent_mals[i] for i in range(len(recent_highs)))

    def are_candles_fully_above_mah(self, candle_count: int) -> bool:
        candle_start_index: int = -candle_count - 1

        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[candle_start_index:-1].reset_index(drop=True)

        mah_series: pd.Series = self.processed_data["SMA_10_h"]
        recent_mahs = mah_series.iloc[candle_start_index:-1].reset_index(drop=True)

        return all(recent_lows[i] > recent_mahs[i] for i in range(len(recent_lows)))

    def is_recent_rsi_too_low(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_40"]
        recent_rsis = rsi_series.iloc[-candle_count:].reset_index(drop=True)
        bottom_rsi: Decimal = Decimal(recent_rsis.min())

        if bottom_rsi < 40:
            self.logger().info(f"is_recent_rsi_too_low() | bottom_rsi:{bottom_rsi}")

        return bottom_rsi < 40

    def is_recent_rsi_too_high(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_40"]
        recent_rsis = rsi_series.iloc[-candle_count:].reset_index(drop=True)
        peak_rsi: Decimal = Decimal(recent_rsis.max())

        if peak_rsi > 60:
            self.logger().info(f"is_recent_rsi_too_high() | peak_rsi:{peak_rsi}")

        return peak_rsi > 60

    def is_price_continuing_down_trend(self) -> bool:
        current_price: Decimal = self.get_current_close()

        close_series: pd.Series = self.processed_data["close"]
        last_close = Decimal(close_series.iloc[-2])

        self.logger().info(f"is_price_continuing_down_trend() | current_price:{current_price} | last_close:{last_close}")

        if current_price > last_close:
            self.trend_continuation_counter = 0
            self.logger().info("is_price_continuing_down_trend() | resetting self.trend_continuation_counter to 0")
            return False

        self.trend_continuation_counter += 1
        self.logger().info(f"is_price_continuing_down_trend() | incremented self.trend_continuation_counter to:{self.trend_continuation_counter}")

        return self.trend_continuation_counter > 9

    def is_price_continuing_up_trend(self) -> bool:
        current_price: Decimal = self.get_current_close()

        close_series: pd.Series = self.processed_data["close"]
        last_close = Decimal(close_series.iloc[-2])

        self.logger().info(f"is_price_continuing_up_trend() | current_price:{current_price} | last_close:{last_close}")

        if current_price < last_close:
            self.trend_continuation_counter = 0
            self.logger().info("is_price_continuing_up_trend() | resetting self.trend_continuation_counter to 0")
            return False

        self.trend_continuation_counter += 1
        self.logger().info(f"is_price_continuing_up_trend() | incremented self.trend_continuation_counter to:{self.trend_continuation_counter}")

        return self.trend_continuation_counter > 9

    def save_ma_channel_sell_price_delta_pct(self, candle_count: int):
        high_series: pd.Series = self.processed_data["high"]
        high: Decimal = Decimal(high_series.iloc[-candle_count])

        current_price = self.get_current_close()

        delta_pct: Decimal = (high - current_price) / current_price * 100

        self.logger().info(f"save_ma_channel_sell_price_delta_pct() | high:{high} | current_price:{current_price} | delta_pct:{delta_pct}")

        self.save_price_spike_or_crash_pct(delta_pct, self.get_market_data_provider_time())

    def save_ma_channel_buy_price_delta_pct(self, candle_count: int):
        low_series: pd.Series = self.processed_data["low"]
        low: Decimal = Decimal(low_series.iloc[-candle_count])

        current_price = self.get_current_close()

        delta_pct: Decimal = (current_price - low) / current_price * 100

        self.logger().info(f"save_ma_channel_buy_price_delta_pct() | low:{low} | current_price:{current_price} | delta_pct:{delta_pct}")

        self.save_price_spike_or_crash_pct(delta_pct, self.get_market_data_provider_time())
