import asyncio
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
from scripts.ashbringer_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import timestamp_to_iso
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Generate config file: create --script-config ashbringer_config
# Start the bot: start --script ashbringer.py --conf conf_ashbringer_GOAT.yml
#                start --script ashbringer.py --conf conf_ashbringer_AI16Z.yml
#                start --script ashbringer.py --conf conf_ashbringer_AIXBT.yml
#                start --script ashbringer.py --conf conf_ashbringer_FARTCOIN.yml
#                start --script ashbringer.py --conf conf_ashbringer_MOODENG.yml
#                start --script ashbringer.py --conf conf_ashbringer_PENGU.yml
#                start --script ashbringer.py --conf conf_ashbringer_PNUT.yml
#                start --script ashbringer.py --conf conf_ashbringer_POPCAT.yml
#                start --script ashbringer.py --conf conf_ashbringer_WIF.yml
# Quickstart script: -p=a -f ashbringer.py -c conf_ashbringer_GOAT.yml

ORDER_REF_TREND_REVERSAL: str = "TrendReversal"
CANDLE_COUNT_FOR_TR_STOCH_REVERSAL: int = 3
CANDLE_COUNT_FOR_TR_CONTEXT: int = CANDLE_COUNT_FOR_TR_STOCH_REVERSAL
CANDLE_DURATION_MINUTES: int = 3


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()
        self.reset_tr_context()

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
            open_order_type=OrderType.MARKET,
            stop_loss=self.compute_tr_sl_pct(4) / 100
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

        candles_df["SMA_8"] = candles_df.ta.sma(length=8)

        # Calling the lower-level function, because the one in core.py has a bug in the argument names
        stoch_13_df = stoch(
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            k=13,
            d=1,
            smooth_k=1
        )

        candles_df["STOCH_13_k"] = stoch_13_df["STOCHk_13_1_1"]

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        tr_context_lifetime_minutes: int = CANDLE_COUNT_FOR_TR_CONTEXT * CANDLE_DURATION_MINUTES + 1
        self.check_tr_context(tr_context_lifetime_minutes)

        self.create_actions_proposal_trend_reversal()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()
        self.stop_actions_proposal_trend_reversal()

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
                    "RSI_20",
                    "SMA_8",
                    "STOCH_13_k"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display], table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Trend Reversal start/stop action proposals
    #

    def create_actions_proposal_trend_reversal(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_TREND_REVERSAL)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_trend_reversal_order(TradeType.BUY, active_orders):
            triple_barrier = self.get_triple_barrier()

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.BUY, self.get_mid_price(), triple_barrier, self.config.amount_quote, ORDER_REF_TREND_REVERSAL)
            )

    def can_create_trend_reversal_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_TREND_REVERSAL, 5):
            return False

        if len(active_tracked_orders) > 0:
            return False

        history_candle_count: int = 20

        if (
            self.is_recent_rsi_low_enough(3) and
            self.is_price_crashing(history_candle_count) and
            self.is_price_bottom_recent(history_candle_count, 3) and
            self.did_price_rebound(history_candle_count)
        ):
            self.logger().info(f"can_create_trend_reversal_order() > Opening Trend Reversal Buy at {self.get_current_close()} | Current Stoch 13:{self.get_current_stoch(13)}")
            return True

        return False

    def stop_actions_proposal_trend_reversal(self):
        _, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_TREND_REVERSAL)

        if len(filled_buy_orders) > 0:
            if self.is_price_over_ma(8) and self.has_stoch_reversed_for_tr_buy(CANDLE_COUNT_FOR_TR_STOCH_REVERSAL, 13):
                self.logger().info(f"stop_actions_proposal_trend_reversal() > Closing Trend Reversal Buy at {self.get_current_close()}")
                self.market_close_orders(filled_buy_orders, CloseType.COMPLETED)
                self.reset_tr_context()

    #
    # Getters on `self.processed_data[]`
    #

    def get_current_close(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[-1])

    def get_current_ma(self, length: int) -> Decimal:
        sma_series: pd.Series = self.processed_data[f"SMA_{length}"]
        return Decimal(sma_series.iloc[-1])

    def get_current_stoch(self, length: int) -> Decimal:
        stoch_series: pd.Series = self.processed_data[f"STOCH_{length}_k"]
        return Decimal(stoch_series.iloc[-1])

    #
    # Trend Reversal Context
    #

    def reset_tr_context(self):
        self.save_tr_price_change_pct(Decimal(0.0), self.get_market_data_provider_time())
        self.save_tr_peak_stoch(Decimal(50.0), self.get_market_data_provider_time())

        self.tr_price_reversal_counter: int = 0
        self.tr_stoch_reversal_counter: int = 0
        self.logger().info("TR context is reset")

    def save_tr_price_change_pct(self, price_change_pct: Decimal, timestamp: float):
        self.saved_tr_price_change_pct: Tuple[Decimal, float] = price_change_pct, timestamp

    def save_tr_peak_stoch(self, peak_stoch: Decimal, timestamp: float):
        self.saved_tr_peak_stoch: Tuple[Decimal, float] = peak_stoch, timestamp

    def check_tr_context(self, lifetime_minutes: int):
        _, saved_price_change_pct_timestamp = self.saved_tr_price_change_pct
        _, saved_peak_stoch_timestamp = self.saved_tr_peak_stoch

        most_recent_timestamp: float = max([
            saved_price_change_pct_timestamp,
            saved_peak_stoch_timestamp
        ])

        last_acceptable_timestamp = self.get_market_data_provider_time() - lifetime_minutes * 60

        is_outdated: bool = most_recent_timestamp < last_acceptable_timestamp

        if is_outdated and not self.is_tr_context_default():
            self.logger().info("check_tr_context() | Resetting outdated context")
            self.reset_tr_context()

    def is_tr_context_default(self) -> bool:
        saved_price_change_pct, _ = self.saved_tr_price_change_pct
        saved_peak_stoch, _ = self.saved_tr_peak_stoch

        return (
            saved_price_change_pct == Decimal(0.0) and
            saved_peak_stoch == Decimal(50.0) and
            self.tr_price_reversal_counter == 0 and
            self.tr_stoch_reversal_counter == 0
        )

    #
    # Trend Reversal functions
    #

    def get_current_bottom(self, candle_count: int) -> Decimal:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-candle_count:].reset_index(drop=True)

        return Decimal(recent_lows.min())

    def is_recent_rsi_low_enough(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_20"]
        recent_rsis = rsi_series.iloc[-candle_count:].reset_index(drop=True)
        bottom_rsi: Decimal = Decimal(recent_rsis.min())

        if bottom_rsi < 30:
            self.logger().info(f"is_recent_rsi_low_enough() | bottom_rsi:{bottom_rsi}")

        return bottom_rsi < 30

    def is_price_crashing(self, candle_count: int) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-candle_count:].reset_index(drop=True)
        bottom_price = Decimal(recent_lows.min())
        bottom_price_index = recent_lows.idxmin()

        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-candle_count:].reset_index(drop=True)
        peak_price = Decimal(recent_highs.max())
        peak_price_index = recent_highs.idxmax()

        if peak_price_index > bottom_price_index:
            return False

        price_delta_pct: Decimal = (peak_price - bottom_price) / bottom_price * 100
        is_crashing = self.config.min_price_delta_pct_to_open_tr < price_delta_pct

        if is_crashing:
            self.logger().info(f"is_price_crashing() | bottom_price_index:{bottom_price_index} | bottom_price:{bottom_price} | peak_price_index:{peak_price_index} | peak_price:{peak_price}")
            self.logger().info(f"is_price_crashing() | current_price:{self.get_current_close()} | price_delta_pct:{price_delta_pct}")
            self.save_tr_price_change_pct(price_delta_pct, self.get_market_data_provider_time())

        return is_crashing

    def is_price_bottom_recent(self, history_candle_count: int, recent_candle_count: int) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        lows = low_series.iloc[-history_candle_count:].reset_index(drop=True)

        bottom_price_index = lows.idxmin()

        self.logger().info(f"is_price_bottom_recent() | bottom_price_index:{bottom_price_index}")

        return bottom_price_index >= history_candle_count - recent_candle_count  # >= 20 - 3

    def did_price_rebound(self, candle_count: int) -> bool:
        saved_price_change_pct, _ = self.saved_tr_price_change_pct
        price_threshold_pct: Decimal = saved_price_change_pct / 8
        price_top_limit_pct: Decimal = saved_price_change_pct / 4

        bottom_price = self.get_current_bottom(candle_count)
        price_threshold: Decimal = bottom_price * (1 + price_threshold_pct / 100)
        price_top_limit: Decimal = bottom_price * (1 + price_top_limit_pct / 100)

        current_price: Decimal = self.get_current_close()

        self.logger().info(f"did_price_rebound() | saved_price_change_pct:{saved_price_change_pct} | bottom_price:{bottom_price} | current_price:{current_price}")
        self.logger().info(f"did_price_rebound() | price_threshold_pct:{price_threshold_pct} | price_threshold:{price_threshold}")
        self.logger().info(f"did_price_rebound() | price_top_limit_pct:{price_top_limit_pct} | price_top_limit:{price_top_limit}")

        if not price_threshold < current_price < price_top_limit:
            self.tr_price_reversal_counter = 0
            self.logger().info("did_price_rebound() | resetting self.tr_price_reversal_counter to 0")
            return False

        self.tr_price_reversal_counter += 1
        self.logger().info(f"did_price_rebound() | incremented self.tr_price_reversal_counter to:{self.tr_price_reversal_counter}")

        return self.tr_price_reversal_counter > 119

    def compute_tr_sl_pct(self, candle_count: int) -> Decimal:
        bottom_price = self.get_current_bottom(candle_count)
        current_price: Decimal = self.get_current_close()

        delta_pct_with_bottom: Decimal = (current_price - bottom_price) / current_price * 100
        sl_pct: Decimal = delta_pct_with_bottom * Decimal(0.8)

        self.logger().info(f"compute_tr_sl_pct() | bottom_price:{bottom_price} | current_price:{current_price} | sl_pct:{sl_pct}")

        return sl_pct

    def is_price_over_ma(self, length: int) -> bool:
        current_price: Decimal = self.get_current_close()
        current_ma: Decimal = self.get_current_ma(length)

        is_over: bool = current_price > current_ma
        self.logger().info(f"is_price_over_ma(): {is_over} | current_price:{current_price} | current_ma:{current_ma}")

        if not is_over:
            self.logger().info("is_price_over_ma() | resetting self.tr_stoch_reversal_counter to 0")
            self.tr_stoch_reversal_counter = 0

        return is_over

    def has_stoch_reversed_for_tr_buy(self, candle_count: int, stoch_length: int) -> bool:
        stoch_series: pd.Series = self.processed_data[f"STOCH_{stoch_length}_k"]
        recent_stochs = stoch_series.iloc[-candle_count:].reset_index(drop=True)

        peak_stoch: Decimal = Decimal(recent_stochs.max())
        saved_peak_stoch, _ = self.saved_tr_peak_stoch

        if max([peak_stoch, saved_peak_stoch]) <= 50:
            return False

        if peak_stoch > saved_peak_stoch:
            timestamp_series: pd.Series = self.processed_data["timestamp"]
            recent_timestamps = timestamp_series.iloc[-candle_count:].reset_index(drop=True)
            peak_stoch_index = recent_stochs.idxmax()

            peak_stoch_timestamp = recent_timestamps.iloc[peak_stoch_index]

            self.logger().info(f"has_stoch_reversed_for_tr_buy() | peak_stoch_index:{peak_stoch_index} | peak_stoch_timestamp:{timestamp_to_iso(peak_stoch_timestamp)}")
            self.save_tr_peak_stoch(peak_stoch, peak_stoch_timestamp)

        saved_peak_stoch, _ = self.saved_tr_peak_stoch

        stoch_threshold: Decimal = saved_peak_stoch - 3
        current_stoch = self.get_current_stoch(stoch_length)

        self.logger().info(f"has_stoch_reversed_for_tr_buy() | saved_peak_stoch:{saved_peak_stoch} | current_stoch:{current_stoch} | stoch_threshold:{stoch_threshold} | current_price:{self.get_current_close()}")

        if current_stoch > stoch_threshold:
            self.tr_stoch_reversal_counter = 0
            self.logger().info("has_stoch_reversed_for_tr_buy() | resetting self.tr_stoch_reversal_counter to 0")
            return False

        self.tr_stoch_reversal_counter += 1
        self.logger().info(f"has_stoch_reversed_for_tr_buy() | incremented self.tr_stoch_reversal_counter to:{self.tr_stoch_reversal_counter}")

        return self.tr_stoch_reversal_counter > 4

    # def are_candles_green(self, candle_count: int) -> bool:
    #     candle_start_index: int = -candle_count - 1
    #
    #     open_series: pd.Series = self.processed_data["open"]
    #     recent_opens = open_series.iloc[candle_start_index:-1].reset_index(drop=True)
    #
    #     close_series: pd.Series = self.processed_data["close"]
    #     recent_closes = close_series.iloc[candle_start_index:-1].reset_index(drop=True)
    #
    #     return all(recent_closes[i] > recent_opens[i] for i in range(len(recent_opens)))
