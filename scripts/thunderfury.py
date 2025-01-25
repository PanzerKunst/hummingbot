from decimal import Decimal
from typing import Dict, List, Tuple

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.tracked_order_details import TrackedOrderDetails
from scripts.thunderfury_config import ExcaliburConfig

# Generate config file: create --script-config thunderfury
# Start the bot: start --script thunderfury.py --conf conf_thunderfury_GOAT.yml
#                start --script thunderfury.py --conf conf_thunderfury_AI16Z.yml
#                start --script thunderfury.py --conf conf_thunderfury_AIXBT.yml
#                start --script thunderfury.py --conf conf_thunderfury_FARTCOIN.yml
#                start --script thunderfury.py --conf conf_thunderfury_MOODENG.yml
#                start --script thunderfury.py --conf conf_thunderfury_PENGU.yml
#                start --script thunderfury.py --conf conf_thunderfury_PNUT.yml
#                start --script thunderfury.py --conf conf_thunderfury_POPCAT.yml
#                start --script thunderfury.py --conf conf_thunderfury_VIRTUAL.yml
#                start --script thunderfury.py --conf conf_thunderfury_WIF.yml
# Quickstart script: -p=a -f thunderfury.py -c conf_thunderfury_GOAT.yml

ORDER_REF_MEAN_REVERSION: str = "MeanReversion"
CANDLE_COUNT_FOR_MR_PRICE_CHANGE: int = 3
CANDLE_DURATION_MINUTES: int = 1


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()
        self.reset_mr_context()

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

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

        # candles_df["SMA_7"] = candles_df.ta.sma(length=7)

        # Calling the lower-level function, because the one in core.py has a bug in the argument names
        # stoch_10_df = stoch(
        #     high=candles_df["high"],
        #     low=candles_df["low"],
        #     close=candles_df["close"],
        #     k=10,
        #     d=1,
        #     smooth_k=1
        # )
        #
        # candles_df["STOCH_10_k"] = stoch_10_df["STOCHk_10_1_1"]

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        mr_context_lifetime_minutes: int = CANDLE_COUNT_FOR_MR_PRICE_CHANGE * CANDLE_DURATION_MINUTES + 1
        self.check_mr_context(mr_context_lifetime_minutes)

        self.create_actions_proposal_mean_reversion()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()
        self.stop_actions_proposal_mean_reversion()

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
                    "RSI_20"
                    # "SMA_7",
                    # "STOCH_10_k"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display], table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Quote amount and Triple Barrier
    #

    def get_position_quote_amount(self, side: TradeType) -> Decimal:
        if side == TradeType.SELL:
            return self.config.amount_quote * Decimal(0.75)  # Less, because closing an unprofitable Short position costs significantly more

        return self.config.amount_quote

    def get_triple_barrier(self, side: TradeType) -> TripleBarrier:
        stop_loss_pct: Decimal = (
            self.compute_sl_pct_for_sell(2) if side == TradeType.SELL
            else self.compute_sl_pct_for_buy(2)
        )

        take_profit_pct: Decimal = (
            stop_loss_pct * Decimal(0.75) if side == TradeType.SELL
            else stop_loss_pct * Decimal(0.5)
        )

        return TripleBarrier(
            stop_loss_delta=stop_loss_pct / 100,
            take_profit_delta=take_profit_pct / 100,
            open_order_type=OrderType.MARKET
        )

    #
    # Mean Reversion start/stop action proposals
    #

    def create_actions_proposal_mean_reversion(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MEAN_REVERSION)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_mean_reversion_order(TradeType.SELL, active_orders):
            triple_barrier = self.get_triple_barrier(TradeType.SELL)
            amount_quote: Decimal = self.get_position_quote_amount(TradeType.SELL)
            self.create_order(TradeType.SELL, self.get_current_close(), triple_barrier, amount_quote, ORDER_REF_MEAN_REVERSION)

        if self.can_create_mean_reversion_order(TradeType.BUY, active_orders):
            triple_barrier = self.get_triple_barrier(TradeType.BUY)
            amount_quote: Decimal = self.get_position_quote_amount(TradeType.BUY)
            self.create_order(TradeType.BUY, self.get_current_close(), triple_barrier, amount_quote, ORDER_REF_MEAN_REVERSION)

    def can_create_mean_reversion_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        amount_quote: Decimal = self.get_position_quote_amount(side)

        if not self.can_create_order(side, amount_quote, ORDER_REF_MEAN_REVERSION, 5):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if (
                self.has_price_spiked_for_mr(CANDLE_COUNT_FOR_MR_PRICE_CHANGE) and
                not self.is_price_spike_a_reversal(CANDLE_COUNT_FOR_MR_PRICE_CHANGE, 5, self.config.min_price_delta_pct_to_open_mr) and
                self.did_price_rebound_for_mr_sell(CANDLE_COUNT_FOR_MR_PRICE_CHANGE) and
                (self.is_peak_on_current_candle(CANDLE_COUNT_FOR_MR_PRICE_CHANGE) or self.is_current_price_below_open()) and
                self.did_volume_spike(2) and
                self.did_rsi_spike(5)
            ):
                self.logger().info(f"can_create_mean_reversion_order() > Opening Mean Reversion Sell at {self.get_current_close()}")
                self.reset_mr_context()
                return True

            return False

        if (
            self.has_price_crashed_for_mr(CANDLE_COUNT_FOR_MR_PRICE_CHANGE) and
            not self.is_price_crash_a_reversal(CANDLE_COUNT_FOR_MR_PRICE_CHANGE, 5, self.config.min_price_delta_pct_to_open_mr) and
            self.did_price_rebound_for_mr_buy(CANDLE_COUNT_FOR_MR_PRICE_CHANGE) and
            (self.is_bottom_on_current_candle(CANDLE_COUNT_FOR_MR_PRICE_CHANGE) or self.is_current_price_above_open()) and
            self.did_volume_spike(2) and
            self.did_rsi_crash(5)
        ):
            self.logger().info(f"can_create_mean_reversion_order() > Opening Mean Reversion Buy at {self.get_current_close()}")
            self.reset_mr_context()
            return True

        return False

    def stop_actions_proposal_mean_reversion(self):
        pass
        # filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MEAN_REVERSION)
        #
        # if len(filled_sell_orders) > 0:
        #     if (
        #         self.has_price_rebounded_enough_to_close_sell(40) and
        #         self.is_price_under_ma(7) and
        #         self.has_stoch_reversed_for_mr_sell(CANDLE_COUNT_FOR_MR_PRICE_CHANGE_AND_STOCH_REVERSAL, 10)
        #     ):
        #         self.logger().info(f"stop_actions_proposal_mean_reversion() > Closing Mean Reversion Sell at {self.get_current_close()}")
        #         self.market_close_orders(filled_sell_orders, CloseType.COMPLETED)
        #         self.reset_mr_context()
        #
        # if len(filled_buy_orders) > 0:
        #     if (
        #         self.has_price_rebounded_enough_to_close_buy(40) and
        #         self.is_price_over_ma(7) and
        #         self.has_stoch_reversed_for_mr_buy(CANDLE_COUNT_FOR_MR_PRICE_CHANGE_AND_STOCH_REVERSAL, 10)
        #     ):
        #         self.logger().info(f"stop_actions_proposal_mean_reversion() > Closing Mean Reversion Buy at {self.get_current_close()}")
        #         self.market_close_orders(filled_buy_orders, CloseType.COMPLETED)
        #         self.reset_mr_context()

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

    def get_current_ma(self, length: int) -> Decimal:
        sma_series: pd.Series = self.processed_data[f"SMA_{length}"]
        return Decimal(sma_series.iloc[-1])

    def get_current_stoch(self, length: int) -> Decimal:
        stoch_series: pd.Series = self.processed_data[f"STOCH_{length}_k"]
        return Decimal(stoch_series.iloc[-1])

    #
    # Mean Reversion context
    #

    def reset_mr_context(self):
        self.save_mr_price_change_pct(Decimal(0.0), self.get_market_data_provider_time())
        # self.save_mr_bottom_stoch(Decimal(52.0), self.get_market_data_provider_time())
        # self.save_mr_peak_stoch(Decimal(48.0), self.get_market_data_provider_time())

        # self.mr_stoch_reversal_counter: int = 0
        self.mr_price_reversal_counter: int = 0
        self.logger().info("Mean Reversion context is reset")

    def save_mr_price_change_pct(self, price_change_pct: Decimal, timestamp: float):
        self.saved_mr_price_change_pct: Tuple[Decimal, float] = price_change_pct, timestamp

    # def save_mr_bottom_stoch(self, bottom_stoch: Decimal, timestamp: float):
    #     self.saved_mr_bottom_stoch: Tuple[Decimal, float] = bottom_stoch, timestamp
    #
    # def save_mr_peak_stoch(self, peak_stoch: Decimal, timestamp: float):
    #     self.saved_mr_peak_stoch: Tuple[Decimal, float] = peak_stoch, timestamp

    def check_mr_context(self, lifetime_minutes: int):
        _, saved_price_change_pct_timestamp = self.saved_mr_price_change_pct
        # _, saved_bottom_stoch_timestamp = self.saved_mr_bottom_stoch
        # _, saved_peak_stoch_timestamp = self.saved_mr_peak_stoch

        most_recent_timestamp: float = max([
            saved_price_change_pct_timestamp,
            # saved_bottom_stoch_timestamp,
            # saved_peak_stoch_timestamp
        ])

        last_acceptable_timestamp = self.get_market_data_provider_time() - lifetime_minutes * 60

        is_outdated: bool = most_recent_timestamp < last_acceptable_timestamp

        if is_outdated and not self.is_mr_context_default():
            self.logger().info("check_mr_context() | Resetting outdated context")
            self.reset_mr_context()

    def is_mr_context_default(self) -> bool:
        saved_price_change_pct, _ = self.saved_mr_price_change_pct
        # saved_bottom_stoch, _ = self.saved_mr_bottom_stoch
        # saved_peak_stoch, _ = self.saved_mr_peak_stoch

        return (
            saved_price_change_pct == Decimal(0.0) and
            # saved_bottom_stoch == Decimal(52.0) and
            # saved_peak_stoch == Decimal(48.0) and
            # self.mr_stoch_reversal_counter == 0 and
            self.mr_price_reversal_counter == 0
        )

    #
    # Mean Reversion functions
    #

    def get_current_peak(self, candle_count: int) -> Decimal:
        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-candle_count:].reset_index(drop=True)

        return Decimal(recent_highs.max())

    def get_current_bottom(self, candle_count: int) -> Decimal:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-candle_count:].reset_index(drop=True)

        return Decimal(recent_lows.min())

    def has_price_spiked_for_mr(self, candle_count: int) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-candle_count:].reset_index(drop=True)
        bottom_price = Decimal(recent_lows.min())
        bottom_price_index = recent_lows.idxmin()

        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-candle_count:].reset_index(drop=True)
        peak_price = Decimal(recent_highs.max())
        peak_price_index = recent_highs.idxmax()

        if bottom_price_index > peak_price_index:
            return False

        price_delta_pct: Decimal = (peak_price - bottom_price) / bottom_price * 100
        is_spiking = self.config.min_price_delta_pct_to_open_mr < price_delta_pct

        if is_spiking:
            self.logger().info(f"has_price_spiked_for_mr() | bottom_price_index:{bottom_price_index} | bottom_price:{bottom_price} | peak_price_index:{peak_price_index} | peak_price:{peak_price}")
            self.logger().info(f"has_price_spiked_for_mr() | current_price:{self.get_current_close()} | price_delta_pct:{price_delta_pct}")
            self.save_mr_price_change_pct(price_delta_pct, self.get_market_data_provider_time())

        return is_spiking

    def has_price_crashed_for_mr(self, candle_count: int) -> bool:
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
        is_crashing = self.config.min_price_delta_pct_to_open_mr < price_delta_pct

        if is_crashing:
            self.logger().info(f"has_price_crashed_for_mr() | bottom_price_index:{bottom_price_index} | bottom_price:{bottom_price} | peak_price_index:{peak_price_index} | peak_price:{peak_price}")
            self.logger().info(f"has_price_crashed_for_mr() | current_price:{self.get_current_close()} | price_delta_pct:{price_delta_pct}")
            self.save_mr_price_change_pct(price_delta_pct, self.get_market_data_provider_time())

        return is_crashing

    def is_price_spike_a_reversal(self, candle_count: int, multiplier_for_previous_high: int, min_delta_to_open: Decimal) -> bool:
        candle_end_index: int = -candle_count
        candle_start_index: int = candle_end_index * multiplier_for_previous_high

        high_series: pd.Series = self.processed_data["high"]
        previous_highs = high_series.iloc[candle_start_index:candle_end_index].reset_index(drop=True)

        previous_peak = Decimal(previous_highs.max())
        current_peak = self.get_current_peak(candle_count)
        delta_pct: Decimal = (current_peak - previous_peak) / previous_peak * 100

        is_reversal: bool = delta_pct < min_delta_to_open * Decimal(0.67)

        self.logger().info(f"is_price_spike_a_reversal(): {is_reversal} | current_peak:{current_peak} | previous_peak:{previous_peak} | delta_pct:{delta_pct}")

        # TODO return is_reversal
        return False

    def is_price_crash_a_reversal(self, candle_count: int, multiplier_for_previous_low: int, min_delta_to_open: Decimal) -> bool:
        candle_end_index: int = -candle_count
        candle_start_index: int = candle_end_index * multiplier_for_previous_low

        low_series: pd.Series = self.processed_data["low"]
        previous_lows = low_series.iloc[candle_start_index:candle_end_index].reset_index(drop=True)

        previous_bottom = Decimal(previous_lows.min())
        current_bottom = self.get_current_bottom(candle_count)
        delta_pct: Decimal = (previous_bottom - current_bottom) / current_bottom * 100

        is_reversal: bool = delta_pct < min_delta_to_open * Decimal(0.67)

        self.logger().info(f"is_price_crash_a_reversal(): {is_reversal} | current_bottom:{current_bottom} | previous_bottom:{previous_bottom} | delta_pct:{delta_pct}")

        # TODO return is_reversal
        return False

    def did_price_rebound_for_mr_sell(self, candle_count: int) -> bool:
        saved_price_change_pct, _ = self.saved_mr_price_change_pct
        price_threshold_pct: Decimal = saved_price_change_pct / 6
        price_bottom_limit_pct: Decimal = saved_price_change_pct / 3

        peak_price = self.get_current_peak(candle_count)
        price_threshold: Decimal = peak_price * (1 - price_threshold_pct / 100)
        price_bottom_limit: Decimal = peak_price * (1 - price_bottom_limit_pct / 100)

        current_price: Decimal = self.get_current_close()

        self.logger().info(f"did_price_rebound_for_mr_sell() | saved_price_change_pct:{saved_price_change_pct} | peak_price:{peak_price} | current_price:{current_price}")
        self.logger().info(f"did_price_rebound_for_mr_sell() | price_threshold_pct:{price_threshold_pct} | price_threshold:{price_threshold}")
        self.logger().info(f"did_price_rebound_for_mr_sell() | price_bottom_limit_pct:{price_bottom_limit_pct} | price_bottom_limit:{price_bottom_limit}")

        if current_price > price_threshold:
            self.mr_price_reversal_counter = 0
            self.logger().info("did_price_rebound_for_mr_sell() | current_price > price_threshold | resetting self.mr_price_reversal_counter to 0")
            return False

        if current_price < price_bottom_limit:
            self.mr_price_reversal_counter = 0
            self.logger().info("did_price_rebound_for_mr_sell() | current_price < price_bottom_limit | resetting self.mr_price_reversal_counter to 0")
            return False

        self.mr_price_reversal_counter += 1
        self.logger().info(f"did_price_rebound_for_mr_sell() | incremented self.mr_price_reversal_counter to:{self.mr_price_reversal_counter}")

        return self.mr_price_reversal_counter > 19

    def did_price_rebound_for_mr_buy(self, candle_count: int) -> bool:
        saved_price_change_pct, _ = self.saved_mr_price_change_pct
        price_threshold_pct: Decimal = saved_price_change_pct / 6
        price_top_limit_pct: Decimal = saved_price_change_pct / 3

        bottom_price = self.get_current_bottom(candle_count)
        price_threshold: Decimal = bottom_price * (1 + price_threshold_pct / 100)
        price_top_limit: Decimal = bottom_price * (1 + price_top_limit_pct / 100)

        current_price: Decimal = self.get_current_close()

        self.logger().info(f"did_price_rebound_for_mr_buy() | saved_price_change_pct:{saved_price_change_pct} | bottom_price:{bottom_price} | current_price:{current_price}")
        self.logger().info(f"did_price_rebound_for_mr_buy() | price_threshold_pct:{price_threshold_pct} | price_threshold:{price_threshold}")
        self.logger().info(f"did_price_rebound_for_mr_buy() | price_top_limit_pct:{price_top_limit_pct} | price_top_limit:{price_top_limit}")

        if current_price < price_threshold:
            self.mr_price_reversal_counter = 0
            self.logger().info("did_price_rebound_for_mr_buy() | current_price < price_threshold | resetting self.mr_price_reversal_counter to 0")
            return False

        if current_price > price_top_limit:
            self.mr_price_reversal_counter = 0
            self.logger().info("did_price_rebound_for_mr_buy() | current_price > price_top_limit | resetting self.mr_price_reversal_counter to 0")
            return False

        self.mr_price_reversal_counter += 1
        self.logger().info(f"did_price_rebound_for_mr_buy() | incremented self.mr_price_reversal_counter to:{self.mr_price_reversal_counter}")

        return self.mr_price_reversal_counter > 19

    def is_peak_on_current_candle(self, candle_count: int) -> bool:
        current_peak = self.get_current_peak(candle_count)
        current_high = self.get_current_high()

        self.logger().info(f"is_peak_on_current_candle() | current_peak:{current_peak} | current_high:{current_high}")

        return current_peak == current_high

    def is_bottom_on_current_candle(self, candle_count: int) -> bool:
        current_bottom = self.get_current_bottom(candle_count)
        current_low = self.get_current_low()

        self.logger().info(f"is_bottom_on_current_candle() | current_bottom:{current_bottom} | current_low:{current_low}")

        return current_bottom == current_low

    def is_current_price_below_open(self) -> bool:
        current_price = self.get_current_close()
        open_price = self.get_current_open()

        self.logger().info(f"is_current_price_below_open() | open_price:{open_price} | current_price:{current_price}")

        return current_price < open_price

    def is_current_price_above_open(self) -> bool:
        current_price = self.get_current_close()
        open_price = self.get_current_open()

        self.logger().info(f"is_current_price_above_open() | open_price:{open_price} | current_price:{current_price}")

        return current_price > open_price

    def did_rsi_spike(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_20"]
        recent_rsis = rsi_series.iloc[-candle_count:].reset_index(drop=True)

        bottom_rsi: Decimal = Decimal(recent_rsis.min())
        bottom_rsi_index = recent_rsis.idxmin()

        peak_rsi: Decimal = Decimal(recent_rsis.max())
        peak_rsi_index = recent_rsis.idxmax()

        if peak_rsi_index < bottom_rsi_index:
            return False

        if peak_rsi < 55:
            return False

        rsi_delta: Decimal = peak_rsi - bottom_rsi

        self.logger().info(f"did_rsi_spike() | bottom_rsi:{bottom_rsi} | peak_rsi:{peak_rsi} | rsi_delta:{rsi_delta}")

        # TODO return rsi_delta > 15
        return True

    def did_rsi_crash(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_20"]
        recent_rsis = rsi_series.iloc[-candle_count:].reset_index(drop=True)

        bottom_rsi: Decimal = Decimal(recent_rsis.min())
        bottom_rsi_index = recent_rsis.idxmin()

        peak_rsi: Decimal = Decimal(recent_rsis.max())
        peak_rsi_index = recent_rsis.idxmax()

        if bottom_rsi_index < peak_rsi_index:
            return False

        if bottom_rsi > 45:
            return False

        rsi_delta: Decimal = peak_rsi - bottom_rsi

        self.logger().info(f"did_rsi_crash() | bottom_rsi:{bottom_rsi} | peak_rsi:{peak_rsi} | rsi_delta:{rsi_delta}")

        # TODO return rsi_delta > 15
        return True

    def did_volume_spike(self, candle_count: int) -> bool:
        vol_series: pd.Series = self.processed_data["volume"]

        recent_vol = vol_series.iloc[-candle_count:].reset_index(drop=True)

        previous_vol_end_index: int = -candle_count
        previous_vol_start_index: int = previous_vol_end_index * 3
        previous_vol = vol_series.iloc[previous_vol_start_index:previous_vol_end_index].reset_index(drop=True)

        total_recent_vol: Decimal = Decimal(recent_vol.sum())
        total_previous_vol: Decimal = Decimal(previous_vol.sum())
        ratio_recent_vs_previous: Decimal = total_recent_vol / total_previous_vol

        self.logger().info(f"did_volume_spike() | ratio_recent_vs_previous:{ratio_recent_vs_previous}")

        if ratio_recent_vs_previous < 3:
            return False

        recent_and_previous_vol = vol_series.iloc[previous_vol_start_index:].reset_index(drop=True)
        peak_vol = Decimal(recent_and_previous_vol.max())
        peak_vol_index = recent_and_previous_vol.idxmax()
        pre_peak_vol = Decimal(recent_and_previous_vol.iloc[peak_vol_index - 1])
        ratio_peak_vs_pre_peak: Decimal = peak_vol / pre_peak_vol

        self.logger().info(f"did_volume_spike() | ratio_peak_vs_pre_peak:{ratio_peak_vs_pre_peak}")

        return ratio_peak_vs_pre_peak > 10

    def compute_sl_pct_for_sell(self, candle_count: int) -> Decimal:
        peak_price = self.get_current_peak(candle_count)
        current_price: Decimal = self.get_current_close()

        delta_pct_with_peak: Decimal = (peak_price - current_price) / current_price * 100
        sl_pct: Decimal = delta_pct_with_peak * Decimal(0.8)

        self.logger().info(f"compute_sl_pct_for_sell() | peak_price:{peak_price} | current_price:{current_price} | sl_pct:{sl_pct}")

        return sl_pct

    def compute_sl_pct_for_buy(self, candle_count: int) -> Decimal:
        bottom_price = self.get_current_bottom(candle_count)
        current_price: Decimal = self.get_current_close()

        delta_pct_with_bottom: Decimal = (current_price - bottom_price) / current_price * 100
        sl_pct: Decimal = delta_pct_with_bottom * Decimal(1.25)

        self.logger().info(f"compute_sl_pct_for_buy() | bottom_price:{bottom_price} | current_price:{current_price} | sl_pct:{sl_pct}")

        return sl_pct

    # def compute_tp_pct_for_sell(self, candle_count: int) -> Decimal:
    #     bottom_price = self.get_current_bottom(candle_count)
    #     current_price: Decimal = self.get_current_close()
    #
    #     delta_pct_with_bottom: Decimal = (current_price - bottom_price) / current_price * 100
    #
    #     self.logger().info(f"compute_tp_pct_for_sell() | bottom_price:{bottom_price} | current_price:{current_price} | delta_pct_with_bottom:{delta_pct_with_bottom}")
    #
    #     return delta_pct_with_bottom / 2
    #
    # def compute_tp_pct_for_buy(self, candle_count: int) -> Decimal:
    #     peak_price = self.get_current_peak(candle_count)
    #     current_price: Decimal = self.get_current_close()
    #
    #     delta_pct_with_peak: Decimal = (peak_price - current_price) / current_price * 100
    #
    #     self.logger().info(f"compute_tp_pct_for_buy() | peak_price:{peak_price} | current_price:{current_price} | delta_pct_with_peak:{delta_pct_with_peak}")
    #
    #     return delta_pct_with_peak / 2
    #
    # def has_price_rebounded_enough_to_close_sell(self, rebound_pct: int) -> bool:
    #     saved_zoomed_out_price_change_pct, _ = self.saved_mr_zoomed_out_price_change_pct
    #     min_rebound_pct: Decimal = saved_zoomed_out_price_change_pct * rebound_pct / 100
    #
    #     peak_price = self.get_current_peak(15)
    #     current_price: Decimal = self.get_current_close()
    #     delta_pct_with_peak: Decimal = (peak_price - current_price) / current_price * 100
    #
    #     self.logger().info(f"has_price_rebounded_enough_to_close_sell() | rebound_pct:{rebound_pct} | min_rebound_pct:{min_rebound_pct}")
    #     self.logger().info(f"has_price_rebounded_enough_to_close_sell() | peak_price:{peak_price} | current_price:{current_price} | delta_pct_with_peak:{delta_pct_with_peak}")
    #
    #     has_rebounded_enough: bool = min_rebound_pct < delta_pct_with_peak
    #
    #     if not has_rebounded_enough:
    #         self.logger().info("has_price_rebounded_enough_to_close_sell() | resetting self.mr_stoch_reversal_counter to 0")
    #         self.mr_stoch_reversal_counter = 0
    #
    #     return has_rebounded_enough
    #
    # def has_price_rebounded_enough_to_close_buy(self, rebound_pct: int) -> bool:
    #     saved_zoomed_out_price_change_pct, _ = self.saved_mr_zoomed_out_price_change_pct
    #     min_rebound_pct: Decimal = saved_zoomed_out_price_change_pct * rebound_pct / 100
    #
    #     bottom_price = self.get_current_bottom(15)
    #     current_price: Decimal = self.get_current_close()
    #     delta_pct_with_bottom: Decimal = (current_price - bottom_price) / current_price * 100
    #
    #     self.logger().info(f"has_price_rebounded_enough_to_close_buy() | rebound_pct:{rebound_pct} | min_rebound_pct:{min_rebound_pct}")
    #     self.logger().info(f"has_price_rebounded_enough_to_close_buy() | bottom_price:{bottom_price} | current_price:{current_price} | delta_pct_with_bottom:{delta_pct_with_bottom}")
    #
    #     has_rebounded_enough: bool = min_rebound_pct < delta_pct_with_bottom
    #
    #     if not has_rebounded_enough:
    #         self.logger().info("has_price_rebounded_enough_to_close_buy() | resetting self.mr_stoch_reversal_counter to 0")
    #         self.mr_stoch_reversal_counter = 0
    #
    #     return has_rebounded_enough
    #
    # def is_price_under_ma(self, length: int) -> bool:
    #     current_price: Decimal = self.get_current_close()
    #     current_ma: Decimal = self.get_current_ma(length)
    #
    #     is_under: bool = current_price < current_ma
    #     self.logger().info(f"is_price_under_ma(): {is_under} | current_price:{current_price} | current_ma:{current_ma}")
    #
    #     if not is_under:
    #         self.logger().info("is_price_under_ma() | resetting self.mr_stoch_reversal_counter to 0")
    #         self.mr_stoch_reversal_counter = 0
    #
    #     return is_under
    #
    # def is_price_over_ma(self, length: int) -> bool:
    #     current_price: Decimal = self.get_current_close()
    #     current_ma: Decimal = self.get_current_ma(length)
    #
    #     is_over: bool = current_price > current_ma
    #     self.logger().info(f"is_price_over_ma(): {is_over} | current_price:{current_price} | current_ma:{current_ma}")
    #
    #     if not is_over:
    #         self.logger().info("is_price_over_ma() | resetting self.mr_stoch_reversal_counter to 0")
    #         self.mr_stoch_reversal_counter = 0
    #
    #     return is_over
    #
    # def has_stoch_reversed_for_mr_sell(self, candle_count: int, stoch_length: int) -> bool:
    #     stoch_series: pd.Series = self.processed_data[f"STOCH_{stoch_length}_k"]
    #     recent_stochs = stoch_series.iloc[-candle_count:].reset_index(drop=True)
    #
    #     bottom_stoch: Decimal = Decimal(recent_stochs.min())
    #     saved_bottom_stoch, _ = self.saved_mr_bottom_stoch
    #
    #     if min([bottom_stoch, saved_bottom_stoch]) >= 52:
    #         return False
    #
    #     if bottom_stoch < saved_bottom_stoch:
    #         timestamp_series: pd.Series = self.processed_data["timestamp"]
    #         recent_timestamps = timestamp_series.iloc[-candle_count:].reset_index(drop=True)
    #         bottom_stoch_index = recent_stochs.idxmin()
    #
    #         bottom_stoch_timestamp = recent_timestamps.iloc[bottom_stoch_index]
    #
    #         self.logger().info(f"has_stoch_reversed_for_mr_sell() | bottom_stoch_index:{bottom_stoch_index} | bottom_stoch_timestamp:{timestamp_to_iso(bottom_stoch_timestamp)}")
    #         self.save_mr_bottom_stoch(bottom_stoch, bottom_stoch_timestamp)
    #
    #     saved_bottom_stoch, _ = self.saved_mr_bottom_stoch
    #
    #     stoch_threshold: Decimal = saved_bottom_stoch + 3
    #     current_stoch = self.get_current_stoch(stoch_length)
    #
    #     self.logger().info(f"has_stoch_reversed_for_mr_sell() | saved_bottom_stoch:{saved_bottom_stoch} | current_stoch:{current_stoch} | stoch_threshold:{stoch_threshold} | current_price:{self.get_current_close()}")
    #
    #     if current_stoch < stoch_threshold:
    #         self.mr_stoch_reversal_counter = 0
    #         self.logger().info("has_stoch_reversed_for_mr_sell() | resetting self.mr_stoch_reversal_counter to 0")
    #         return False
    #
    #     self.mr_stoch_reversal_counter += 1
    #     self.logger().info(f"has_stoch_reversed_for_mr_sell() | incremented self.mr_stoch_reversal_counter to:{self.mr_stoch_reversal_counter}")
    #
    #     return self.mr_stoch_reversal_counter > 2
    #
    # def has_stoch_reversed_for_mr_buy(self, candle_count: int, stoch_length: int) -> bool:
    #     stoch_series: pd.Series = self.processed_data[f"STOCH_{stoch_length}_k"]
    #     recent_stochs = stoch_series.iloc[-candle_count:].reset_index(drop=True)
    #
    #     peak_stoch: Decimal = Decimal(recent_stochs.max())
    #     saved_peak_stoch, _ = self.saved_mr_peak_stoch
    #
    #     if max([peak_stoch, saved_peak_stoch]) <= 48:
    #         return False
    #
    #     if peak_stoch > saved_peak_stoch:
    #         timestamp_series: pd.Series = self.processed_data["timestamp"]
    #         recent_timestamps = timestamp_series.iloc[-candle_count:].reset_index(drop=True)
    #         peak_stoch_index = recent_stochs.idxmax()
    #
    #         peak_stoch_timestamp = recent_timestamps.iloc[peak_stoch_index]
    #
    #         self.logger().info(f"has_stoch_reversed_for_mr_buy() | peak_stoch_index:{peak_stoch_index} | peak_stoch_timestamp:{timestamp_to_iso(peak_stoch_timestamp)}")
    #         self.save_mr_peak_stoch(peak_stoch, peak_stoch_timestamp)
    #
    #     saved_peak_stoch, _ = self.saved_mr_peak_stoch
    #
    #     stoch_threshold: Decimal = saved_peak_stoch - 3
    #     current_stoch = self.get_current_stoch(stoch_length)
    #
    #     self.logger().info(f"has_stoch_reversed_for_mr_buy() | saved_peak_stoch:{saved_peak_stoch} | current_stoch:{current_stoch} | stoch_threshold:{stoch_threshold} | current_price:{self.get_current_close()}")
    #
    #     if current_stoch > stoch_threshold:
    #         self.mr_stoch_reversal_counter = 0
    #         self.logger().info("has_stoch_reversed_for_mr_buy() | resetting self.mr_stoch_reversal_counter to 0")
    #         return False
    #
    #     self.mr_stoch_reversal_counter += 1
    #     self.logger().info(f"has_stoch_reversed_for_mr_buy() | incremented self.mr_stoch_reversal_counter to:{self.mr_stoch_reversal_counter}")
    #
    #     return self.mr_stoch_reversal_counter > 2

    # def is_sell_order_profitable(self, filled_sell_orders: List[TrackedOrderDetails]) -> bool:
    #     pnl_pct: Decimal = compute_sell_orders_pnl_pct(filled_sell_orders, self.get_mid_price())
    #
    #     self.logger().info(f"is_sell_order_profitable() | pnl_pct:{pnl_pct}")
    #
    #     return pnl_pct > 0
    #
    # def is_buy_order_profitable(self, filled_buy_orders: List[TrackedOrderDetails]) -> bool:
    #     pnl_pct: Decimal = compute_buy_orders_pnl_pct(filled_buy_orders, self.get_mid_price())
    #
    #     self.logger().info(f"is_buy_order_profitable() | pnl_pct:{pnl_pct}")
    #
    #     return pnl_pct > 0
