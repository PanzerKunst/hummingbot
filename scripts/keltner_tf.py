from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
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

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def get_triple_barrier(self, side: TradeType) -> TripleBarrier:
        stop_loss_pct: Decimal = (
            self.compute_sl_pct_for_sell(2) if side == TradeType.SELL
            else self.compute_sl_pct_for_buy(2)
        )

        take_profit_pct: Decimal = stop_loss_pct * 2 / 3

        return TripleBarrier(
            stop_loss_delta=stop_loss_pct / 100,
            take_profit_delta=take_profit_pct / 100,
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

        candles_df["RSI_20"] = candles_df.ta.rsi(length=20)

        sma_20_df = candles_df.ta.sma(length=20)

        atr_20_df = candles_df.ta.atr(length=20)

        kc_mult: Decimal = Decimal(2.0)
        candles_df["KC_UPPER"] = sma_20_df + atr_20_df * kc_mult
        candles_df["KC_LOWER"] = sma_20_df - atr_20_df * kc_mult

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
    # Mean Reversion start/stop action proposals
    #

    def create_actions_proposal_mean_reversion(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_KELTNER_TF)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_mean_reversion_order(TradeType.SELL, active_orders):
            triple_barrier = self.get_triple_barrier(TradeType.SELL)
            self.create_order(TradeType.SELL, self.get_current_close(), triple_barrier, self.config.amount_quote, ORDER_REF_KELTNER_TF)

        if self.can_create_mean_reversion_order(TradeType.BUY, active_orders):
            triple_barrier = self.get_triple_barrier(TradeType.BUY)
            self.create_order(TradeType.BUY, self.get_current_close(), triple_barrier, self.config.amount_quote, ORDER_REF_KELTNER_TF)

    def can_create_mean_reversion_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_KELTNER_TF, 5):
            return False

        if len(active_tracked_orders) > 0:
            return False

        # TODO: remove
        CANDLE_COUNT_FOR_MR_PRICE_CHANGE_AND_STOCH_REVERSAL = 3

        if side == TradeType.SELL:
            if (
                self.has_price_spiked_for_mr(CANDLE_COUNT_FOR_MR_PRICE_CHANGE_AND_STOCH_REVERSAL) and
                not self.is_price_spike_a_reversal(CANDLE_COUNT_FOR_MR_PRICE_CHANGE_AND_STOCH_REVERSAL, 5, self.config.min_price_delta_pct_to_open_mr) and
                (self.is_peak_on_current_candle(CANDLE_COUNT_FOR_MR_PRICE_CHANGE_AND_STOCH_REVERSAL) or self.is_current_price_below_open()) and
                self.did_volume_spike(2) and
                self.did_rsi_spike(5)
            ):
                self.logger().info(f"can_create_mean_reversion_order() > Opening Mean Reversion Sell at {self.get_current_close()}")
                return True

            return False

        if (
            self.has_price_crashed_for_mr(CANDLE_COUNT_FOR_MR_PRICE_CHANGE_AND_STOCH_REVERSAL) and
            not self.is_price_crash_a_reversal(CANDLE_COUNT_FOR_MR_PRICE_CHANGE_AND_STOCH_REVERSAL, 5, self.config.min_price_delta_pct_to_open_mr) and
            (self.is_bottom_on_current_candle(CANDLE_COUNT_FOR_MR_PRICE_CHANGE_AND_STOCH_REVERSAL) or self.is_current_price_above_open()) and
            self.did_volume_spike(2) and
            self.did_rsi_crash(5)
        ):
            self.logger().info(f"can_create_mean_reversion_order() > Opening Mean Reversion Buy at {self.get_current_close()}")
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

    #
    # Keltner TF functions
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

        self.logger().info(f"did_volume_spike() | total_recent_vol:{total_recent_vol} | total_previous_vol:{total_previous_vol} | ratio:{ratio_recent_vs_previous}")

        if ratio_recent_vs_previous < 3:
            return False

        recent_and_previous_vol = vol_series.iloc[previous_vol_start_index:].reset_index(drop=True)
        peak_vol = Decimal(recent_and_previous_vol.max())
        peak_vol_index = recent_and_previous_vol.idxmax()
        pre_peak_vol = Decimal(recent_and_previous_vol.iloc[peak_vol_index - 1])

        self.logger().info(f"did_volume_spike() | peak_vol:{peak_vol} | pre_peak_vol:{pre_peak_vol}")

        return peak_vol > pre_peak_vol * 10

    def compute_sl_pct_for_sell(self, candle_count: int) -> Decimal:
        peak_price = self.get_current_peak(candle_count)
        current_price: Decimal = self.get_current_close()

        delta_pct_with_peak: Decimal = (peak_price - current_price) / current_price * 100
        sl_pct: Decimal = delta_pct_with_peak

        self.logger().info(f"compute_sl_pct_for_sell() | peak_price:{peak_price} | current_price:{current_price} | sl_pct:{sl_pct}")

        return sl_pct

    def compute_sl_pct_for_buy(self, candle_count: int) -> Decimal:
        bottom_price = self.get_current_bottom(candle_count)
        current_price: Decimal = self.get_current_close()

        delta_pct_with_bottom: Decimal = (current_price - bottom_price) / current_price * 100
        sl_pct: Decimal = delta_pct_with_bottom

        self.logger().info(f"compute_sl_pct_for_buy() | bottom_price:{bottom_price} | current_price:{current_price} | sl_pct:{sl_pct}")

        return sl_pct
