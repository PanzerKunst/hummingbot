import asyncio
from decimal import Decimal
from typing import Dict, List

import pandas as pd
from pandas_ta import sma

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.excalibur_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import compute_buy_orders_pnl_pct, compute_sell_orders_pnl_pct, was_an_order_recently_opened
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Trend following via comparing 2 MAs, and reversions based on RSI & Stochastic
# Generate config file: create --script-config excalibur
# Start the bot: start --script excalibur.py --conf conf_excalibur_GOAT.yml
#                start --script excalibur.py --conf conf_excalibur_CHILLGUY.yml
#                start --script excalibur.py --conf conf_excalibur_FLOKI.yml
#                start --script excalibur.py --conf conf_excalibur_MOODENG.yml
#                start --script excalibur.py --conf conf_excalibur_NEIRO.yml
#                start --script excalibur.py --conf conf_excalibur_PNUT.yml
#                start --script excalibur.py --conf conf_excalibur_POPCAT.yml
# Quickstart script: -p=a -f excalibur.py -c conf_excalibur_GOAT.yml

ORDER_REF_MA_CROSS = "MaCross"
ORDER_REF_MA_CHANNEL = "MaChannel"


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

    def get_triple_barrier(self, ref: str) -> TripleBarrier:
        if ref == ORDER_REF_MA_CHANNEL:
            return TripleBarrier(
                open_order_type=OrderType.MARKET,
                take_profit=self.config.ma_cross_take_profit_pct / 100
            )

        return TripleBarrier(
            open_order_type=OrderType.MARKET,
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

        candles_df["SMA_19"] = candles_df.ta.sma(length=19)
        candles_df["SMA_75"] = candles_df.ta.sma(length=75)
        candles_df["SMA_300"] = candles_df.ta.sma(length=300)

        candles_df["SMA_10_h"] = sma(close=candles_df["high"], length=10)
        candles_df["SMA_10_l"] = sma(close=candles_df["low"], length=10)

        # Calling the lower-level function, because the one in core.py has a bug in the argument names
        # stoch_40_df = stoch(
        #     high=candles_df["high"],
        #     low=candles_df["low"],
        #     close=candles_df["close"],
        #     k=40,
        #     d=6,
        #     smooth_k=8
        # )
        #
        # candles_df["STOCH_40_k"] = stoch_40_df["STOCHk_40_6_8"]

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        # TODO self.create_actions_proposal_ma_cross()
        self.create_actions_proposal_ma_channel()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        # TODO self.stop_actions_proposal_ma_cross()
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
                    "RSI_40",
                    "SMA_19",
                    "SMA_75",
                    "SMA_300",
                    "SMA_10_h",
                    "SMA_10_l"  # ,
                    # "STOCH_40_k"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display], table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # MA Cross start/stop action proposals
    #

    def create_actions_proposal_ma_cross(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MA_CROSS)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_ma_cross_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MA_CROSS)

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.SELL, entry_price, triple_barrier, self.config.amount_quote, ORDER_REF_MA_CROSS)
            )

        if self.can_create_ma_cross_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MA_CROSS)

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.BUY, entry_price, triple_barrier, self.config.amount_quote, ORDER_REF_MA_CROSS)
            )

    def can_create_ma_cross_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        # Same cooldown as candle duration
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_MA_CROSS, 3):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.did_short_ma_cross_under_long():
                self.logger().info("can_create_ma_cross_order() > Short MA crossed under long")

                return (
                    not self.is_current_price_over_short_ma() and
                    self.is_price_close_enough_to_short_ma() and
                    not self.did_rsi_recently_crash() and
                    not self.did_price_suddenly_drop_to_short_ma()
                )

            return False

        if self.did_short_ma_cross_over_long():
            self.logger().info("can_create_ma_cross_order() > Short MA crossed over long")

            return (
                self.is_current_price_over_short_ma() and
                self.is_price_close_enough_to_short_ma() and
                not self.did_rsi_recently_spike() and
                not self.did_price_suddenly_rise_to_short_ma()
            )

        return False

    def stop_actions_proposal_ma_cross(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MA_CROSS)

        if len(filled_sell_orders) > 0:
            if self.has_order_been_open_long_enough(filled_sell_orders):
                if not self.is_sell_order_profitable(filled_sell_orders) and self.is_current_price_over_short_ma():
                    self.logger().info("stop_actions_proposal_ma_cross() > Stop Loss on Sell MA-X")
                    self.market_close_orders(filled_sell_orders, CloseType.STOP_LOSS)
                elif self.did_tiny_ma_bottom():
                    self.logger().info("stop_actions_proposal_ma_cross() > Closing Sell MA-X: tiny MA bottomed")
                    self.market_close_orders(filled_sell_orders, CloseType.TAKE_PROFIT)

        if len(filled_buy_orders) > 0:
            if self.has_order_been_open_long_enough(filled_buy_orders):
                if not self.is_buy_order_profitable(filled_buy_orders) and not self.is_current_price_over_short_ma():
                    self.logger().info("stop_actions_proposal_ma_cross() > Stop Loss on Buy MA-X")
                    self.market_close_orders(filled_buy_orders, CloseType.STOP_LOSS)
                elif self.did_tiny_ma_peak():
                    self.logger().info("stop_actions_proposal_ma_cross() > Closing Buy MA-X: tiny MA peaked")
                    self.market_close_orders(filled_buy_orders, CloseType.TAKE_PROFIT)

    #
    # MA Channel start/stop action proposals
    #

    def create_actions_proposal_ma_channel(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MA_CHANNEL)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_ma_channel_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MA_CHANNEL)

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.SELL, entry_price, triple_barrier, self.config.amount_quote, ORDER_REF_MA_CHANNEL)
            )

        if self.can_create_ma_channel_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MA_CHANNEL)

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.BUY, entry_price, triple_barrier, self.config.amount_quote, ORDER_REF_MA_CHANNEL)
            )

    def can_create_ma_channel_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_MA_CHANNEL, 0):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.are_candles_fully_below_mal():
                self.logger().info("can_create_ma_channel_order() > 5 candles fully below MAL")
                return True

            return False

        if self.are_candles_fully_above_mah():
            self.logger().info("can_create_ma_channel_order() > 5 candles fully above MAH")
            return True

        return False

    def stop_actions_proposal_ma_channel(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MA_CHANNEL)

        if len(filled_sell_orders) > 0:
            if self.is_current_price_over_mah():
                self.logger().info("stop_actions_proposal_ma_channel() > Closing Sell MA-C")
                self.market_close_orders(filled_sell_orders, CloseType.COMPLETED)

        if len(filled_buy_orders) > 0:
            if self.is_current_price_under_mal():
                self.logger().info("stop_actions_proposal_ma_channel() > Closing Buy MA-C")
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

    def get_current_ma(self, length: int) -> Decimal:
        return self._get_ma_at_index(length, -1)

    def get_latest_ma(self, length: int) -> Decimal:
        return self._get_ma_at_index(length, -2)

    def get_previous_ma(self, length: int) -> Decimal:
        return self._get_ma_at_index(length, -3)

    def _get_ma_at_index(self, length: int, index: int) -> Decimal:
        sma_series: pd.Series = self.processed_data[f"SMA_{length}"]
        return Decimal(sma_series.iloc[index])

    def get_current_mah(self) -> Decimal:
        smah_series: pd.Series = self.processed_data["SMA_10_h"]
        return Decimal(smah_series.iloc[-1])

    def get_current_mal(self) -> Decimal:
        smal_series: pd.Series = self.processed_data["SMA_10_l"]
        return Decimal(smal_series.iloc[-1])

    # def get_current_stoch(self, length: int) -> Decimal:
    #     return self._get_stoch_at_index(length, -1)
    #
    # def get_latest_stoch(self, length: int) -> Decimal:
    #     return self._get_stoch_at_index(length, -2)
    #
    # def _get_stoch_at_index(self, length: int, index: int) -> Decimal:
    #     stoch_series: pd.Series = self.processed_data[f"STOCH_{length}_k"]
    #     return Decimal(stoch_series.iloc[index])

    #
    # MA Cross functions
    #

    def did_short_ma_cross_under_long(self) -> bool:
        return not self.is_latest_short_ma_over_long() and self.is_previous_short_ma_over_long()

    def did_short_ma_cross_over_long(self) -> bool:
        return self.is_latest_short_ma_over_long() and not self.is_previous_short_ma_over_long()

    def is_latest_short_ma_over_long(self) -> bool:
        latest_short_minus_long: Decimal = self.get_latest_ma(75) - self.get_latest_ma(300)
        return latest_short_minus_long > 0

    def is_previous_short_ma_over_long(self) -> bool:
        previous_short_minus_long: Decimal = self.get_previous_ma(75) - self.get_previous_ma(300)
        return previous_short_minus_long > 0

    def is_current_price_over_short_ma(self) -> bool:
        current_price_minus_short_ma: Decimal = self.get_current_close() - self.get_current_ma(75)
        return current_price_minus_short_ma > 0

    def is_price_close_enough_to_short_ma(self):
        latest_close = self.get_latest_close()
        delta_pct: Decimal = (latest_close - self.get_latest_ma(75)) / latest_close * 100

        self.logger().info(f"is_price_close_enough_to_short_ma() | latest_close:{latest_close} | latest_short_ma:{self.get_latest_ma(75)} | delta_pct:{delta_pct}")

        return abs(delta_pct) < self.config.max_price_delta_pct_with_short_ma_to_open

    def did_rsi_recently_crash(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_40"]
        recent_rsis = rsi_series.iloc[-10:].reset_index(drop=True)

        bottom_rsi = Decimal(recent_rsis.min())
        bottom_rsi_index = recent_rsis.idxmin()

        if bottom_rsi_index == 0:
            return False

        peak_rsi = Decimal(recent_rsis.iloc[0:bottom_rsi_index].max())
        start_delta: Decimal = peak_rsi - bottom_rsi

        self.logger().info(f"did_rsi_recently_crash() | bottom_rsi_index:{bottom_rsi_index} | peak_rsi:{peak_rsi} | start_delta:{start_delta}")

        return start_delta > 14

    def did_rsi_recently_spike(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_40"]
        recent_rsis = rsi_series.iloc[-10:].reset_index(drop=True)

        peak_rsi = Decimal(recent_rsis.max())
        peak_rsi_index = recent_rsis.idxmax()

        if peak_rsi_index == 0:
            return False

        bottom_rsi = Decimal(recent_rsis.iloc[0:peak_rsi_index].min())
        start_delta: Decimal = peak_rsi - bottom_rsi

        self.logger().info(f"did_rsi_recently_spike() | peak_rsi_index:{peak_rsi_index} | bottom_rsi:{bottom_rsi} | start_delta:{start_delta}")

        return start_delta > 14

    def did_price_suddenly_rise_to_short_ma(self) -> bool:
        current_close = self.get_current_close()

        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-10:]
        min_price: Decimal = Decimal(recent_lows.min())

        price_delta_pct: Decimal = (current_close - min_price) / current_close * 100

        self.logger().info(f"did_price_suddenly_rise_to_short_ma() | current_close:{current_close} | min_price:{min_price} | price_delta_pct:{price_delta_pct}")

        return price_delta_pct > self.config.min_price_delta_pct_for_sudden_reversal_to_short_ma

    def did_price_suddenly_drop_to_short_ma(self) -> bool:
        current_close = self.get_current_close()

        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-10:]
        max_price: Decimal = Decimal(recent_highs.max())

        price_delta_pct: Decimal = (max_price - current_close) / current_close * 100

        self.logger().info(f"did_price_suddenly_drop_to_short_ma() | current_close:{current_close} | max_price:{max_price} | price_delta_pct:{price_delta_pct}")

        return price_delta_pct > self.config.min_price_delta_pct_for_sudden_reversal_to_short_ma

    def has_order_been_open_long_enough(self, filled_orders: List[TrackedOrderDetails]) -> bool:
        return not was_an_order_recently_opened(filled_orders, 20 * 60, self.get_market_data_provider_time())

    def is_sell_order_profitable(self, filled_sell_orders: List[TrackedOrderDetails]) -> bool:
        pnl_pct: Decimal = compute_sell_orders_pnl_pct(filled_sell_orders, self.get_mid_price())

        return pnl_pct > 0

    def is_buy_order_profitable(self, filled_buy_orders: List[TrackedOrderDetails]) -> bool:
        pnl_pct: Decimal = compute_buy_orders_pnl_pct(filled_buy_orders, self.get_mid_price())

        return pnl_pct > 0

    def did_tiny_ma_bottom(self) -> bool:
        ma_series: pd.Series = self.processed_data["SMA_19"]
        recent_mas = ma_series.iloc[-8:].reset_index(drop=True)
        bottom_ma: Decimal = Decimal(recent_mas.min())

        current_ma = self.get_current_ma(19)
        ma_threshold: Decimal = bottom_ma * (1 + self.config.tiny_ma_reversal_bps / 10000)

        if current_ma > ma_threshold:
            self.logger().info(f"did_tiny_ma_bottom() | current_ma:{current_ma} | ma_threshold:{ma_threshold}")

        return current_ma > ma_threshold

    def did_tiny_ma_peak(self) -> bool:
        ma_series: pd.Series = self.processed_data["SMA_19"]
        recent_mas = ma_series.iloc[-8:].reset_index(drop=True)
        peak_ma: Decimal = Decimal(recent_mas.max())

        current_ma = self.get_current_ma(19)
        ma_threshold: Decimal = peak_ma * (1 - self.config.tiny_ma_reversal_bps / 10000)

        if current_ma < ma_threshold:
            self.logger().info(f"did_tiny_ma_peak() | current_ma:{current_ma} | ma_threshold:{ma_threshold}")

        return current_ma < ma_threshold

    #
    # MA Channel functions
    #

    def are_candles_fully_below_mal(self) -> bool:
        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-5:]

        mal_series: pd.Series = self.processed_data["SMA_10_l"]
        recent_mals = mal_series.iloc[-5:]

        return all(recent_highs[i] < recent_mals[i] for i in range(len(recent_highs)))

    def are_candles_fully_above_mah(self) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-5:]

        mah_series: pd.Series = self.processed_data["SMA_10_h"]
        recent_mahs = mah_series.iloc[-5:]

        return all(recent_lows[i] > recent_mahs[i] for i in range(len(recent_lows)))

    def is_current_price_over_mah(self) -> bool:
        current_price_minus_current_mah: Decimal = self.get_current_close() - self.get_current_mah()
        return current_price_minus_current_mah > 0

    def is_current_price_under_mal(self) -> bool:
        current_mal_minus_current_price: Decimal = self.get_current_mal() - self.get_current_close()
        return current_mal_minus_current_price > 0
