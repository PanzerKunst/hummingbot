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
from scripts.excalibur_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import was_an_order_recently_opened
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Trend following via comparing 2 MAs, and trend reversal based on RSI & Stochastic
# Generate config file: create --script-config excalibur
# Start the bot: start --script excalibur.py --conf conf_excalibur_GOAT.yml
#                start --script excalibur.py --conf conf_excalibur_MOODENG.yml
#                start --script excalibur.py --conf conf_excalibur_PNUT.yml
# Quickstart script: -p=a -f excalibur.py -c conf_excalibur_GOAT.yml

ORDER_REF_MA_CROSS = "MaCross"
ORDER_REF_FAST_REVERSAL = "FastReversal"
ORDER_REF_SLOW_REVERSAL = "SlowReversal"


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
        if order_ref == ORDER_REF_MA_CROSS:
            return TripleBarrier(
                open_order_type=OrderType.MARKET,
                stop_loss=self.config.ma_cross_stop_loss_pct / 100
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

        candles_df["RSI_20"] = candles_df.ta.rsi(length=20)
        candles_df["RSI_40"] = candles_df.ta.rsi(length=40)

        candles_df["SMA_75"] = candles_df.ta.sma(length=75)
        candles_df["SMA_300"] = candles_df.ta.sma(length=300)

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

        stoch_80_df = stoch(
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            k=80,
            d=6,
            smooth_k=16
        )

        candles_df["STOCH_80_k"] = stoch_80_df["STOCHk_80_6_16"]

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        self.create_actions_proposal_ma_cross()
        self.create_actions_proposal_fast_rev()
        self.create_actions_proposal_slow_rev()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        self.stop_actions_proposal_ma_cross()
        self.stop_actions_proposal_fast_rev()
        self.stop_actions_proposal_slow_rev()

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
                    "SMA_75",
                    "SMA_300",
                    "STOCH_40_k",
                    "STOCH_80_k"
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
                self.create_twap_market_orders(TradeType.SELL, entry_price, triple_barrier, self.config.amount_quote_ma_cross, ORDER_REF_MA_CROSS)
            )

        if self.can_create_ma_cross_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MA_CROSS)

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.BUY, entry_price, triple_barrier, self.config.amount_quote_ma_cross, ORDER_REF_MA_CROSS)
            )

    def can_create_ma_cross_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote_ma_cross, 0, ORDER_REF_MA_CROSS):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.did_short_ma_cross_under_long():
                self.logger().info("can_create_ma_cross_order() > Short MA crossed under long")

                return (
                    self.is_price_close_enough_to_short_ma() and
                    not self.is_stoch_too_low_to_open_ma_x_sell() and
                    not self.is_rsi_too_low_to_open_ma_x_sell() and
                    not self.did_price_suddenly_rise_to_short_ma()
                )

            return False

        if self.did_short_ma_cross_over_long():
            self.logger().info("can_create_ma_cross_order() > Short MA crossed over long")

            return (
                self.is_price_close_enough_to_short_ma() and
                not self.is_stoch_too_high_to_open_ma_x_buy() and
                not self.is_rsi_too_high_to_open_ma_x_buy() and
                not self.did_price_suddenly_drop_to_short_ma()
            )

        return False

    def stop_actions_proposal_ma_cross(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MA_CROSS)

        if len(filled_sell_orders) > 0:
            if self.did_short_ma_cross_over_long():
                self.logger().info("stop_actions_proposal_ma_cross(SELL) > Short MA crossed over long")
                self.market_close_orders(filled_sell_orders, CloseType.COMPLETED)

        if len(filled_buy_orders) > 0:
            if self.did_short_ma_cross_under_long():
                self.logger().info("stop_actions_proposal_ma_cross(BUY) > Short MA crossed under long")
                self.market_close_orders(filled_buy_orders, CloseType.COMPLETED)

    #
    # Fast Reversal start/stop action proposals
    #

    def create_actions_proposal_fast_rev(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_FAST_REVERSAL)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_fast_rev_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_FAST_REVERSAL)
            self.create_order(TradeType.SELL, entry_price, triple_barrier, self.config.amount_quote_tr, ORDER_REF_FAST_REVERSAL)

        if self.can_create_fast_rev_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_FAST_REVERSAL)
            self.create_order(TradeType.BUY, entry_price, triple_barrier, self.config.amount_quote_tr, ORDER_REF_FAST_REVERSAL)

    def can_create_fast_rev_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote_tr, 0, ORDER_REF_FAST_REVERSAL):
            return False

        if len(active_tracked_orders) > 1:
            return False

        if was_an_order_recently_opened(active_tracked_orders, 8 * 60, self.get_market_data_provider_time()):
            return False

        if side == TradeType.SELL:
            if self.is_stoch_spike_good_to_open_fast_rev_sell() and self.is_rsi_spike_good_to_open_fast_rev():
                self.logger().info("can_create_fast_rev_order() > Opening Sell reversal")
                return True

            return False

        if self.is_stoch_crash_good_to_open_fast_rev_buy() and self.is_rsi_crash_good_to_open_fast_rev():
            self.logger().info("can_create_fast_rev_order() > Opening Buy reversal")
            return True

        return False

    def stop_actions_proposal_fast_rev(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_FAST_REVERSAL)

        if len(filled_sell_orders) > 0:
            if self.should_close_rev_sell_due_to_stoch_reversal():
                self.logger().info("stop_actions_proposal_fast_rev() > Closing Sell reversal")
                self.market_close_orders(filled_sell_orders, CloseType.TAKE_PROFIT)

        if len(filled_buy_orders) > 0:
            if self.should_close_rev_buy_due_to_stoch_reversal():
                self.logger().info("stop_actions_proposal_fast_rev() > Closing Buy reversal")
                self.market_close_orders(filled_buy_orders, CloseType.TAKE_PROFIT)

    #
    # Slow Reversal start/stop action proposals
    #

    def create_actions_proposal_slow_rev(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_SLOW_REVERSAL)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_slow_rev_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_SLOW_REVERSAL)
            self.create_order(TradeType.SELL, entry_price, triple_barrier, self.config.amount_quote_tr, ORDER_REF_SLOW_REVERSAL)

        if self.can_create_slow_rev_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_SLOW_REVERSAL)
            self.create_order(TradeType.BUY, entry_price, triple_barrier, self.config.amount_quote_tr, ORDER_REF_SLOW_REVERSAL)

    def can_create_slow_rev_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote_tr, 0, ORDER_REF_SLOW_REVERSAL):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.is_stoch_spike_good_to_open_slow_rev_sell(40) and self.is_stoch_spike_good_to_open_slow_rev_sell(80):
                self.logger().info("can_create_slow_rev_order() > Opening Sell reversal")
                return True

            return False

        if self.is_stoch_crash_good_to_open_slow_rev_buy(40) and self.is_stoch_crash_good_to_open_slow_rev_buy(80):
            self.logger().info("can_create_slow_rev_order() > Opening Buy reversal")
            return True

        return False

    def stop_actions_proposal_slow_rev(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_SLOW_REVERSAL)

        if len(filled_sell_orders) > 0:
            if self.should_close_rev_sell_due_to_stoch_reversal():
                self.logger().info("stop_actions_proposal_slow_rev() > Closing Sell reversal")
                self.market_close_orders(filled_sell_orders, CloseType.TAKE_PROFIT)

        if len(filled_buy_orders) > 0:
            if self.should_close_rev_buy_due_to_stoch_reversal():
                self.logger().info("stop_actions_proposal_slow_rev() > Closing Buy reversal")
                self.market_close_orders(filled_buy_orders, CloseType.TAKE_PROFIT)

    #
    # Getters on `self.processed_data[]`
    #

    def get_latest_close(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[-2])

    def get_current_rsi(self, length: int) -> Decimal:
        rsi_series: pd.Series = self.processed_data[f"RSI_{length}"]
        return Decimal(rsi_series.iloc[-1])

    def get_latest_ma(self, length: int) -> Decimal:
        return self._get_ma_at_index(length, -2)

    def get_previous_ma(self, length: int) -> Decimal:
        return self._get_ma_at_index(length, -3)

    def _get_ma_at_index(self, length: int, index: int) -> Decimal:
        sma_series: pd.Series = self.processed_data[f"SMA_{length}"]
        return Decimal(sma_series.iloc[index])

    def get_current_stoch(self, length: int) -> Decimal:
        return self._get_stoch_at_index(length, -1)

    def get_latest_stoch(self, length: int) -> Decimal:
        return self._get_stoch_at_index(length, -2)

    def _get_stoch_at_index(self, length: int, index: int) -> Decimal:
        stoch_series: pd.Series = self.processed_data[f"STOCH_{length}_k"]
        return Decimal(stoch_series.iloc[index])

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

    def is_price_close_enough_to_short_ma(self):
        latest_close = self.get_latest_close()
        delta_pct: Decimal = (latest_close - self.get_latest_ma(75)) / latest_close * 100

        self.logger().info(f"is_price_close_enough_to_short_ma() | latest_close:{latest_close} | latest_short_ma:{self.get_latest_ma(75)} | delta_pct:{delta_pct}")

        return abs(delta_pct) < self.config.max_price_delta_pct_with_short_ma_to_open

    def is_stoch_too_low_to_open_ma_x_sell(self) -> bool:
        current_stoch = self.get_current_stoch(40)

        self.logger().info(f"is_stoch_too_low_to_open_ma_x_sell() | current_stoch:{current_stoch}")

        return current_stoch < 20

    def is_stoch_too_high_to_open_ma_x_buy(self) -> bool:
        current_stoch = self.get_current_stoch(40)

        self.logger().info(f"is_stoch_too_high_to_open_ma_x_buy() | current_stoch:{current_stoch}")

        return current_stoch > 80

    def is_rsi_too_low_to_open_ma_x_sell(self) -> bool:
        current_rsi = self.get_current_rsi(20)

        self.logger().info(f"is_rsi_too_low_to_open_ma_x_sell() | current_rsi:{current_rsi}")

        if current_rsi < 37.5:
            return True

        rsi_series: pd.Series = self.processed_data["RSI_20"]
        recent_rsis = rsi_series.iloc[-10:]

        min_rsi = Decimal(recent_rsis.min())

        self.logger().info(f"is_rsi_too_low_to_open_ma_x_sell() | min_rsi:{min_rsi}")

        return min_rsi < 30

    def is_rsi_too_high_to_open_ma_x_buy(self) -> bool:
        current_rsi = self.get_current_rsi(20)

        self.logger().info(f"is_rsi_too_high_to_open_ma_x_buy() | current_rsi:{current_rsi}")

        if current_rsi > 62.5:
            return True

        rsi_series: pd.Series = self.processed_data["RSI_20"]
        recent_rsis = rsi_series.iloc[-10:]

        max_rsi = Decimal(recent_rsis.max())

        self.logger().info(f"is_rsi_too_high_to_open_ma_x_buy() | max_rsi:{max_rsi}")

        return max_rsi > 70

    def did_price_suddenly_rise_to_short_ma(self) -> bool:
        latest_close = self.get_latest_close()

        close_series: pd.Series = self.processed_data["close"]
        recent_prices = close_series.iloc[-21:-1]  # 20 items, last one excluded
        min_price: Decimal = Decimal(recent_prices.min())

        price_delta_pct: Decimal = (latest_close - min_price) / latest_close * 100

        self.logger().info(f"did_price_suddenly_rise_to_short_ma() | latest_close:{latest_close} | min_price:{min_price} | price_delta_pct:{price_delta_pct}")

        # The percentage difference between min_price and current_price is over x%
        return price_delta_pct > self.config.min_price_delta_pct_for_sudden_reversal_to_short_ma

    def did_price_suddenly_drop_to_short_ma(self) -> bool:
        latest_close = self.get_latest_close()

        close_series: pd.Series = self.processed_data["close"]
        recent_prices = close_series.iloc[-21:-1]  # 20 items, last one excluded
        max_price: Decimal = Decimal(recent_prices.max())

        price_delta_pct: Decimal = (max_price - latest_close) / latest_close * 100

        self.logger().info(f"did_price_suddenly_drop_to_short_ma() | latest_close:{latest_close} | max_price:{max_price} | price_delta_pct:{price_delta_pct}")

        return price_delta_pct > self.config.min_price_delta_pct_for_sudden_reversal_to_short_ma

    #
    # Fast reversal functions
    #

    def is_stoch_spike_good_to_open_fast_rev_sell(self) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_40_k"]
        recent_stochs = stoch_series.iloc[-10:]
        peak_stoch: Decimal = Decimal(recent_stochs.max())

        if peak_stoch < 90:
            return False

        current_stoch = self.get_current_stoch(40)
        max_acceptable_stoch: Decimal = peak_stoch - 2

        # TODO: remove
        if current_stoch < max_acceptable_stoch:
            self.logger().info(f"is_stoch_spike_good_to_open_fast_rev_sell() | peak_stoch:{peak_stoch} | current_stoch:{current_stoch}")

        return current_stoch < max_acceptable_stoch

    def is_stoch_crash_good_to_open_fast_rev_buy(self) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_40_k"]
        recent_stochs = stoch_series.iloc[-10:]
        bottom_stoch: Decimal = Decimal(recent_stochs.min())

        if bottom_stoch > 10:
            return False

        current_stoch = self.get_current_stoch(40)
        min_acceptable_stoch: Decimal = bottom_stoch + 2

        # TODO: remove
        if current_stoch > min_acceptable_stoch:
            self.logger().info(f"is_stoch_crash_good_to_open_fast_rev_buy() | bottom_stoch:{bottom_stoch} | current_stoch:{current_stoch}")

        return current_stoch > min_acceptable_stoch

    def is_rsi_spike_good_to_open_fast_rev(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_40"].reset_index(drop=True)
        recent_rsis = rsi_series.iloc[-12:]

        peak_rsi = Decimal(recent_rsis.max())

        if peak_rsi < self.config.rsi_peak_to_open_fast_rev:
            return False

        current_rsi = self.get_current_rsi(40)
        max_acceptable_rsi: Decimal = peak_rsi - 2

        self.logger().info(f"is_rsi_spike_good_to_open_fast_rev() | peak_rsi:{peak_rsi} | current_rsi:{current_rsi} | max_acceptable_rsi:{max_acceptable_rsi}")

        if current_rsi > max_acceptable_rsi:
            return False

        peak_rsi_index = recent_rsis.idxmax()
        bottom_rsi = Decimal(recent_rsis.iloc[0:peak_rsi_index].min())
        start_delta: Decimal = peak_rsi - bottom_rsi

        self.logger().info(f"is_rsi_spike_good_to_open_fast_rev() | bottom_rsi:{bottom_rsi} | start_delta:{start_delta}")

        return start_delta > 12

    def is_rsi_crash_good_to_open_fast_rev(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_40"].reset_index(drop=True)
        recent_rsis = rsi_series.iloc[-12:]

        bottom_rsi = Decimal(recent_rsis.min())

        if bottom_rsi > self.config.rsi_bottom_to_open_fast_rev:
            return False

        current_rsi = self.get_current_rsi(40)
        min_acceptable_rsi: Decimal = bottom_rsi + 2

        self.logger().info(f"is_rsi_crash_good_to_open_fast_rev() | bottom_rsi:{bottom_rsi} | current_rsi:{current_rsi} | min_acceptable_rsi:{min_acceptable_rsi}")

        if current_rsi < min_acceptable_rsi:
            return False

        bottom_rsi_index = recent_rsis.idxmin()
        peak_rsi = Decimal(recent_rsis.iloc[0:bottom_rsi_index].max())
        start_delta: Decimal = peak_rsi - bottom_rsi

        self.logger().info(f"is_rsi_crash_good_to_open_fast_rev() | peak_rsi:{peak_rsi} | start_delta:{start_delta}")

        return start_delta > 12

    def should_close_rev_sell_due_to_stoch_reversal(self) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_40_k"]
        recent_stochs = stoch_series.iloc[-8:]
        bottom_stoch: Decimal = Decimal(recent_stochs.min())

        if bottom_stoch > 43:
            return False

        current_stoch = self.get_current_stoch(40)
        min_acceptable_stoch: Decimal = bottom_stoch + 1

        self.logger().info(f"should_close_rev_sell_due_to_stoch_reversal() | bottom_stoch:{bottom_stoch} | current_stoch:{current_stoch}")

        return current_stoch > min_acceptable_stoch

    def should_close_rev_buy_due_to_stoch_reversal(self) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_40_k"]
        recent_stochs = stoch_series.iloc[-8:]
        peak_stoch: Decimal = Decimal(recent_stochs.max())

        if peak_stoch < 57:
            return False

        current_stoch = self.get_current_stoch(40)
        max_acceptable_stoch: Decimal = peak_stoch - 1

        self.logger().info(f"should_close_rev_buy_due_to_stoch_reversal() | peak_stoch:{peak_stoch} | current_stoch:{current_stoch}")

        return current_stoch < max_acceptable_stoch

    #
    # Slow reversal functions
    #

    def is_stoch_spike_good_to_open_slow_rev_sell(self, stoch_length: int) -> bool:
        stoch_series: pd.Series = self.processed_data[f"STOCH_{stoch_length}_k"]
        recent_stochs = stoch_series.iloc[-8:]
        peak_stoch: Decimal = Decimal(recent_stochs.max())

        if peak_stoch < 88:
            return False

        current_stoch = self.get_current_stoch(stoch_length)
        max_acceptable_stoch: Decimal = peak_stoch - 2

        # TODO: remove
        if current_stoch < max_acceptable_stoch:
            self.logger().info(f"is_stoch_spike_good_to_open_slow_rev_sell({stoch_length}) | peak_stoch:{peak_stoch} | current_stoch:{current_stoch}")

        return current_stoch < max_acceptable_stoch

    def is_stoch_crash_good_to_open_slow_rev_buy(self, stoch_length: int) -> bool:
        stoch_series: pd.Series = self.processed_data[f"STOCH_{stoch_length}_k"]
        recent_stochs = stoch_series.iloc[-8:]
        bottom_stoch: Decimal = Decimal(recent_stochs.min())

        if bottom_stoch > 12:
            return False

        current_stoch = self.get_current_stoch(stoch_length)
        min_acceptable_stoch: Decimal = bottom_stoch + 2

        # TODO: remove
        if current_stoch > min_acceptable_stoch:
            self.logger().info(f"is_stoch_crash_good_to_open_slow_rev_buy({stoch_length}) | bottom_stoch:{bottom_stoch} | current_stoch:{current_stoch}")

        return current_stoch > min_acceptable_stoch
