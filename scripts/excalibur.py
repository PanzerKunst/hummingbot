import asyncio
from decimal import Decimal
from typing import Dict, List

import pandas as pd

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

# Trends via comparing 2 SMAs
# Generate config file: create --script-config excalibur
# Start the bot: start --script excalibur.py --conf conf_excalibur_GOAT.yml
#                start --script excalibur.py --conf conf_excalibur_MOODENG.yml
#                start --script excalibur.py --conf conf_excalibur_POPCAT.yml
# Quickstart script: -p=a -f excalibur.py -c conf_excalibur_POPCAT.yml

ORDER_REF_SMA_CROSS = "SmaCross"
ORDER_REF_MEAN_REVERSION = "MeanReversion"


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()
        self.reset_context_sma_cross()

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
                open_order_type=OrderType.MARKET
            )

        return TripleBarrier(
            open_order_type=OrderType.MARKET,
            stop_loss=self.config.mean_reversion_stop_loss_pct / 100,
            take_profit=self.config.mean_reversion_take_profit_pct / 100
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

        candles_df["SMA_short"] = candles_df.ta.sma(length=self.config.sma_short)
        candles_df["SMA_long"] = candles_df.ta.sma(length=self.config.sma_long)

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        self.create_actions_proposal_sma_cross()
        self.create_actions_proposal_mean_reversion()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        self.stop_actions_proposal_sma_cross()
        self.stop_actions_proposal_mean_reversion()

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
                    "SMA_short",
                    "SMA_long"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display], table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_sma_cross_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, ORDER_REF_SMA_CROSS):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.did_short_sma_cross_under_long():
                self.logger().info("can_create_sma_cross_order() > Short SMA crossed under long")
                return not self.is_price_too_far_from_sma() and not self.did_price_suddenly_rise_to_short_sma()

            return False

        if self.did_short_sma_cross_over_long():
            self.logger().info("can_create_sma_cross_order() > Short SMA crossed over long")
            return not self.is_price_too_far_from_sma() and not self.did_price_suddenly_drop_to_short_sma()

        return False

    #
    # Custom functions specific to this controller
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

    def create_actions_proposal_mean_reversion(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MEAN_REVERSION)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_mean_reversion_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MEAN_REVERSION)
            asyncio.get_running_loop().create_task(self.create_twap_market_orders(TradeType.SELL, entry_price, triple_barrier, ORDER_REF_MEAN_REVERSION))

        if self.can_create_mean_reversion_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier(ORDER_REF_MEAN_REVERSION)
            asyncio.get_running_loop().create_task(self.create_twap_market_orders(TradeType.BUY, entry_price, triple_barrier, ORDER_REF_MEAN_REVERSION))

    def can_create_mean_reversion_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, ORDER_REF_MEAN_REVERSION):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if self.did_rsi_spike_and_recover() and self.was_rsi_spike_sudden():
                self.logger().info("can_create_mean_reversion_order() > Sudden RSI spike just ended")
                return True

            return False

        if self.did_rsi_crash_and_recover() and self.was_rsi_crash_sudden():
            self.logger().info("can_create_mean_reversion_order() > Sudden RSI crash just ended")
            return True

        return False

    def stop_actions_proposal_sma_cross(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_SMA_CROSS)

        if len(filled_sell_orders) > 0:
            if self.did_short_sma_cross_over_long():
                self.logger().info("stop_actions_proposal_sma_cross(SELL) > Short SMA crossed over long")
                self.close_sma_cross_orders(filled_sell_orders, CloseType.COMPLETED)

            else:
                if self.should_close_sma_cross_orders_when_price_hits_sma:
                    if self.is_current_price_over_short_sma():
                        self.logger().info("stop_actions_proposal_sma_cross(SELL) > current_price_is_over_short_sma")
                        self.close_sma_cross_orders(filled_sell_orders, CloseType.TAKE_PROFIT)

                elif self.did_price_suddenly_rise_to_short_sma():
                    self.close_sma_cross_orders(filled_sell_orders, CloseType.COMPLETED)

                if self.did_rsi_crash_and_recover():
                    self.logger().info("stop_actions_proposal_sma_cross(SELL) > rsi_did_crash_and_recover")

                    if self.was_rsi_crash_sudden():
                        self.logger().info("stop_actions_proposal_sma_cross(SELL) > rsi_crash_was_sudden")
                        self.close_sma_cross_orders(filled_sell_orders, CloseType.TAKE_PROFIT)
                    elif not self.should_close_sma_cross_orders_when_price_hits_sma:
                        self.logger().info("stop_actions_proposal_sma_cross(SELL) > setting self.should_close_when_price_hits_sma to TRUE")
                        self.should_close_sma_cross_orders_when_price_hits_sma = True

        if len(filled_buy_orders) > 0:
            if self.did_short_sma_cross_under_long():
                self.logger().info("stop_actions_proposal_sma_cross(BUY) > Short SMA crossed under long")
                self.close_sma_cross_orders(filled_buy_orders, CloseType.COMPLETED)

            else:
                if self.should_close_sma_cross_orders_when_price_hits_sma:
                    if self.is_current_price_under_short_sma():
                        self.logger().info("stop_actions_proposal_sma_cross(BUY) > current_price_is_under_short_sma")
                        self.close_sma_cross_orders(filled_buy_orders, CloseType.TAKE_PROFIT)

                elif self.did_price_suddenly_drop_to_short_sma():
                    self.close_sma_cross_orders(filled_sell_orders, CloseType.COMPLETED)

                if self.did_rsi_spike_and_recover():
                    self.logger().info("stop_actions_proposal_sma_cross(BUY) > rsi_did_spike_and_recover")

                    if self.was_rsi_spike_sudden():
                        self.logger().info("stop_actions_proposal_sma_cross(BUY) > rsi_spike_was_sudden")
                        self.close_sma_cross_orders(filled_buy_orders, CloseType.TAKE_PROFIT)
                    elif not self.should_close_sma_cross_orders_when_price_hits_sma:
                        self.logger().info("stop_actions_proposal_sma_cross(BUY) > setting self.should_close_when_price_hits_sma to TRUE")
                        self.should_close_sma_cross_orders_when_price_hits_sma = True

    def stop_actions_proposal_mean_reversion(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MEAN_REVERSION)

        if len(filled_sell_orders) > 0:
            if self.is_current_price_under_short_sma():
                self.logger().info("stop_actions_proposal_mean_reversion(SELL) > current_price_is_under_short_sma")
                self.market_close_orders(filled_sell_orders, CloseType.COMPLETED)

        if len(filled_buy_orders) > 0:
            if self.is_current_price_over_short_sma():
                self.logger().info("stop_actions_proposal_mean_reversion(BUY) > current_price_is_over_short_sma")
                self.market_close_orders(filled_buy_orders, CloseType.COMPLETED)

    def get_latest_close(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[-2])

    def get_latest_rsi(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["RSI"]
        return Decimal(rsi_series.iloc[-2])

    def get_latest_sma(self, short_or_long: str) -> Decimal:
        return self._get_sma_at_index(short_or_long, -2)

    def get_previous_sma(self, short_or_long: str) -> Decimal:
        return self._get_sma_at_index(short_or_long, -3)

    def _get_sma_at_index(self, short_or_long: str, index: int) -> Decimal:
        sma_series: pd.Series = self.processed_data[f"SMA_{short_or_long}"]
        return Decimal(sma_series.iloc[index])

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

    def did_rsi_crash_and_recover(self) -> bool:
        rsi_last_complete_candle = self.get_latest_rsi()
        rsi_series: pd.Series = self.processed_data["RSI"]
        older_rsis = rsi_series.iloc[-14:-2]  # 12 items, last one excluded

        self.logger().info(f"did_rsi_crash_and_recover() | rsi_last_complete_candle:{rsi_last_complete_candle} | older_rsis.min():{older_rsis.min()}")

        return older_rsis.min() < self.config.rsi_threshold_take_profit_sell and rsi_last_complete_candle > 30

    def did_rsi_spike_and_recover(self) -> bool:
        rsi_last_complete_candle = self.get_latest_rsi()
        rsi_series: pd.Series = self.processed_data["RSI"]
        older_rsis = rsi_series.iloc[-14:-2]

        self.logger().info(f"did_rsi_spike_and_recover() | rsi_last_complete_candle:{rsi_last_complete_candle} | older_rsis.max():{older_rsis.max()}")

        return older_rsis.max() > self.config.rsi_threshold_take_profit_buy and rsi_last_complete_candle < 70

    def is_price_too_far_from_sma(self) -> bool:
        latest_sma = self.get_latest_sma("long")
        latest_close = self.get_latest_close()
        delta_pct: Decimal = abs(latest_close - latest_sma) / latest_close * 100

        self.logger().info(f"is_price_too_far_from_sma(): {delta_pct > self.config.max_price_delta_pct_with_sma_to_open_position} | latest_close:{latest_close} | latest_sma:{latest_sma} | delta_pct:{delta_pct}")

        return delta_pct > self.config.max_price_delta_pct_with_sma_to_open_position

    def was_rsi_crash_sudden(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        recent_rsis = rsi_series.iloc[-26:-1]  # 25 items, last one excluded

        min_rsi = recent_rsis.min()
        min_rsi_index = recent_rsis.idxmin()

        # Determine the range to search for the max RSI value, considering only 7 preceding items
        start_index = max(min_rsi_index - 7, recent_rsis.index[0])

        max_rsi = recent_rsis.loc[start_index:min_rsi_index].max()

        self.logger().info(f"was_rsi_crash_sudden() | min_rsi:{min_rsi} | max_rsi:{max_rsi} | result:{max_rsi - min_rsi > self.config.min_rsi_delta_for_sudden_change}")

        return max_rsi - min_rsi > self.config.min_rsi_delta_for_sudden_change

    def was_rsi_spike_sudden(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        recent_rsis = rsi_series.iloc[-26:-1]  # 25 items, last one excluded

        max_rsi = recent_rsis.max()
        max_rsi_index = recent_rsis.idxmax()

        # Determine the range to search for the min RSI value, considering only 7 preceding items
        start_index = max(max_rsi_index - 7, recent_rsis.index[0])

        min_rsi = recent_rsis.loc[start_index:max_rsi_index].min()

        self.logger().info(f"was_rsi_spike_sudden() | min_rsi:{min_rsi} | max_rsi:{max_rsi} | result:{max_rsi - min_rsi > self.config.min_rsi_delta_for_sudden_change}")

        return max_rsi - min_rsi > self.config.min_rsi_delta_for_sudden_change

    def did_price_suddenly_rise_to_short_sma(self) -> bool:
        if self.is_current_price_under_short_sma():
            return False

        close_series: pd.Series = self.processed_data["close"]
        recent_prices = close_series.iloc[-12:-2]  # 10 items, last one excluded
        min_price: Decimal = Decimal(recent_prices.min())

        price_delta_pct: Decimal = (self.get_latest_close() - min_price) / 100

        self.logger().info(f"did_price_suddenly_rise_to_short_sma() | self.get_latest_close():{self.get_latest_close()} | min_price:{min_price} | price_delta_pct:{price_delta_pct}")

        # The percentage difference between min_price and current_price is over x%
        return price_delta_pct > self.config.min_price_delta_pct_for_sudden_reversal_to_short_sma

    def did_price_suddenly_drop_to_short_sma(self) -> bool:
        if self.is_current_price_over_short_sma():
            return False

        close_series: pd.Series = self.processed_data["close"]
        recent_prices = close_series.iloc[-12:-2]  # 10 items, last one excluded
        max_price: Decimal = Decimal(recent_prices.max())

        price_delta_pct: Decimal = (max_price - self.get_latest_close()) / 100

        self.logger().info(f"did_price_suddenly_drop_to_short_sma() | self.get_latest_close():{self.get_latest_close()} | max_price:{max_price} | price_delta_pct:{price_delta_pct}")

        return price_delta_pct > self.config.min_price_delta_pct_for_sudden_reversal_to_short_sma

    def close_sma_cross_orders(self, filled_orders: List[TrackedOrderDetails], close_type: CloseType):
        self.market_close_orders(filled_orders, close_type)
        self.reset_context_sma_cross()

    def reset_context_sma_cross(self):
        self.should_close_sma_cross_orders_when_price_hits_sma = False
