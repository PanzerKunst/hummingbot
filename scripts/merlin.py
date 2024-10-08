from decimal import Decimal
from typing import Dict, List, Tuple, Optional

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.pk.merlin_config import MerlinConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_utils import average
from scripts.pk.tracked_order_details import TrackedOrderDetails


# Generate config file: create --script-config merlin
# Start the bot: start --script merlin.py --conf conf_merlin_XDC.yml
# Quickstart script: -p=a -f merlin.py -c conf_merlin_XDC.yml


class MerlinStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: MerlinConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: MerlinConfig):
        super().__init__(connectors, config)

        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.trading_pair,
                interval=config.candles_interval,
                max_records=config.candles_length
            ))

        self.processed_data = pd.DataFrame()

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp

    def get_triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=Decimal(self.config.stop_loss_pct / 100),
            take_profit=Decimal(self.config.take_profit_pct / 100),
            time_limit=self.config.filled_order_expiration_min * 60,
            open_order_type=OrderType.LIMIT,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,
            time_limit_order_type=OrderType.LIMIT
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

        bbands_for_volatility = candles_df.ta.bbands(length=self.config.bbands_length_for_volatility, std=self.config.bbands_std_dev_for_volatility)
        candles_df["bbb_for_volatility"] = bbands_for_volatility[f"BBB_{self.config.bbands_length_for_volatility}_{self.config.bbands_std_dev_for_volatility}"]

        candles_df["rsi"] = candles_df.ta.rsi(length=self.config.rsi_length)
        candles_df["normalized_rsi"] = candles_df["rsi"].apply(self.normalize_rsi)

        bbands_for_trend = candles_df.ta.bbands(length=self.config.bbands_length_for_trend, std=self.config.bbands_std_dev_for_trend)
        candles_df["bbp"] = bbands_for_trend[f"BBP_{self.config.bbands_length_for_trend}_{self.config.bbands_std_dev_for_trend}"]
        candles_df["normalized_bbp"] = candles_df["bbp"].apply(self.normalize_bbp)

        self.processed_data = candles_df

    # TODO: Currently bugged
    # def update_indicators(self, df: pd.DataFrame):
    #     rows_to_add = []
    #
    #     for _, row in df.iterrows():
    #         timestamp = row["timestamp"]
    #         bbb_for_volatility = row["bbb_for_volatility"]
    #         normalized_rsi = row["normalized_rsi"]
    #
    #         if pd.notna(bbb_for_volatility) and pd.notna(normalized_rsi):
    #             if self._get_indicators_for_timestamp(timestamp) is None:  # Do not refactor to `if not [...]`
    #                 rows_to_add.append(row)
    #
    #     if len(rows_to_add) > 0:
    #         new_rows = pd.DataFrame(rows_to_add)
    #         self.processed_data = pd.concat([self.processed_data, new_rows], ignore_index=True)
    #         self.processed_data["index"] = self.processed_data["timestamp"]
    #         self.processed_data.set_index("index", inplace=True)
    #
    # def _get_indicators_for_timestamp(self, timestamp: float) -> Optional[pd.Series]:
    #     matching_row = self.processed_data.query(f"timestamp == {timestamp}")
    #
    #     if matching_row.empty:
    #         return None
    #     else:
    #         return matching_row.iloc[0]

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        if self.is_high_volatility():
            return []

        mid_price = self.get_mid_price()
        unfilled_sell_orders, unfilled_buy_orders = self.get_unfilled_tracked_orders_by_side()

        if self.can_create_mm_order(TradeType.SELL, unfilled_sell_orders):
            entry_price, is_order_on_same_side = self.adjust_sell_price(mid_price)
            triple_barrier_config = self.get_triple_barrier_config()
            amount_multiplier: Decimal = Decimal(1.5) if is_order_on_same_side else Decimal(1)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config, amount_multiplier)

        if self.can_create_mm_order(TradeType.BUY, unfilled_buy_orders):
            entry_price, is_order_on_same_side = self.adjust_buy_price(mid_price)
            triple_barrier_config = self.get_triple_barrier_config()
            amount_multiplier: Decimal = Decimal(1.5) if is_order_on_same_side else Decimal(1)
            self.create_order(TradeType.BUY, entry_price, triple_barrier_config, amount_multiplier)

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        unfilled_sell_orders, unfilled_buy_orders = self.get_unfilled_tracked_orders_by_side()

        if self.is_high_volatility():
            self.logger().info(f"##### is_high_volatility -> Stopping unfilled executors #####")

            for unfilled_order in unfilled_sell_orders + unfilled_buy_orders:
                self.cancel_tracked_order(unfilled_order)

            return []

        if self.should_stop_unfilled_orders_after_sl(TradeType.SELL):
            for unfilled_order in unfilled_sell_orders:
                self.logger().info("should_stop_unfilled_shorts and unfilled_sell_orders not empty")
                self.cancel_tracked_order(unfilled_order)

        if self.should_stop_unfilled_orders_after_sl(TradeType.BUY):
            for unfilled_order in unfilled_buy_orders:
                self.logger().info("should_stop_unfilled_longs and unfilled_buy_orders not empty")
                self.cancel_tracked_order(unfilled_order)

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
                    "bbb_for_volatility",
                    "rsi",
                    "normalized_rsi",
                    "bbp",
                    "normalized_bbp"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(self.config.candles_length), table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_mm_order(self, side: TradeType, unfilled_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(unfilled_tracked_orders) > 0:
            return False

        return not self.is_high_volatility()

    #
    # Custom functions specific to this controller
    #

    @staticmethod
    def normalize_rsi(rsi: float) -> Decimal:
        return Decimal(rsi * 2 - 100)

    @staticmethod
    def denormalize_rsi(normalized_rsi: Decimal) -> Decimal:
        return (normalized_rsi + 100) / 2

    @staticmethod
    def normalize_bbp(bbp: float) -> Decimal:
        return Decimal(bbp - 0.5)

    def get_latest_bbb(self) -> Decimal:
        bbb_series: pd.Series = self.processed_data["bbb_for_volatility"]
        bbb_current_incomplete_minute = Decimal(bbb_series.iloc[-1])
        bbb_previous_full_minute = Decimal(bbb_series.iloc[-2])
        return max(bbb_current_incomplete_minute, bbb_previous_full_minute)

    def get_avg_last_tree_bbb(self) -> Decimal:
        bbb_series: pd.Series = self.processed_data["bbb_for_volatility"]
        bbb_last_full_minute = Decimal(bbb_series.iloc[-2])
        bbb_before_that = Decimal(bbb_series.iloc[-3])
        bbb_even_before_that = Decimal(bbb_series.iloc[-4])
        return average(bbb_last_full_minute, bbb_before_that, bbb_even_before_that)

    def get_latest_normalized_rsi(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["normalized_rsi"]
        return Decimal(rsi_series.iloc[-1])

    def is_high_volatility(self) -> bool:
        # TODO: remove
        self.logger().info(f"is_high_volatility() | latest_bbb: {self.get_latest_bbb()}")

        return self.get_latest_bbb() > self.config.high_volatility_threshold

    def is_still_trending_up(self) -> bool:
        return self.get_latest_normalized_bbp() > -0.2

    def is_still_trending_down(self) -> bool:
        return self.get_latest_normalized_bbp() < 0.2

    def has_been_trending_down_for_a_while(self) -> bool:
        bbp_series: pd.Series = self.processed_data["normalized_bbp"]

        for i in range(2, 10):  # 8 times, the ending point is exclusive.
            normalized_bbp = Decimal(bbp_series.iloc[-i])

            if normalized_bbp > 0:
                return False

        return True

    def has_been_trending_up_for_a_while(self) -> bool:
        bbp_series: pd.Series = self.processed_data["normalized_bbp"]

        for i in range(2, 10):
            normalized_bbp = Decimal(bbp_series.iloc[-i])

            if normalized_bbp < 0:
                return False

        return True

    def get_recent_volume(self) -> int:
        volume_series: pd.Series = self.processed_data["volume"]
        volume_current_incomplete_minute = volume_series.iloc[-1]
        volume_previous_full_minute = volume_series.iloc[-2]
        volume_before_that = volume_series.iloc[-3]
        return max(volume_current_incomplete_minute, volume_previous_full_minute, volume_before_that)

    def get_latest_normalized_bbp(self) -> Decimal:
        bbp_series: pd.Series = self.processed_data["normalized_bbp"]
        bbp_previous_full_minute = Decimal(bbp_series.iloc[-2])
        bbp_current_incomplete_minute = Decimal(bbp_series.iloc[-1])

        return (
            max(bbp_previous_full_minute, bbp_current_incomplete_minute) if bbp_previous_full_minute > 0
            else min(bbp_previous_full_minute, bbp_current_incomplete_minute)
        )

    def is_rsi_at_the_edges(self, normalized_rsi: Decimal) -> bool:
        rsi = self.denormalize_rsi(normalized_rsi)
        self.logger().info(f"is_rsi_at_the_edges? rsi: {rsi}")
        return rsi > 50+18 or rsi < 50-18

    def is_last_terminated_order_sl(self, side: TradeType) -> bool:
        last_terminated_filled_order = self.find_last_terminated_filled_order(side)
        return last_terminated_filled_order and last_terminated_filled_order.close_type == CloseType.STOP_LOSS

    def get_sl_price(self, side: TradeType) -> Optional[Decimal]:
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side()
        filled_orders = filled_sell_orders if side == TradeType.SELL else filled_buy_orders

        if len(filled_orders) == 0:
            return None

        last_filled_order = filled_orders[-1]
        entry_price = last_filled_order.entry_price
        stop_loss = last_filled_order.triple_barrier_config.stop_loss

        sl_price_sell: Decimal = entry_price * (1 + stop_loss)
        sl_price_buy: Decimal = entry_price * (1 - stop_loss)

        return sl_price_sell if side == TradeType.SELL else sl_price_buy

    def should_stop_unfilled_orders_after_sl(self, side: TradeType) -> bool:
        last_terminated_filled_order = self.find_last_terminated_filled_order(side)

        return (
            self.is_last_terminated_order_sl(side) and
            last_terminated_filled_order.terminated_at + self.config.cooldown_time_min * 60 > self.market_data_provider.time()
        )

    def adjust_sell_price(self, mid_price: Decimal) -> Tuple[Decimal, bool]:
        default_adjustment: Decimal = self.config.default_spread_pct / 100

        filled_order_sl_price = self.get_sl_price(TradeType.SELL)
        is_order_on_same_side: bool = filled_order_sl_price is not None

        if filled_order_sl_price:
            self.logger().info(f"There is a filled Short order. Updating mid_price to: {filled_order_sl_price} and default_adjustment to 0.1%")
            mid_price = filled_order_sl_price
            default_adjustment = Decimal(0.1) / 100

        volatility_adjustment_pct: Decimal = Decimal(self.get_recent_volume() / self.config.volume_for_1_pct_volatility_adjustment)

        # TODO: remove
        self.logger().info(f"volatility_adjustment_pct initialized to: {volatility_adjustment_pct}")

        avg_last_three_bbb = self.get_avg_last_tree_bbb()
        if avg_last_three_bbb > 0:
            volatility_adjustment_pct += avg_last_three_bbb * Decimal(0.5)
            # TODO: try removing `* Decimal(0.5)` and instead have a smaller default spread.

        trend_adjustment_pct = Decimal(0)

        if self.is_last_terminated_order_sl(TradeType.SELL) and self.is_still_trending_up():
            self.logger().info("self.is_last_sell_executor_sl() and self.is_still_trending_up(), increasing trend_adjustment_pct")
            trend_adjustment_pct += self.config.default_spread_pct * Decimal(1.5)

        if self.has_been_trending_up_for_a_while() or self.has_been_trending_down_for_a_while():
            self.logger().info("self.has_been_trending_for_a_while(), increasing trend_adjustment_pct")
            trend_adjustment_pct += self.config.default_spread_pct * Decimal(1.5)

        rsi_adjustment_pct = Decimal(0)

        latest_normalized_rsi = self.get_latest_normalized_rsi()

        if self.is_rsi_at_the_edges(latest_normalized_rsi):
            rsi_adjustment_pct = -latest_normalized_rsi * Decimal(0.01)

        total_adjustment = default_adjustment + volatility_adjustment_pct / 100 + trend_adjustment_pct / 100 + rsi_adjustment_pct / 100

        entry_price = mid_price * Decimal(1 + total_adjustment)

        self.logger().info(f"Adjusting SELL price. mid:{mid_price}, avg_last_three_bbb:{avg_last_three_bbb}")
        self.logger().info(f"Adjusting SELL price. def_adj:{default_adjustment}, volatility_adjustment_pct:{volatility_adjustment_pct}, trend_adjustment_pct:{trend_adjustment_pct}, rsi_adjustment_pct:{rsi_adjustment_pct}")
        self.logger().info(f"Adjusting SELL price. total_adj:{total_adjustment}, entry_price:{entry_price}")

        return entry_price, is_order_on_same_side

    def adjust_buy_price(self, mid_price: Decimal) -> Tuple[Decimal, bool]:
        default_adjustment: Decimal = self.config.default_spread_pct / 100

        filled_order_sl_price = self.get_sl_price(TradeType.BUY)
        is_order_on_same_side: bool = filled_order_sl_price is not None

        if filled_order_sl_price:
            self.logger().info(f"There is a filled Long order. Updating mid_price to: {filled_order_sl_price} and default_adjustment to 0.1%")
            mid_price = filled_order_sl_price
            default_adjustment = Decimal(0.1) / 100

        volatility_adjustment_pct: Decimal = Decimal(self.get_recent_volume() / self.config.volume_for_1_pct_volatility_adjustment)

        # TODO: remove
        self.logger().info(f"volatility_adjustment_pct initialized to: {volatility_adjustment_pct}")

        avg_last_three_bbb = self.get_avg_last_tree_bbb()
        if avg_last_three_bbb > 0:
            volatility_adjustment_pct += avg_last_three_bbb * Decimal(0.5)

        trend_adjustment_pct = Decimal(0)

        if self.is_last_terminated_order_sl(TradeType.BUY) and self.is_still_trending_down():
            self.logger().info("self.is_last_buy_executor_sl() and self.is_still_trending_down(), increasing trend_adjustment_pct")
            trend_adjustment_pct += self.config.default_spread_pct * Decimal(1.5)

        if self.has_been_trending_up_for_a_while() or self.has_been_trending_down_for_a_while():
            self.logger().info("self.has_been_trending_for_a_while(), increasing trend_adjustment_pct")
            trend_adjustment_pct += self.config.default_spread_pct * Decimal(1.5)

        rsi_adjustment_pct = Decimal(0)

        latest_normalized_rsi = self.get_latest_normalized_rsi()

        if self.is_rsi_at_the_edges(latest_normalized_rsi):
            rsi_adjustment_pct = latest_normalized_rsi * Decimal(0.01)

        total_adjustment = default_adjustment + volatility_adjustment_pct / 100 + trend_adjustment_pct / 100 + rsi_adjustment_pct / 100

        entry_price = mid_price * Decimal(1 - total_adjustment)

        self.logger().info(f"Adjusting BUY price. mid:{mid_price}, avg_last_three_bbb:{avg_last_three_bbb}")
        self.logger().info(f"Adjusting BUY price. def_adj:{default_adjustment}, volatility_adjustment_pct:{volatility_adjustment_pct}, trend_adjustment_pct:{trend_adjustment_pct}, rsi_adjustment_pct:{rsi_adjustment_pct}")
        self.logger().info(f"Adjusting BUY price. total_adj:{total_adjustment}, entry_price:{entry_price}")

        return entry_price, is_order_on_same_side
