from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.pk.arthur_config import ArthurConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_utils import average
from scripts.pk.tracked_order_details import TrackedOrderDetails


# Trend start and reversals, dependant on sudden price movements and RSI
# Generate config file: create --script-config arthur
# Start the bot: start --script arthur.py --conf conf_arthur_BOME.yml
#                start --script arthur.py --conf conf_arthur_BONK.yml
#                start --script arthur.py --conf conf_arthur_DOGS.yml
#                start --script arthur.py --conf conf_arthur_FLOKI.yml
#                start --script arthur.py --conf conf_arthur_MOODENG.yml
#                start --script arthur.py --conf conf_arthur_NEIRO.yml
#                start --script arthur.py --conf conf_arthur_NEIROETH.yml
#                start --script arthur.py --conf conf_arthur_PEOPLE.yml
#                start --script arthur.py --conf conf_arthur_POPCAT.yml
#                start --script arthur.py --conf conf_arthur_TURBO.yml
#                start --script arthur.py --conf conf_arthur_WIF.yml
# Quickstart script: -p=a -f arthur.py -c conf_arthur_POPCAT.yml


class ArthurStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ArthurConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ArthurConfig):
        super().__init__(connectors, config)

        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_pair,
                interval=config.candles_interval,
                max_records=config.candles_length
            ))

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

    def get_triple_barrier_config(self, sl_tp_pct: Decimal, expiration_min: int, open_order_type: OrderType) -> TripleBarrierConfig:
        # TODO: remove
        self.logger().info(f"get_triple_barrier_config() | sl_tp_pct:{sl_tp_pct}")

        return TripleBarrierConfig(
            stop_loss=Decimal(sl_tp_pct / 100),
            take_profit=Decimal(sl_tp_pct / 100),
            open_order_type=open_order_type,
            take_profit_order_type=OrderType.MARKET,  # TODO: LIMIT
            stop_loss_order_type=OrderType.MARKET,
            time_limit=expiration_min * 60
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

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()

        if self.can_create_trend_start_order(TradeType.SELL, active_sell_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp_for_trend_start(TradeType.SELL)
            triple_barrier_config = self.get_triple_barrier_config(sl_tp_pct, self.config.filled_trend_start_order_expiration_min, OrderType.MARKET)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_create_trend_start_order(TradeType.BUY, active_buy_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp_for_trend_start(TradeType.BUY)
            triple_barrier_config = self.get_triple_barrier_config(sl_tp_pct, self.config.filled_trend_start_order_expiration_min, OrderType.MARKET)
            self.create_order(TradeType.BUY, entry_price, triple_barrier_config)

        if self.can_create_trend_reversal_order(TradeType.SELL, active_sell_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp_for_trend_reversal(TradeType.SELL)
            triple_barrier_config = self.get_triple_barrier_config(sl_tp_pct, self.config.filled_trend_reversal_order_expiration_min, OrderType.LIMIT)
            self.create_order(TradeType.SELL, entry_price, triple_barrier_config)

        if self.can_create_trend_reversal_order(TradeType.BUY, active_buy_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            sl_tp_pct: Decimal = self.compute_sl_and_tp_for_trend_reversal(TradeType.BUY)
            triple_barrier_config = self.get_triple_barrier_config(sl_tp_pct, self.config.filled_trend_reversal_order_expiration_min, OrderType.LIMIT)
            self.create_order(TradeType.BUY, entry_price, triple_barrier_config)

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        return []  # Always return []

    def format_status(self) -> str:
        original_status = super().format_status()
        custom_status = ["\n"]

        if self.ready_to_trade:
            if not self.processed_data.empty:
                columns_to_display = [
                    "timestamp_iso",
                    "close",
                    "high",
                    "low",
                    "volume",
                    "RSI"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(self.config.rsi_length), table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_trend_start_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if not self.is_recent_volume_enough():
            return False

        self.logger().info("recent_volume_is_enough")

        delta_pct = self.compute_delta_pct(side)

        if delta_pct < self.config.trend_start_price_change_threshold_pct:
            return False

        self.logger().info(f"delta_pct above threshold: {delta_pct}")

        if side == TradeType.SELL:
            is_rsi_in_range = self.is_rsi_in_range_for_trend_start_sell_order()
            self.logger().info(f"is_rsi_in_range: {is_rsi_in_range}")
            return is_rsi_in_range

        is_rsi_in_range = self.is_rsi_in_range_for_trend_start_buy_order()
        self.logger().info(f"is_rsi_in_range: {is_rsi_in_range}")
        return is_rsi_in_range

    def can_create_trend_reversal_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            # During the last 12min, there was a completed trend_start trade with TP on the opposite side
            last_terminated_filled_order = self.find_last_terminated_filled_order(TradeType.BUY)

            if not last_terminated_filled_order or last_terminated_filled_order.close_type != CloseType.TAKE_PROFIT:
                return False

            if last_terminated_filled_order.terminated_at + 12 * 60 < self.market_data_provider.time():
                return False

            self.logger().info(f"can_create_trend_reversal_order({side}) > There was a TP order within the last 12min")

            # During the last 7min, RSI exceeded TH
            if not self.did_rsi_recently_jump():
                return False

            self.logger().info(f"can_create_trend_reversal_order({side}) > rsi_did_recently_jump")

            # Current RSI is now back to over 30 / below 70
            if not self.has_rsi_recovered_from_jump():
                return False

            self.logger().info(f"can_create_trend_reversal_order({side}) > rsi_has_recovered_from_jump")

            return True

        last_terminated_filled_order = self.find_last_terminated_filled_order(TradeType.SELL)

        if not last_terminated_filled_order or last_terminated_filled_order.close_type != CloseType.TAKE_PROFIT:
            return False

        if last_terminated_filled_order.terminated_at + 12 * 60 < self.market_data_provider.time():
            return False

        self.logger().info(f"can_create_trend_reversal_order({side}) > There was a TP order within the last 12min")

        if not self.did_rsi_recently_crash():
            return False

        self.logger().info(f"can_create_trend_reversal_order({side}) > rsi_did_recently_crash")

        if not self.has_rsi_recovered_from_crash():
            return False

        self.logger().info(f"can_create_trend_reversal_order({side}) > rsi_has_recovered_from_crash")

        return True

    #
    # Custom functions specific to this controller
    #

    def is_rsi_in_range_for_trend_start_sell_order(self) -> bool:
        if self.get_recent_rsi_avg() > 70:
            self.logger().info("is_rsi_in_range_for_trend_start_sell_order: too risky as the price has been trending up for a while")
            return False

        rsi_series: pd.Series = self.processed_data["RSI"]
        rsi_2candles_before = Decimal(rsi_series.iloc[-3])

        return rsi_2candles_before > self.config.trend_start_sell_latest_complete_candle_min_rsi

    def is_rsi_in_range_for_trend_start_buy_order(self) -> bool:
        if self.get_recent_rsi_avg() < 30:
            self.logger().info("is_rsi_in_range_for_trend_start_buy_order: too risky as the price has been trending down for a while")
            return False

        rsi_series: pd.Series = self.processed_data["RSI"]
        rsi_2candles_before = Decimal(rsi_series.iloc[-3])

        return rsi_2candles_before < self.config.trend_start_buy_latest_complete_candle_max_rsi

    def get_recent_rsi_avg(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["RSI"]
        rsi_3candles_before = Decimal(rsi_series.iloc[-4])
        rsi_4candles_before = Decimal(rsi_series.iloc[-5])
        rsi_5candles_before = Decimal(rsi_series.iloc[-6])
        rsi_6candles_before = Decimal(rsi_series.iloc[-7])
        rsi_7candles_before = Decimal(rsi_series.iloc[-8])

        return average(rsi_3candles_before, rsi_4candles_before, rsi_5candles_before, rsi_6candles_before, rsi_7candles_before)

    def did_rsi_recently_jump(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        recent_rsis = rsi_series.iloc[-8:-1]  # 7 items, last one excluded

        # TODO: remove
        self.logger().info(f"did_rsi_recently_jump() | max(recent_rsis):{max(recent_rsis)}")

        return max(recent_rsis) > self.config.trend_reversal_sell_min_rsi

    def did_rsi_recently_crash(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        recent_rsis = rsi_series.iloc[-8:-1]  # 7 items, last one excluded

        # TODO: remove
        self.logger().info(f"did_rsi_recently_crash() | min(recent_rsis):{min(recent_rsis)}")

        return min(recent_rsis) < self.config.trend_reversal_buy_max_rsi

    def has_rsi_recovered_from_jump(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        current_rsi = rsi_series.iloc[-1]

        # TODO: remove
        self.logger().info(f"has_rsi_recovered_from_jump() | current_rsi:{current_rsi}")

        return current_rsi < 70

    def has_rsi_recovered_from_crash(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI"]
        current_rsi = rsi_series.iloc[-1]

        # TODO: remove
        self.logger().info(f"has_rsi_recovered_from_crash() | current_rsi:{current_rsi}")

        return current_rsi > 30

    def is_recent_volume_enough(self) -> bool:
        volume_series: pd.Series = self.processed_data["volume"]
        current_volume = volume_series.iloc[-1]
        volume_latest_complete_candle = volume_series.iloc[-2]

        if current_volume < volume_latest_complete_candle:
            return False

        recent_volumes = [current_volume, volume_latest_complete_candle]
        older_volumes = volume_series.iloc[-9:-2]  # 7 items, last one excluded

        return sum(recent_volumes) > sum(older_volumes) * 3

    def compute_delta_pct(self, side: TradeType) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        current_close_price = Decimal(close_series.iloc[-1])

        high_series: pd.Series = self.processed_data["high"]
        high_latest_complete_candle = Decimal(high_series.iloc[-2])
        high_1candle_before = Decimal(high_series.iloc[-3])
        high_2candles_before = Decimal(high_series.iloc[-4])
        high_3candles_before = Decimal(high_series.iloc[-5])

        highest_price = max(high_latest_complete_candle, high_1candle_before, high_2candles_before, high_3candles_before)
        delta_pct_sell = (highest_price - current_close_price) / current_close_price * 100

        if side == TradeType.SELL:
            return delta_pct_sell

        low_series: pd.Series = self.processed_data["low"]
        low_latest_complete_candle = Decimal(low_series.iloc[-2])
        low_1candle_before = Decimal(low_series.iloc[-3])
        low_2candles_before = Decimal(low_series.iloc[-4])
        low_3candles_before = Decimal(low_series.iloc[-5])

        lowest_price = min(low_latest_complete_candle, low_1candle_before, low_2candles_before, low_3candles_before)
        delta_pct_buy = (current_close_price - lowest_price) / current_close_price * 100

        return delta_pct_buy

    def compute_sl_and_tp_for_trend_start(self, side: TradeType) -> Decimal:
        delta_pct = self.compute_delta_pct(side)
        return delta_pct * Decimal(0.7)

    def compute_sl_and_tp_for_trend_reversal(self, side: TradeType) -> Decimal:
        opposite_side = TradeType.BUY if side == TradeType.SELL else TradeType.SELL
        last_terminated_filled_order = self.find_last_terminated_filled_order(opposite_side)
        trend_start_sl_tp_pct: Decimal = last_terminated_filled_order.triple_barrier_config.take_profit * 100

        return trend_start_sl_tp_pct * Decimal(0.8)
