from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pandas_ta as ta  # noqa: F401
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from scripts.utility.my_utils import has_order_expired


class OverhillConfig(ControllerConfigBase):
    controller_name: str = "overhill"
    connector_name: str = "kucoin_perpetual"  # Do not rename attribute - used by BacktestingEngineBase
    trading_pair: str = "AAVE-USDT"  # Do not rename attribute - used by BacktestingEngineBase

    leverage: int = 20
    position_mode: PositionMode = PositionMode.HEDGE
    total_amount_quote: int = Field(90, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = 1

    # Triple Barrier
    stop_loss_pct: float = Field(1.0, client_data=ClientFieldData(is_updatable=True))
    take_profit_pct: float = Field(5.0, client_data=ClientFieldData(is_updatable=True))
    filled_order_expiration_min: int = 1000

    # Technical analysis
    bbands_length: int = 12
    bbands_std_dev: float = 2.0

    # Candles
    candles_connector: str = "binance"
    candles_interval: str = "1m"
    candles_length: int = bbands_length * 2
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor

    # Trading algo
    trend_begin_length: int = Field(16, client_data=ClientFieldData(is_updatable=True))
    trend_end_length: int = Field(12, client_data=ClientFieldData(is_updatable=True))
    trend_bbp_threshold: float = Field(0.15, client_data=ClientFieldData(is_updatable=True))
    delta_with_best_bid_or_ask_bps: int = Field(300, client_data=ClientFieldData(is_updatable=True))

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=self.stop_loss_pct / 100,
            take_profit=self.take_profit_pct / 100,
            time_limit=self.filled_order_expiration_min * 60,
            open_order_type=OrderType.LIMIT,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,  # Only market orders are supported for time_limit and stop_loss
            time_limit_order_type=OrderType.MARKET  # Only market orders are supported for time_limit and stop_loss
        )

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_name not in markets:
            markets[self.connector_name] = set()
        markets[self.connector_name].add(self.trading_pair)
        return markets

    # Generate config file: create --controller-config generic.overhill
    # Start the bot: start --script v2_with_controllers.py --conf conf_v2_with_controllers_overhill.yml


class Overhill(ControllerBase):
    def __init__(self, config: OverhillConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=self.config.candles_connector,
                trading_pair=self.config.trading_pair,
                interval=self.config.candles_interval,
                max_records=self.config.candles_length
            )]

        self.indicators_df = pd.DataFrame(columns=["timestamp", "timestamp_iso", "bbp"])

    async def update_processed_data(self):
        candles_config = self.config.candles_config[0]

        candles_df = self.market_data_provider.get_candles_df(connector_name=candles_config.connector,
                                                              trading_pair=candles_config.trading_pair,
                                                              interval=candles_config.interval,
                                                              max_records=candles_config.max_records)
        candles_df["index"] = candles_df["timestamp"]
        candles_df.set_index("index", inplace=True)

        candles_df["timestamp_iso"] = pd.to_datetime(candles_df["timestamp"], unit="s")
        candles_df.ta.bbands(length=self.config.bbands_length, std=self.config.bbands_std_dev, append=True)

        bbp_column_name = f"BBP_{self.config.bbands_length}_{self.config.bbands_std_dev}"
        self.update_indicators(candles_df, bbp_column_name)
        candles_df[bbp_column_name] = self.indicators_df["bbp"]
        candles_df["normalized_bbp"] = candles_df[bbp_column_name].apply(self.get_normalized_bbp)

        self.processed_data["features"] = candles_df

    def determine_executor_actions(self) -> List[ExecutorAction]:
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def create_actions_proposal(self) -> List[ExecutorAction]:
        create_actions = []

        mid_price = self.get_mid_price()
        unfilled_sell_executors, unfilled_buy_executors = self.get_unfilled_executors_by_side()

        if self.can_create_executor(unfilled_sell_executors, TradeType.SELL):
            sell_price = self.adjust_sell_price(mid_price)
            sell_executor_config = self.get_executor_config(TradeType.SELL, sell_price)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=sell_executor_config))

        if self.can_create_executor(unfilled_buy_executors, TradeType.BUY):
            buy_price = self.adjust_buy_price(mid_price)
            buy_executor_config = self.get_executor_config(TradeType.BUY, buy_price)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=buy_executor_config))

        return create_actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        stop_actions = []

        unfilled_sell_executors, unfilled_buy_executors = self.get_unfilled_executors_by_side()

        for unfilled_executor in unfilled_sell_executors + unfilled_buy_executors:
            has_expired = has_order_expired(unfilled_executor, self.config.unfilled_order_expiration_min * 60, self.market_data_provider.time())
            if has_expired:
                stop_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=unfilled_executor.id))

        latest_bbp = self.get_latest_normalized_bbp()

        trading_sell_executors = self.get_trading_executors_on_side(TradeType.SELL)
        for trading_executor in trading_sell_executors:
            if self.has_finished_a_hill_trend(latest_bbp):
                stop_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=trading_executor.id))

        trading_buy_executors = self.get_trading_executors_on_side(TradeType.BUY)
        for trading_executor in trading_buy_executors:
            if self.has_finished_a_valley_trend(latest_bbp):
                stop_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=trading_executor.id))

        return stop_actions

    def to_format_status(self) -> List[str]:
        features_df = self.processed_data.get("features", pd.DataFrame())

        if features_df.empty:
            return []

        columns_to_display = [
            "timestamp_iso",
            "close",
            f"BBP_{self.config.bbands_length}_{self.config.bbands_std_dev}",
            "normalized_bbp"
        ]

        return [format_df_for_printout(features_df[columns_to_display].tail(self.config.candles_length), table_format="psql", )]

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_executor(self, unfilled_executors: List[ExecutorInfo], side: TradeType) -> bool:
        if self.get_position_quote_amount() == 0 or len(unfilled_executors) > 0:
            return False

        latest_bbp = self.get_latest_normalized_bbp()

        if side == TradeType.SELL:
            return self.has_passed_a_hill_trend(latest_bbp, self.config.trend_begin_length)

        if side == TradeType.BUY:
            return self.has_passed_a_valley_trend(latest_bbp, self.config.trend_begin_length)

    def get_trade_connector(self) -> Optional[ConnectorBase]:
        try:
            return self.market_data_provider.get_connector(self.config.connector_name)
        except ValueError:  # When backtesting
            return None

    def get_mid_price(self):
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)

    def get_position_quote_amount(self) -> Decimal:
        # If balance = 100 USDT with leverage 20x, the quote position should be 500
        # TODO return Decimal(self.config.total_amount_quote * self.config.leverage / 4)
        return Decimal(self.config.total_amount_quote * self.config.leverage / 20)

    def get_best_ask(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestAsk)

    def get_best_bid(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestBid)

    def _get_best_ask_or_bid(self, price_type: PriceType) -> Decimal:
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, price_type)

    def get_executor_config(self, side: TradeType, ref_price: Decimal) -> PositionExecutorConfig:
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=side,
            entry_price=ref_price,
            amount=self.get_position_quote_amount() / ref_price,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage
        )

    def get_unfilled_executors_by_side(self) -> Tuple[List[ExecutorInfo], List[ExecutorInfo]]:
        unfilled_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda e: e.connector_name == self.config.connector_name and e.is_active and not e.is_trading
        )

        unfilled_sell_executors = [e for e in unfilled_executors if e.side == TradeType.SELL]
        unfilled_buy_executors = [e for e in unfilled_executors if e.side == TradeType.BUY]

        return unfilled_sell_executors, unfilled_buy_executors

    def get_trading_executors_on_side(self, side: TradeType) -> List[ExecutorInfo]:
        trading_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda e: e.connector_name == self.config.connector_name and e.is_active and e.is_trading
        )

        return [e for e in trading_executors if e.side == side]

    def get_last_terminated_executor_by_side(self) -> Tuple[Optional[ExecutorInfo], Optional[ExecutorInfo]]:
        terminated_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda e: e.connector_name == self.config.connector_name and e.is_done
        )

        last_sell_executor: Optional[ExecutorInfo] = None
        last_buy_executor: Optional[ExecutorInfo] = None

        for executor in reversed(terminated_executors):
            if last_sell_executor is None and executor.side == TradeType.SELL:
                last_sell_executor = executor
            if last_buy_executor is None and executor.side == TradeType.BUY:
                last_buy_executor = executor

            # If both are found, no need to continue the loop
            if last_sell_executor and last_buy_executor:
                break

        return last_sell_executor, last_buy_executor

    def adjust_sell_price(self, mid_price: Decimal) -> Decimal:
        total_adjustment = self.config.delta_with_best_bid_or_ask_bps / 10000

        ref_price = mid_price * Decimal(1 + total_adjustment)

        self.logger().info(f"Adjusting SELL price. mid:{mid_price}")
        self.logger().info(f"Adjusting SELL price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        return ref_price

    def adjust_buy_price(self, mid_price: Decimal) -> Decimal:
        total_adjustment = self.config.delta_with_best_bid_or_ask_bps / 10000

        ref_price = mid_price * Decimal(1 - total_adjustment)

        self.logger().info(f"Adjusting BUY price. mid:{mid_price}")
        self.logger().info(f"Adjusting BUY price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        return ref_price

    #
    # Custom functions specific to this controller
    #

    def update_indicators(self, df: pd.DataFrame, bbp_column_name: str):
        rows_to_add = []

        for _, row in df.iloc[:-1].iterrows():  # Last row is excluded, as it contains incomplete data
            if pd.notna(row[bbp_column_name]):
                timestamp = row["timestamp"]
                if self._get_indicators_for_timestamp(timestamp) is None:
                    rows_to_add.append({
                        "timestamp": timestamp,
                        "timestamp_iso": row["timestamp_iso"],
                        "bbp": row[bbp_column_name]
                    })

        if len(rows_to_add) > 0:
            new_rows = pd.DataFrame(rows_to_add)
            self.indicators_df = pd.concat([self.indicators_df, new_rows], ignore_index=True)
            self.indicators_df["index"] = self.indicators_df["timestamp"]
            self.indicators_df.set_index("index", inplace=True)

    def _get_indicators_for_timestamp(self, timestamp: float) -> Optional:
        matching_row = self.indicators_df.query(f"timestamp == {timestamp}")

        if not matching_row.empty:
            return matching_row.iloc[0]
        else:
            return None

    @staticmethod
    def get_normalized_bbp(bbp: float) -> float:
        return bbp - 0.5

    # TODO: remove
    # def get_latest_normalized_bbp(self, realtime_or_complete: str) -> float:
    #     if realtime_or_complete not in ["real-time", "complete"]:
    #         self.logger().error("get_latest_normalized_bbp() called with invalid argument")
    #         HummingbotApplication.main_application().stop()
    #         return -1.0
    #
    #     index = -1 if realtime_or_complete == "real-time" else -2
    #     return self.processed_data["features"]["normalized_bbp"].iloc[index]

    def get_latest_normalized_bbp(self) -> float:
        return self.processed_data["features"]["normalized_bbp"].iloc[-2]

    def has_passed_a_hill_trend(self, latest_bbp: float, trend_length: int) -> bool:
        if latest_bbp > -self.config.trend_bbp_threshold:
            return False

        intervals_to_consider: int = self._get_intervals_to_consider(trend_length)
        preceding_items = self.processed_data["features"]["normalized_bbp"].iloc[-intervals_to_consider:-2]

        return self._contains_consecutive_positive_items(preceding_items, trend_length)

    def has_passed_a_valley_trend(self, latest_bbp: float, trend_length: int) -> bool:
        if latest_bbp < self.config.trend_bbp_threshold:
            return False

        intervals_to_consider: int = self._get_intervals_to_consider(trend_length)
        preceding_items = self.processed_data["features"]["normalized_bbp"].iloc[-intervals_to_consider:-2]

        return self._contains_consecutive_negative_items(preceding_items, trend_length)

    def has_finished_a_valley_trend(self, latest_bbp: float) -> bool:
        return self.has_passed_a_hill_trend(latest_bbp, self.config.trend_end_length)

    def has_finished_a_hill_trend(self, latest_bbp: float) -> bool:
        return self.has_passed_a_valley_trend(latest_bbp, self.config.trend_end_length)

    # TODO @staticmethod
    def _contains_consecutive_positive_items(self, series: List, count: int) -> bool:
        consecutive_count = 0

        for value in series:
            if value > 0:
                consecutive_count += 1
                if consecutive_count >= count:
                    self.logger().info(f"_contains_consecutive_positive_items. Returning True. count:{count}")
                    self.logger().info(f"_contains_consecutive_positive_items. series:{series}")
                    return True
            else:
                consecutive_count = 0

        return False

    # TODO @staticmethod
    def _contains_consecutive_negative_items(self, series: List, count: int) -> bool:
        consecutive_count = 0

        for value in series:
            if value < 0:
                consecutive_count += 1
                if consecutive_count >= count:
                    self.logger().info(f"_contains_consecutive_negative_items. Returning True. count:{count}")
                    self.logger().info(f"_contains_consecutive_negative_items. series:{series}")
                    return True
            else:
                consecutive_count = 0

        return False

    @staticmethod
    def _get_intervals_to_consider(trend_length: int) -> int:
        return trend_length + 4
