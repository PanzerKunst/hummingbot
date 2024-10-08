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


class ExcaliburConfig(ControllerConfigBase):
    controller_name: str = "excalibur"
    connector_name: str = "kucoin_perpetual"  # Do not rename attribute - used by BacktestingEngineBase
    trading_pair: str = "SOL-USDT"  # Do not rename attribute - used by BacktestingEngineBase

    leverage: int = 20
    position_mode: PositionMode = PositionMode.ONEWAY
    total_amount_quote: int = Field(5, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = 1

    # Triple Barrier
    stop_loss_pct: Decimal = Field(8, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    sma_short_length: int = Field(50, client_data=ClientFieldData(is_updatable=True))
    sma_long_length: int = Field(100, client_data=ClientFieldData(is_updatable=True))

    # Candles
    candles_connector: str = "binance"
    candles_interval: str = "15m"
    candles_length: int = 150
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor

    # Trading algo
    delta_with_mid_price_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=self.stop_loss_pct / 100,
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

    # Generate config file: create --controller-config generic.excalibur
    # Start the bot: start --script v2_with_controllers.py --conf conf_v2_with_controllers_excalibur.yml
    # Quickstart script: -p=a -f v2_with_controllers.py -c conf_v2_with_controllers_excalibur.yml


class Excalibur(ControllerBase):
    def __init__(self, config: ExcaliburConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=self.config.candles_connector,
                trading_pair=self.config.trading_pair,
                interval=self.config.candles_interval,
                max_records=self.config.candles_length
            )]

        self.indicators_df = pd.DataFrame(columns=["timestamp", "timestamp_iso", "sma_short", "sma_long"])

    async def update_processed_data(self):
        candles_config = self.config.candles_config[0]

        candles_df = self.market_data_provider.get_candles_df(connector_name=candles_config.connector,
                                                              trading_pair=candles_config.trading_pair,
                                                              interval=candles_config.interval,
                                                              max_records=candles_config.max_records)
        candles_df["index"] = candles_df["timestamp"]
        candles_df.set_index("index", inplace=True)

        candles_df["timestamp_iso"] = pd.to_datetime(candles_df["timestamp"], unit="s")

        candles_df["sma_short"] = candles_df.ta.sma(length=self.config.sma_short_length)
        candles_df["sma_long"] = candles_df.ta.sma(length=self.config.sma_long_length)

        self.update_indicators(candles_df)
        candles_df["sma_short"] = self.indicators_df["sma_short"]
        candles_df["sma_long"] = self.indicators_df["sma_long"]

        self.processed_data["features"] = candles_df

    def determine_executor_actions(self) -> List[ExecutorAction]:
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def create_actions_proposal(self) -> List[ExecutorAction]:
        create_actions = []
        active_sell_executors, active_buy_executors = self.get_active_executors_by_side()
        mid_price = self.get_mid_price()

        if self.can_create_executor(active_sell_executors, TradeType.SELL):
            sell_price = self.adjust_sell_price(mid_price)
            sell_executor_config = self.get_executor_config(TradeType.SELL, sell_price)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=sell_executor_config))

        if self.can_create_executor(active_buy_executors, TradeType.BUY):
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

        trading_sell_executors, trading_buy_executors = self.get_trading_executors_by_side()

        for trading_executor in trading_sell_executors:
            if self.did_long_sma_cross_under_short():
                stop_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=trading_executor.id))

        for trading_executor in trading_buy_executors:
            if self.did_short_sma_cross_under_long():
                stop_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=trading_executor.id))

        return stop_actions

    def to_format_status(self) -> List[str]:
        features_df = self.processed_data.get("features", pd.DataFrame())

        if features_df.empty:
            return []

        columns_to_display = [
            "timestamp_iso",
            "close",
            "sma_short",
            "sma_long"
        ]

        return [format_df_for_printout(features_df[columns_to_display].tail(20), table_format="psql", )]

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_executor(self, active_executors: List[ExecutorInfo], side: TradeType) -> bool:
        if self.get_position_quote_amount() == 0 or len(active_executors) > 0:
            return False

        if side == TradeType.SELL:
            return self.did_short_sma_cross_under_long()

        return self.did_long_sma_cross_under_short()

    def get_trade_connector(self) -> Optional[ConnectorBase]:
        try:
            return self.market_data_provider.get_connector(self.config.connector_name)
        except ValueError:  # When backtesting
            return None

    def get_mid_price(self):
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)

    def get_position_quote_amount(self) -> Decimal:
        # If balance = 100 USDT with leverage 20x, the quote position should be 500
        return Decimal(self.config.total_amount_quote * self.config.leverage / 4)

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

    def get_active_executors_by_side(self) -> Tuple[List[ExecutorInfo], List[ExecutorInfo]]:
        active_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda e: e.connector_name == self.config.connector_name and e.is_active
        )

        active_sell_executors = [e for e in active_executors if e.side == TradeType.SELL]
        active_buy_executors = [e for e in active_executors if e.side == TradeType.BUY]

        return active_sell_executors, active_buy_executors

    def get_unfilled_executors_by_side(self) -> Tuple[List[ExecutorInfo], List[ExecutorInfo]]:
        active_sell_executors, active_buy_executors = self.get_active_executors_by_side()

        unfilled_sell_executors = [e for e in active_sell_executors if not e.is_trading]
        unfilled_buy_executors = [e for e in active_buy_executors if not e.is_trading]

        return unfilled_sell_executors, unfilled_buy_executors

    def get_trading_executors_by_side(self) -> Tuple[List[ExecutorInfo], List[ExecutorInfo]]:
        active_sell_executors, active_buy_executors = self.get_active_executors_by_side()

        trading_sell_executors = [e for e in active_sell_executors if e.is_trading]
        trading_buy_executors = [e for e in active_buy_executors if e.is_trading]

        return trading_sell_executors, trading_buy_executors

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
        total_adjustment = self.config.delta_with_mid_price_bps / 10000

        ref_price = mid_price * Decimal(1 + total_adjustment)

        self.logger().info(f"{self.config.trading_pair} Adjusting SELL price. mid_price:{mid_price}")
        self.logger().info(f"{self.config.trading_pair} Adjusting SELL price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        return ref_price

    def adjust_buy_price(self, mid_price: Decimal) -> Decimal:
        total_adjustment = self.config.delta_with_mid_price_bps / 10000

        ref_price = mid_price * Decimal(1 - total_adjustment)

        self.logger().info(f"{self.config.trading_pair} Adjusting BUY price. mid_price:{mid_price}")
        self.logger().info(f"{self.config.trading_pair} Adjusting BUY price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        return ref_price

    def update_indicators(self, df: pd.DataFrame):
        rows_to_add = []

        for _, row in df.iloc[:-1].iterrows():  # Last row is excluded, as it contains incomplete data
            if pd.notna(row["sma_short"]) and pd.notna(row["sma_long"]):
                timestamp = row["timestamp"]
                if self._get_indicators_for_timestamp(timestamp) is None:
                    rows_to_add.append({
                        "timestamp": timestamp,
                        "timestamp_iso": row["timestamp_iso"],
                        "sma_short": row["sma_short"],
                        "sma_long": row["sma_long"]
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

    #
    # Custom functions specific to this controller
    #

    def get_latest_sma(self, short_or_long: str) -> float:
        return self._get_sma_at_index(short_or_long, -2)

    def get_previous_sma(self, short_or_long: str) -> float:
        return self._get_sma_at_index(short_or_long, -3)

    def _get_sma_at_index(self, short_or_long: str, index: int) -> float:
        return self.processed_data["features"][f"sma_{short_or_long}"].iloc[index]

    def did_short_sma_cross_under_long(self) -> bool:
        result: bool = not self._is_latest_short_sma_over_long() and self._is_previous_short_sma_over_long()

        if result:
            self.logger().info("Short SMA crossed under long.")

        return result

    def did_long_sma_cross_under_short(self) -> bool:
        result: bool = self._is_latest_short_sma_over_long() and not self._is_previous_short_sma_over_long()

        if result:
            self.logger().info("Long SMA crossed under short.")

        return result

    def _is_latest_short_sma_over_long(self) -> bool:
        latest_short_minus_long: float = self.get_latest_sma("short") - self.get_latest_sma("long")
        return latest_short_minus_long > 0

    def _is_previous_short_sma_over_long(self) -> bool:
        previous_short_minus_long: float = self.get_previous_sma("short") - self.get_previous_sma("long")
        return previous_short_minus_long > 0
