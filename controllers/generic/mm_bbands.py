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
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from scripts.utility.my_utils import has_order_expired, timestamp_to_iso


class MmBbandsConfig(ControllerConfigBase):
    # Standard attributes - avoid renaming
    controller_name: str = "mm_bbands"
    connector_name: str = "okx_perpetual"
    trading_pair: str = "POPCAT-USDT"
    total_amount_quote: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    leverage: int = 20
    position_mode: PositionMode = PositionMode.HEDGE

    cooldown_time_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = Field(7, client_data=ClientFieldData(is_updatable=True))

    # Triple Barrier
    stop_loss_pct: Decimal = Field(0.7, client_data=ClientFieldData(is_updatable=True))
    take_profit_pct: Decimal = Field(0.7, client_data=ClientFieldData(is_updatable=True))
    filled_order_expiration_min: int = Field(1000, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    bbands_length_for_trend: int = Field(6, client_data=ClientFieldData(is_updatable=True))
    bbands_std_dev_for_trend: Decimal = Field(2.0, client_data=ClientFieldData(is_updatable=True))
    bbands_length_for_volatility: int = Field(2, client_data=ClientFieldData(is_updatable=True))
    bbands_std_dev_for_volatility: Decimal = Field(3.0, client_data=ClientFieldData(is_updatable=True))
    high_volatility_threshold: Decimal = Field(3.0, client_data=ClientFieldData(is_updatable=True))
    rsi_length: int = Field(12, client_data=ClientFieldData(is_updatable=True))

    # Candles
    candles_connector: str = "okx_perpetual"
    candles_interval: str = "1m"
    candles_length: int = 24
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor

    # Maker orders settings
    default_spread_pct: Decimal = Field(0.5, client_data=ClientFieldData(is_updatable=True))

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_name not in markets:
            markets[self.connector_name] = set()
        markets[self.connector_name].add(self.trading_pair)
        return markets

# Generate config file: create --controller-config generic.mm_bbands
# Start the bot: start --script v2_with_controllers.py --conf conf_v2_with_controllers_mm_bbands.yml
# Quickstart script: -p=a -f v2_with_controllers.py -c conf_v2_with_controllers_mm_bbands.yml


class MmBbands(ControllerBase):
    last_terminated_sell_executor: Optional[ExecutorInfo] = None
    last_terminated_sell_executor_timestamp: float = 0.0
    last_terminated_buy_executor: Optional[ExecutorInfo] = None
    last_terminated_buy_executor_timestamp: float = 0.0

    def __init__(self, config: MmBbandsConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=self.config.candles_connector,
                trading_pair=self.config.trading_pair,
                interval=self.config.candles_interval,
                max_records=self.config.candles_length
            )]

    async def update_processed_data(self):
        candles_config = self.config.candles_config[0]

        candles_df = self.market_data_provider.get_candles_df(connector_name=candles_config.connector,
                                                              trading_pair=candles_config.trading_pair,
                                                              interval=candles_config.interval,
                                                              max_records=candles_config.max_records)
        candles_df["timestamp_iso"] = pd.to_datetime(candles_df["timestamp"], unit="s")

        bbands_for_trend = candles_df.ta.bbands(length=self.config.bbands_length_for_trend, std=self.config.bbands_std_dev_for_trend)
        candles_df["bbp"] = bbands_for_trend[f"BBP_{self.config.bbands_length_for_trend}_{self.config.bbands_std_dev_for_trend}"]
        candles_df["normalized_bbp"] = candles_df["bbp"].apply(self.normalize_bbp)

        bbands_for_volatility = candles_df.ta.bbands(length=self.config.bbands_length_for_volatility, std=self.config.bbands_std_dev_for_volatility)
        candles_df["bbb_for_volatility"] = bbands_for_volatility[f"BBB_{self.config.bbands_length_for_volatility}_{self.config.bbands_std_dev_for_volatility}"]

        rsi = candles_df.ta.rsi(length=self.config.rsi_length)
        candles_df["normalized_rsi"] = rsi.apply(self.normalize_rsi)

        self.processed_data["features"] = candles_df

        self.check_trading_executors()

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

        is_high_volatility: bool = self.is_high_volatility()

        if is_high_volatility:
            self.logger().info(f"##### is_high_volatility -> Stopping unfilled executors #####")

        unfilled_sell_executors, unfilled_buy_executors = self.get_unfilled_executors_by_side()

        for unfilled_executor in unfilled_sell_executors + unfilled_buy_executors:
            has_expired = has_order_expired(unfilled_executor, self.config.unfilled_order_expiration_min * 60, self.market_data_provider.time())

            # TODO: remove
            if has_expired:
                self.logger().info("has_expired:true")

            if has_expired or is_high_volatility:
                stop_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=unfilled_executor.id))

        return stop_actions

    def to_format_status(self) -> List[str]:
        features_df = self.processed_data.get("features", pd.DataFrame())

        if features_df.empty:
            return []

        columns_to_display = [
            "timestamp_iso",
            "close",
            "normalized_bbp",
            "bbb_for_volatility",
            "normalized_rsi"
        ]

        return [format_df_for_printout(features_df[columns_to_display].tail(self.config.bbands_length_for_trend), table_format="psql", )]

    #
    # Custom functions potentially interesting for other controllers
    #

    # DOES NOT TRIGGER. Only works on V2 scripts
    # def did_fill_order(self, filled_event: OrderFilledEvent):
    #     position = filled_event.position
    #
    #     if not position:
    #         return
    #
    #     # TODO: remove
    #     self.logger().info(f"did_fill_order | filled_event: {filled_event}")

    def can_create_executor(self, unfilled_executors: List[ExecutorInfo], side: TradeType) -> bool:
        if self.get_position_quote_amount(side) == 0 or self.is_high_volatility() or len(unfilled_executors) > 0:
            return False

        last_terminated_timestamp: float = self.last_terminated_sell_executor_timestamp if side == TradeType.SELL else self.last_terminated_buy_executor_timestamp

        if last_terminated_timestamp + self.config.cooldown_time_min * 60 > self.market_data_provider.time():
            self.logger().info("Cooldown not passed yet")
            return False

        return True

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
            if not last_sell_executor and executor.side == TradeType.SELL:
                last_sell_executor = executor
            if not last_buy_executor and executor.side == TradeType.BUY:
                last_buy_executor = executor

            # If both are found, no need to continue the loop
            if last_sell_executor and last_buy_executor:
                break

        return last_sell_executor, last_buy_executor

    def get_trade_connector(self) -> Optional[ConnectorBase]:
        try:
            return self.market_data_provider.get_connector(self.config.connector_name)
        except ValueError:  # When backtesting
            return None

    def get_mid_price(self):
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)

    def get_position_quote_amount(self, side: TradeType) -> Decimal:
        amount_quote = Decimal(self.config.total_amount_quote)

        # If balance = 100 USDT with leverage 20x, the quote position should be 500
        position_quote_amount = amount_quote * self.config.leverage / 4

        if side == TradeType.SELL:
            position_quote_amount = position_quote_amount * Decimal(0.67)  # Less, because closing a Short position on SL costs significantly more

        return position_quote_amount

    def get_best_ask(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestAsk)

    def get_best_bid(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestBid)

    def _get_best_ask_or_bid(self, price_type: PriceType) -> Decimal:
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, price_type)

    def get_executor_config(self, side: TradeType, ref_price: Decimal) -> PositionExecutorConfig:
        triple_barrier_config = self.get_triple_barrier_config()

        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=side,
            entry_price=ref_price,
            amount=self.get_position_quote_amount(side) / ref_price,
            triple_barrier_config=triple_barrier_config,
            leverage=self.config.leverage
        )

    def get_triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=self.config.stop_loss_pct / 100,
            take_profit=self.config.take_profit_pct / 100,
            time_limit=self.config.filled_order_expiration_min * 60,
            open_order_type=OrderType.LIMIT,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,  # Only market orders are supported for time_limit and stop_loss
            time_limit_order_type=OrderType.MARKET  # Only market orders are supported for time_limit and stop_loss
        )

    def adjust_sell_price(self, mid_price: Decimal) -> Decimal:
        volatility_adjustment_pct: Decimal = Decimal(0)

        avg_last_three_bbb = self.get_avg_last_tree_bbb()
        if avg_last_three_bbb > 0:
            volatility_adjustment_pct += avg_last_three_bbb * Decimal(0.5)

        trend_adjustment_pct: Decimal = Decimal(0)

        trading_sell_executors, _ = self.get_trading_executors_by_side()

        # If there is a trading SELL position with negative PnL, we add to the adjustment
        for trading_executor in trading_sell_executors:
            pnl_pct = trading_executor.net_pnl_pct * 100
            if pnl_pct < 0:
                self.logger().info(f"Adding SELL position while negative filled position of pnl_pct {trading_executor.net_pnl_pct}")
                trend_adjustment_pct += -pnl_pct * 2

        if self.has_sl_occurred_on_side(TradeType.SELL) and self.is_still_trending_up():
            self.logger().info("self.has_sl_occurred_on_sell_and_price_trending_up, increasing trend_adjustment_pct")
            trend_adjustment_pct += self.config.default_spread_pct * Decimal(0.5)

        latest_normalized_rsi = self.get_latest_normalized_rsi()
        rsi_adjustment_pct = -latest_normalized_rsi * Decimal(0.012)

        # If we're adding a new position while having a filled one on the same side, we increase the adjustments
        if len(trading_sell_executors) > 0:
            self.logger().info("Adding a position while having a filled one on the same side - increasing the adjustments")
            volatility_adjustment_pct += self.config.default_spread_pct * Decimal(0.5)
            trend_adjustment_pct += self.config.default_spread_pct * Decimal(0.5)

        default_adjustment = self.config.default_spread_pct / 100
        total_adjustment = default_adjustment + volatility_adjustment_pct / 100 + trend_adjustment_pct / 100 + rsi_adjustment_pct / 100

        ref_price = mid_price * Decimal(1 + total_adjustment)

        self.logger().info(f"Adjusting SELL price. mid:{mid_price}, avg_last_three_bbb:{avg_last_three_bbb}")
        self.logger().info(f"Adjusting SELL price. def_adj:{default_adjustment}, volatility_adjustment_pct:{volatility_adjustment_pct}, trend_adjustment_pct:{trend_adjustment_pct}, rsi_adjustment_pct:{rsi_adjustment_pct}")
        self.logger().info(f"Adjusting SELL price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        trading_order_sl_price = self.get_sl_price(TradeType.SELL)

        if trading_order_sl_price and trading_order_sl_price > ref_price:
            self.logger().info(f"There is a trading Short order whose SL {trading_order_sl_price} is above ref_price {ref_price}")
            self.logger().info(f"Returning {trading_order_sl_price * Decimal(1 + 0.002)}")
            return trading_order_sl_price * Decimal(1 + 0.002)

        return ref_price

    def adjust_buy_price(self, mid_price: Decimal) -> Decimal:
        volatility_adjustment_pct: Decimal = Decimal(0)

        avg_last_three_bbb = self.get_avg_last_tree_bbb()
        if avg_last_three_bbb > 0:
            volatility_adjustment_pct += avg_last_three_bbb * Decimal(0.5)

        trend_adjustment_pct: Decimal = Decimal(0)

        _, trading_buy_executors = self.get_trading_executors_by_side()

        # If there is a trading BUY position with negative PnL, we add to the adjustment
        for trading_executor in trading_buy_executors:
            pnl_pct = trading_executor.net_pnl_pct * 100
            if pnl_pct < 0:
                self.logger().info(f"Adding BUY position while negative filled position of pnl_pct {trading_executor.net_pnl_pct}")
                trend_adjustment_pct += -pnl_pct * 2

        if self.has_sl_occurred_on_side(TradeType.BUY) and self.is_still_trending_down():
            self.logger().info("self.has_sl_occurred_on_buy_and_price_trending_down, increasing trend_adjustment_pct")
            trend_adjustment_pct += self.config.default_spread_pct * Decimal(0.5)

        latest_normalized_rsi = self.get_latest_normalized_rsi()
        rsi_adjustment_pct = latest_normalized_rsi * Decimal(0.012)

        # If we're adding a new position while having a filled one on the same side, we increase the adjustments
        if len(trading_buy_executors) > 0:
            self.logger().info("Adding a position while having a filled one on the same side - increasing the adjustments")
            volatility_adjustment_pct += self.config.default_spread_pct * Decimal(0.5)
            trend_adjustment_pct += self.config.default_spread_pct * Decimal(0.5)

        default_adjustment = self.config.default_spread_pct / 100
        total_adjustment = default_adjustment + volatility_adjustment_pct / 100 + trend_adjustment_pct / 100 + rsi_adjustment_pct / 100

        ref_price = mid_price * Decimal(1 - total_adjustment)

        self.logger().info(f"Adjusting BUY price. mid:{mid_price}, avg_last_three_bbb:{avg_last_three_bbb}")
        self.logger().info(f"Adjusting BUY price. def_adj:{default_adjustment}, volatility_adjustment_pct:{volatility_adjustment_pct}, trend_adjustment_pct:{trend_adjustment_pct}, rsi_adjustment_pct:{rsi_adjustment_pct}")
        self.logger().info(f"Adjusting BUY price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        trading_order_sl_price = self.get_sl_price(TradeType.BUY)

        if trading_order_sl_price and trading_order_sl_price < ref_price:
            self.logger().info(f"There is a trading Long order whose SL {trading_order_sl_price} is below ref_price {ref_price}")
            self.logger().info(f"Returning {trading_order_sl_price * Decimal(1 - 0.002)}")
            return trading_order_sl_price * Decimal(1 - 0.002)

        return ref_price

    #
    # Custom functions specific to this controller
    #

    @staticmethod
    def normalize_bbp(bbp: float) -> Decimal:
        return Decimal(bbp - 0.5)

    @staticmethod
    def normalize_rsi(rsi: float) -> Decimal:
        return Decimal(rsi * 2 - 100)

    def get_latest_normalized_bbp(self) -> Decimal:
        bbp_series: pd.Series = self.processed_data["features"]["normalized_bbp"]
        bbp_previous_full_minute = Decimal(bbp_series.iloc[-2])
        bbp_current_incomplete_minute = Decimal(bbp_series.iloc[-1])

        return (
            max(bbp_previous_full_minute, bbp_current_incomplete_minute) if bbp_previous_full_minute > 0
            else min(bbp_previous_full_minute, bbp_current_incomplete_minute)
        )

    def get_latest_bbb(self) -> Decimal:
        bbb_series: pd.Series = self.processed_data["features"]["bbb_for_volatility"]
        bbb_previous_full_minute = Decimal(bbb_series.iloc[-2])
        bbb_current_incomplete_minute = Decimal(bbb_series.iloc[-1])
        return max(bbb_previous_full_minute, bbb_current_incomplete_minute)

    def get_avg_last_tree_bbb(self) -> Decimal:
        bbb_series: pd.Series = self.processed_data["features"]["bbb_for_volatility"]
        bbb_last_full_minute = Decimal(bbb_series.iloc[-2])
        bbb_before_that = Decimal(bbb_series.iloc[-3])
        bbb_even_before_that = Decimal(bbb_series.iloc[-4])
        return (bbb_last_full_minute + bbb_before_that + bbb_even_before_that) / 3

    def is_high_volatility(self) -> bool:
        return self.get_latest_bbb() > self.config.high_volatility_threshold

    def is_still_trending_up(self) -> bool:
        return self.get_latest_normalized_bbp() > -0.2

    def is_still_trending_down(self) -> bool:
        return self.get_latest_normalized_bbp() < 0.2

    def get_latest_normalized_rsi(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["features"]["normalized_rsi"]
        return Decimal(rsi_series.iloc[-1])

    def has_sl_occurred_on_side(self, side: TradeType) -> bool:
        return self._is_last_sell_executor_sl() if side == TradeType.SELL else self._is_last_buy_executor_sl()

    def check_trading_executors(self):
        terminated_sell_executor, terminated_buy_executor = self.get_last_terminated_executor_by_side()

        if terminated_sell_executor:
            self._check_for_stop_loss_on_sell(terminated_sell_executor)

        if terminated_buy_executor:
            self._check_for_stop_loss_on_buy(terminated_buy_executor)

    def _check_for_stop_loss_on_sell(self, last_terminated_executor: Optional[ExecutorInfo]):
        if self.last_terminated_sell_executor and self.last_terminated_sell_executor.id == last_terminated_executor.id:
            return

        close_type = last_terminated_executor.close_type

        if close_type not in (CloseType.TIME_LIMIT, CloseType.TAKE_PROFIT, CloseType.STOP_LOSS):
            return

        # TODO: remove
        self.logger().info(f"_check_for_stop_loss_on_sell() | updating self.last_terminated_sell_executor to: {close_type} with timestamp: {timestamp_to_iso(self.market_data_provider.time())}")

        self.last_terminated_sell_executor = last_terminated_executor
        self.last_terminated_sell_executor_timestamp = self.market_data_provider.time()

    def _check_for_stop_loss_on_buy(self, last_terminated_executor: Optional[ExecutorInfo]):
        if self.last_terminated_buy_executor and self.last_terminated_buy_executor.id == last_terminated_executor.id:
            return

        close_type = last_terminated_executor.close_type

        if close_type not in (CloseType.TIME_LIMIT, CloseType.TAKE_PROFIT, CloseType.STOP_LOSS):
            return

        # TODO: remove
        self.logger().info(f"_check_for_stop_loss_on_buy() | updating self.last_terminated_buy_executor to: {close_type} with timestamp: {timestamp_to_iso(self.market_data_provider.time())}")

        self.last_terminated_buy_executor = last_terminated_executor
        self.last_terminated_buy_executor_timestamp = self.market_data_provider.time()

    def _is_last_sell_executor_sl(self) -> bool:
        return self.last_terminated_sell_executor and self.last_terminated_sell_executor.close_type == CloseType.STOP_LOSS

    def _is_last_buy_executor_sl(self) -> bool:
        return self.last_terminated_buy_executor and self.last_terminated_buy_executor.close_type == CloseType.STOP_LOSS

    # TODO: a smarter system would handle correctly multiple open orders on the same side
    def get_sl_price(self, side: TradeType) -> Optional[Decimal]:
        trading_sell_executors, trading_buy_executors = self.get_trading_executors_by_side()
        trading_executors = trading_sell_executors if side == TradeType.SELL else trading_buy_executors

        if len(trading_executors) == 0:
            return None

        last_trading_executor = trading_executors[-1]
        executor_config = last_trading_executor.config
        entry_price = executor_config.entry_price
        stop_loss = executor_config.triple_barrier_config.stop_loss

        sl_price_sell: Decimal = entry_price * (1 + stop_loss)
        sl_price_buy: Decimal = entry_price * (1 - stop_loss)

        return sl_price_sell if side == TradeType.SELL else sl_price_buy
