from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pandas_ta as ta  # noqa: F401

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from scripts.utility.my_utils import Trend, has_order_expired


class GenericPkConfig(ControllerConfigBase):
    controller_name: str = "generic_pk"
    connector_name: str = "okx_perpetual"  # Do not rename attribute - used by BacktestingEngineBase
    trading_pair: str = "AAVE-USDT"  # Do not rename attribute - used by BacktestingEngineBase

    leverage: int = 20
    position_mode: PositionMode = PositionMode.HEDGE
    total_amount_quote: int = 100  # Specified here primarily to avoid prompt. Used only when backtesting
    unfilled_order_expiration_min: int = 10

    # Triple Barrier
    stop_loss_pct: float = 0.6
    take_profit_pct: float = 0.4
    filled_order_expiration_min: int = 1000

    # TODO: dymanic SL, TP?

    # Technical analysis
    bbands_length_for_trend: int = 12
    bbands_std_dev_for_trend: float = 2.0
    candles_count_for_trend: int = 12
    bbands_length_for_volatility: int = 2
    bbands_std_dev_for_volatility: float = 3.0
    volatility_threshold_bbb: float = 1.0

    # Candles
    candles_connector: str = "okx_perpetual"
    candles_interval: str = "1m"
    candles_length: int = candles_count_for_trend * 2
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor

    # Maker orders settings
    default_spread_pct: float = 0.5
    price_adjustment_volatility_threshold: float = 0.5

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

    # HB command to generate config file:
    # create --controller-config generic.generic_pk
    # start --script v2_with_controllers.py --conf conf_v2_with_controllers_generic_pk.yml


class GenericPk(ControllerBase):
    sl_executor_buy: Optional[ExecutorInfo] = None
    sl_executor_sell: Optional[ExecutorInfo] = None

    def __init__(self, config: GenericPkConfig, *args, **kwargs):
        self.config = config

        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=self.config.candles_connector,
                trading_pair=self.config.trading_pair,
                interval=self.config.candles_interval,
                max_records=self.config.candles_length
            )]

        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        candles_config = self.config.candles_config[0]

        candles_df = self.market_data_provider.get_candles_df(connector_name=candles_config.connector,
                                                              trading_pair=candles_config.trading_pair,
                                                              interval=candles_config.interval,
                                                              max_records=candles_config.max_records)
        candles_df["timestamp_iso"] = pd.to_datetime(candles_df["timestamp"], unit="s")

        # Add indicators
        candles_df.ta.bbands(length=self.config.bbands_length_for_trend, std=self.config.bbands_std_dev_for_trend, append=True)
        candles_df["normalized_bbp"] = candles_df[f"BBP_{self.config.bbands_length_for_trend}_{self.config.bbands_std_dev_for_trend}"].apply(self.get_normalized_bbp)

        window = self.config.candles_count_for_trend

        candles_df["is_trending_up"] = candles_df["normalized_bbp"].rolling(window=window, min_periods=window).apply(
            lambda x: (x > 0).all(), raw=True
        ).astype(bool)

        candles_df["is_trending_down"] = candles_df["normalized_bbp"].rolling(window=window, min_periods=window).apply(
            lambda x: (x < 0).all(), raw=True
        ).astype(bool)

        candles_df["avg_normalized_bbp"] = candles_df["normalized_bbp"].rolling(window=window).mean()

        bbands_for_volatility = candles_df.ta.bbands(length=self.config.bbands_length_for_volatility, std=self.config.bbands_std_dev_for_volatility)
        candles_df["bbb_for_volatility"] = bbands_for_volatility[f"BBB_{self.config.bbands_length_for_volatility}_{self.config.bbands_std_dev_for_volatility}"]

        self.processed_data["features"] = candles_df

        self.check_for_stop_loss()

    def determine_executor_actions(self) -> List[ExecutorAction]:
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def create_actions_proposal(self) -> List[ExecutorAction]:
        quote_amount = self.get_position_quote_amount()

        if quote_amount == 0 or self.is_high_volatility():
            return []

        create_actions = []

        mid_price = self.get_mid_price()
        latest_bbb = self.get_latest_bbb()
        is_trending_up = self.is_trending(Trend.UP)
        is_trending_down = self.is_trending(Trend.DOWN)

        unfilled_executors = self.get_active_executors(True)
        unfilled_sell_executors = [e for e in unfilled_executors if e.side == TradeType.SELL]  # TODO: use function "by_side"

        if len(unfilled_sell_executors) == 0:
            sell_price = self.adjust_sell_price(mid_price, latest_bbb, is_trending_up)
            sell_executor_config = self.get_executor_config(TradeType.SELL, sell_price, quote_amount)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=sell_executor_config))

        unfilled_buy_executors = [e for e in unfilled_executors if e.side == TradeType.BUY]

        if len(unfilled_buy_executors) == 0:
            buy_price = self.adjust_buy_price(mid_price, latest_bbb, is_trending_down)
            buy_executor_config = self.get_executor_config(TradeType.BUY, buy_price, quote_amount)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=buy_executor_config))

        return create_actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        stop_actions = []

        is_high_volatility: bool = self.is_high_volatility()

        if is_high_volatility:
            self.logger().info("##### is_high_volatility -> Stopping unfilled executors #####")

        for unfilled_executor in self.get_active_executors(True):
            has_expired = has_order_expired(unfilled_executor, self.config.unfilled_order_expiration_min * 60, self.market_data_provider.time())
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
            f"BBP_{self.config.bbands_length_for_trend}_{self.config.bbands_std_dev_for_trend}",
            "normalized_bbp",
            "is_trending_up",
            "is_trending_down",
            "avg_normalized_bbp",
            "bbb_for_volatility"
        ]

        return [format_df_for_printout(features_df[columns_to_display].tail(self.config.candles_count_for_trend), table_format="psql", )]

    #
    # Custom functions potentially interesting for other controllers
    #

    def get_active_executors(self, is_non_trading_only: bool = False) -> List[ExecutorInfo]:
        filter_func = (
            lambda e: e.connector_name == self.config.connector_name and e.is_active and not e.is_trading
        ) if is_non_trading_only else (
            lambda e: e.connector_name == self.config.connector_name and e.is_active
        )

        return self.filter_executors(
            executors=self.executors_info,
            filter_func=filter_func
        )

    def get_trade_connector(self) -> Optional[ConnectorBase]:
        try:
            return self.market_data_provider.get_connector(self.config.connector_name)
        except ValueError:  # When backtesting
            return None

    def get_mid_price(self):
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)

    def get_position_quote_amount(self) -> Decimal:
        _, quote_currency = split_hb_trading_pair(self.config.trading_pair)
        trade_connector = self.get_trade_connector()

        if trade_connector is None:  # When backtesting
            return Decimal(self.config.total_amount_quote)

        available_quote_balance = trade_connector.get_available_balance(quote_currency)

        if available_quote_balance < 1:
            return Decimal(0)

        # If balance = 100 USDT with leverage 20x, the quote position should be 500
        # TODO return Decimal(available_quote_balance * self.config.leverage / 4)
        return Decimal(available_quote_balance * self.config.leverage / 20)

    def get_best_ask(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestAsk)

    def get_best_bid(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestBid)

    def _get_best_ask_or_bid(self, price_type: PriceType) -> Decimal:
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, price_type)

    def get_executor_config(self, side: TradeType, ref_price: Decimal, quote_amount: Decimal) -> PositionExecutorConfig:
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=side,
            entry_price=ref_price,
            amount=quote_amount / ref_price,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage
        )

    def get_trading_executors_on_side(self, side: TradeType) -> List[ExecutorInfo]:
        active_executors = self.get_active_executors()
        return [e for e in active_executors if e.is_trading and e.side == side]

    def get_last_terminated_executor_by_side(self) -> Tuple[Optional[ExecutorInfo], Optional[ExecutorInfo]]:
        terminated_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda e: e.connector_name == self.config.connector_name and e.is_done
        )

        last_buy_executor: Optional[ExecutorInfo] = None
        last_sell_executor: Optional[ExecutorInfo] = None

        for executor in reversed(terminated_executors):
            if last_buy_executor is None and executor.side == TradeType.BUY:
                last_buy_executor = executor
            if last_sell_executor is None and executor.side == TradeType.SELL:
                last_sell_executor = executor

            # If both are found, no need to continue the loop
            if last_buy_executor and last_sell_executor:
                break

        return last_buy_executor, last_sell_executor

    #
    # Custom functions specific to this controller
    #

    @staticmethod
    def get_normalized_bbp(bbp: float) -> float:
        return bbp - 0.5

    def get_latest_normalized_bbp(self) -> float:
        return self.processed_data["features"]["normalized_bbp"].iloc[-1]

    def get_latest_avg_normalized_bbp(self) -> float:
        return self.processed_data["features"]["avg_normalized_bbp"].iloc[-1]

    def get_latest_bbb(self) -> float:
        return self.processed_data["features"]["bbb_for_volatility"].iloc[-1]

    def is_trending(self, trend: Trend) -> bool:
        column = "is_trending_up" if trend == Trend.UP else "is_trending_down"
        return self.processed_data["features"][column].iloc[-1]

    def is_market_trending(self) -> bool:
        return self.is_trending(Trend.UP) or self.is_trending(Trend.DOWN)

    def is_high_volatility(self) -> bool:
        return self.get_latest_bbb() > self.config.volatility_threshold_bbb

    def check_for_stop_loss(self):
        last_buy_executor, last_sell_executor = self.get_last_terminated_executor_by_side()

        if last_buy_executor is not None:
            close_type = last_buy_executor.close_type

            # TODO remove
            self.logger().info(f"last_buy_executor: {close_type}")

            if close_type == CloseType.TAKE_PROFIT:
                self.sl_executor_buy = None
            elif close_type == CloseType.STOP_LOSS:
                self.logger().info("##### last_buy_executor is_stop_loss #####")
                self.sl_executor_buy = last_buy_executor

        if last_sell_executor is not None:
            close_type = last_sell_executor.close_type

            # TODO remove
            self.logger().info(f"last_sell_executor: {close_type}")

            if close_type == CloseType.TAKE_PROFIT:
                self.sl_executor_sell = None
            elif close_type == CloseType.STOP_LOSS:
                self.logger().info("##### last_sell_executor is_stop_loss #####")
                self.sl_executor_sell = last_sell_executor

    def has_sl_occurred_on_sell_and_price_trending_up(self) -> bool:
        has_sl_occurred_on_sell: bool = self.sl_executor_sell is not None
        is_trending_up: bool = self.get_latest_normalized_bbp() > -0.2

        if has_sl_occurred_on_sell and not is_trending_up:
            self.logger().info("##### We passed the middle of the road again (has_sl_occurred_on_sell and not is_trending_up) resetting self.sl_executor #####")
            self.sl_executor_sell = None

        return has_sl_occurred_on_sell and is_trending_up

    def has_sl_occurred_on_buy_and_price_trending_down(self) -> bool:
        has_sl_occurred_on_buy: bool = self.sl_executor_buy is not None
        is_trending_down: bool = self.get_latest_normalized_bbp() < 0.2

        if has_sl_occurred_on_buy and not is_trending_down:
            self.logger().info("##### We passed the middle of the road again (has_sl_occurred_on_buy and not is_trending_down) resetting self.sl_executor #####")
            self.sl_executor_buy = None

        return has_sl_occurred_on_buy and is_trending_down

    def adjust_sell_price(self, mid_price: Decimal, latest_bbb: float, is_trending_up: bool) -> Decimal:
        default_adjustment = self.config.default_spread_pct / 100

        volatility_adjustment: float = 0.0

        if latest_bbb > self.config.price_adjustment_volatility_threshold:
            above_threshold = latest_bbb - self.config.price_adjustment_volatility_threshold
            volatility_adjustment += above_threshold * 0.02

        trend_adjustment_pct: float = 0.0

        # If there is a trading SELL position with negative PnL, we add 1% to the adjustment
        for trading_executor in self.get_trading_executors_on_side(TradeType.SELL):
            if trading_executor.net_pnl_pct < 0:
                self.logger().info(f"Adding SELL position while negative filled position of pnl_pct {trading_executor.net_pnl_pct}")
                trend_adjustment_pct += 1

        if self.has_sl_occurred_on_sell_and_price_trending_up():
            self.logger().info("self.has_sl_occurred_on_sell_and_price_trending_up, adding 1% to trend_adjustment_pct")
            trend_adjustment_pct += 1

        # If we're adding a new position while having a filled one on the same side, we double the adjustments
        if len(self.get_trading_executors_on_side(TradeType.SELL)) > 0:
            self.logger().info("Adding a position while having a filled one on the same side - doubling the adjustments")
            volatility_adjustment *= 2
            trend_adjustment_pct *= 2

        total_adjustment = default_adjustment + volatility_adjustment + trend_adjustment_pct / 100

        ref_price = mid_price * Decimal(1 + total_adjustment)

        self.logger().info(f"Adjusting SELL price. mid:{mid_price}, latest_bbb:{latest_bbb}, is_trending_up:{is_trending_up}")
        self.logger().info(f"Adjusting SELL price. def_adj:{default_adjustment}, volatility_adjustment:{volatility_adjustment}, trend_adjustment_pct:{trend_adjustment_pct}")
        self.logger().info(f"Adjusting SELL price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        return ref_price

    def adjust_buy_price(self, mid_price: Decimal, latest_bbb: float, is_trending_down: bool) -> Decimal:
        default_adjustment = self.config.default_spread_pct / 100

        volatility_adjustment: float = 0.0

        if latest_bbb > self.config.price_adjustment_volatility_threshold:
            above_threshold = latest_bbb - self.config.price_adjustment_volatility_threshold
            volatility_adjustment += above_threshold * 0.02

        trend_adjustment_pct: float = 0.0

        # If there is a trading BUY position with negative PnL, we add 1% to the adjustment
        for trading_executor in self.get_trading_executors_on_side(TradeType.BUY):
            if trading_executor.net_pnl_pct < 0:
                self.logger().info(f"Adding BUY position while negative filled position of pnl_pct {trading_executor.net_pnl_pct}")
                trend_adjustment_pct += 1

        if self.has_sl_occurred_on_buy_and_price_trending_down():
            self.logger().info("self.has_sl_occurred_on_buy_and_price_trending_down, adding 1% to trend_adjustment_pct")
            trend_adjustment_pct += 1

        # If we're adding a new position while having a filled one on the same side, we double the adjustments
        if len(self.get_trading_executors_on_side(TradeType.BUY)) > 0:
            self.logger().info("Adding a position while having a filled one on the same side - doubling the adjustments")
            volatility_adjustment *= 2
            trend_adjustment_pct *= 2

        total_adjustment = default_adjustment + volatility_adjustment + trend_adjustment_pct / 100

        ref_price = mid_price * Decimal(1 - total_adjustment)

        self.logger().info(f"Adjusting BUY price. mid:{mid_price}, latest_bbb:{latest_bbb}, is_trending_down:{is_trending_down}")
        self.logger().info(f"Adjusting BUY price. def_adj:{default_adjustment}, volatility_adjustment:{volatility_adjustment}, trend_adjustment_pct:{trend_adjustment_pct}")
        self.logger().info(f"Adjusting BUY price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        return ref_price
