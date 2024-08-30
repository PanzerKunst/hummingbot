from decimal import Decimal
from typing import Dict, List, Optional, Set

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
    stop_loss_pct: float = 0.5
    take_profit_pct: float = 0.3
    filled_order_expiration_min: int = 1000

    # TODO: dymanic SL, TP?

    # Technical analysis
    bbands_length_for_trend: int = 12
    bbands_std_dev_for_trend: float = 2.0
    candles_count_for_trend: int = 16
    bbands_length_for_volatility: int = 2
    bbands_std_dev_for_volatility: float = 3.0
    volatility_threshold_bbb: float = 1.0

    # Candles
    candles_connector: str = "okx_perpetual"
    candles_interval: str = "1m"
    candles_length: int = candles_count_for_trend * 2
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor

    # Maker orders settings
    min_spread_pct: float = 0.3

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

    def determine_executor_actions(self) -> List[ExecutorAction]:
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def create_actions_proposal(self) -> List[ExecutorAction]:
        quote_amount = self.get_position_quote_amount()

        if quote_amount == 0:
            return []

        create_actions = []

        mid_price = self.get_mid_price()
        latest_avg_normalized_bbp = self.get_latest_avg_normalized_bbp()

        unfilled_executors = self.get_active_executors(self.config.connector_name, True)
        unfilled_sell_executors = [e for e in unfilled_executors if e.side == TradeType.SELL]  # TODO: use function "by_side"

        if len(unfilled_sell_executors) == 0:
            sell_price = self.adjust_sell_price(mid_price, latest_avg_normalized_bbp)
            sell_executor_config = self.get_executor_config(TradeType.SELL, sell_price, quote_amount)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=sell_executor_config))

        unfilled_buy_executors = [e for e in unfilled_executors if e.side == TradeType.BUY]

        if len(unfilled_buy_executors) == 0:
            buy_price = self.adjust_buy_price(mid_price, latest_avg_normalized_bbp)
            buy_executor_config = self.get_executor_config(TradeType.BUY, buy_price, quote_amount)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=buy_executor_config))

        return create_actions

    # TODO: in case of a stop-loss, wait until we have crossed the road again before making a new order on that side
    # Or, close all orders on that side, and significantly increase the price adjustment

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        stop_actions = []

        is_market_trending_up: bool = self.is_market_trending(Trend.UP)
        is_market_trending_down: bool = self.is_market_trending(Trend.DOWN)
        is_market_trending: bool = is_market_trending_up or is_market_trending_down
        is_high_volatility: bool = self.get_latest_bbb() > self.config.volatility_threshold_bbb

        active_executors = self.get_active_executors(self.config.connector_name)

        if is_high_volatility or is_market_trending:
            self.logger().info(f"##### is_high_volatility:{is_high_volatility} or is_market_trending:{is_market_trending} -> Stopping unfilled executors #####")
            for unfilled_executor in [e for e in active_executors if not e.is_trading]:
                stop_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=unfilled_executor.id))

        if is_market_trending:
            for filled_executor in [e for e in active_executors if e.is_trading]:
                pnl_pct = filled_executor.net_pnl_pct * 100
                self.logger().info(f"##### is_market_trending + filled_executor with pnl:{pnl_pct} #####")
                if pnl_pct < 0 and abs(pnl_pct) > self.config.stop_loss_pct * 0.7:
                    self.logger().info("##### pnl too close to SL -> closing position #####")
                    stop_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=filled_executor.id))

        for unfilled_executor in [e for e in active_executors if not e.is_trading]:
            if has_order_expired(unfilled_executor, self.config.unfilled_order_expiration_min * 60, self.market_data_provider.time()):
                stop_actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=unfilled_executor.id))

        return stop_actions

    def to_format_status(self) -> List[str]:
        features_df = self.processed_data.get("features", pd.DataFrame())

        if features_df.empty:
            return []

        columns_to_display = [
            "timestamp_iso",
            "close",
            "bbp_for_trend",
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

    def get_active_executors(self, connector_name: str, is_non_trading_only: bool = False) -> List[ExecutorInfo]:
        filter_func = (
            lambda e: e.connector_name == connector_name and e.is_active and not e.is_trading
        ) if is_non_trading_only else (
            lambda e: e.connector_name == connector_name and e.is_active
        )

        active_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=filter_func
        )

        return active_executors

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

    def is_market_trending(self, trend: Trend) -> bool:
        column = "is_trending_up" if trend == Trend.UP else "is_trending_down"
        return self.processed_data["features"][column].iloc[-1]

    def adjust_sell_price(self, mid_price: Decimal, latest_avg_normalized_bbp: float) -> Decimal:
        default_adjustment = self.config.min_spread_pct / 100  # Ex

        bbp_adjustment: float = 0.0

        if latest_avg_normalized_bbp > 0.3:  # Ex
            bbp_adjustment = latest_avg_normalized_bbp * 0.0  # Ex

        total_adjustment = default_adjustment + bbp_adjustment  # Ex

        ref_price = mid_price * Decimal(1 + total_adjustment)  # mid_price *

        self.logger().info(f"Adjusting SELL price. mid:{mid_price}, avg_norm_bbp:{latest_avg_normalized_bbp}")
        self.logger().info(f"Adjusting SELL price. def_adj:{default_adjustment}, bbp_adj:{bbp_adjustment}")
        self.logger().info(f"Adjusting SELL price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        return ref_price

    def adjust_buy_price(self, mid_price: Decimal, latest_avg_normalized_bbp: float) -> Decimal:
        default_adjustment = self.config.min_spread_pct / 100  # Ex

        bbp_adjustment: float = 0.0

        if latest_avg_normalized_bbp < -0.3:  # Ex
            bbp_adjustment = abs(latest_avg_normalized_bbp) * 0.0  # Ex

        total_adjustment = default_adjustment + bbp_adjustment  # Ex

        ref_price = mid_price * Decimal(1 - total_adjustment)  # mid_price *

        self.logger().info(f"Adjusting BUY price. mid:{mid_price}, avg_norm_bbp:{latest_avg_normalized_bbp}")
        self.logger().info(f"Adjusting BUY price. def_adj:{default_adjustment}, bbp_adj:{bbp_adjustment}")
        self.logger().info(f"Adjusting BUY price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        return ref_price
