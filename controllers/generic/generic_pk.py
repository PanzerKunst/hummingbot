from decimal import Decimal
from typing import Dict, List, Optional, Set

import pandas_ta as ta  # noqa: F401

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from scripts.utility.my_utils import has_order_expired


class GenericPkConfig(ControllerConfigBase):
    controller_name: str = "generic_pk"
    connector_name: str = "okx"  # Do not rename attribute - used by BacktestingEngineBase
    trading_pair: str = "AAVE-USDT"  # Do not rename attribute - used by BacktestingEngineBase
    unfilled_order_expiration: int = 60

    leverage: int = 5  # TODO: 20
    position_mode: PositionMode = PositionMode.HEDGE

    # Triple Barrier
    stop_loss_pct: float = 1.4
    take_profit_pct: float = 0.7
    filled_order_expiration_min: int = 60

    # Technical analysis
    bollinger_bands_length: int = 7
    bollinger_bands_std_dev: float = 2.2

    # Candles
    candles_interval: str = "1m"
    candles_length: int = bollinger_bands_length
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor

    # Maker orders settings
    spread_pct: float = 1.0
    bbp_ref_price_adjustment_pct: int = 5

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


class GenericPk(ControllerBase):
    def __init__(self, config: GenericPkConfig, *args, **kwargs):
        self.config = config

        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=self.config.connector_name,
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
        # Add indicators
        candles_df.ta.bbands(length=self.config.bollinger_bands_length, std=self.config.bollinger_bands_std_dev, append=True)
        self.processed_data["features"] = candles_df

    def determine_executor_actions(self) -> List[ExecutorAction]:
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def create_actions_proposal(self) -> List[ExecutorAction]:
        create_actions = []

        latest_df_row = self.processed_data["features"].iloc[-1]
        bollinger_bands_pct = latest_df_row[f"BBP_{self.config.bollinger_bands_length}_{self.config.bollinger_bands_std_dev}"]
        ref_price = self.adjust_ref_price(self.get_mid_price(), bollinger_bands_pct)

        # TODO: remove
        # mid_price = self.get_mid_price()

        sell_price = ref_price * Decimal(1 + self.config.spread_pct / 100)
        buy_price = ref_price * Decimal(1 - self.config.spread_pct / 100)

        # TODO: make sure those are not better than best ask and bid

        unfilled_executors = self.get_active_executors(self.config.connector_name, True)

        sell_executor_config = self.get_executor_config(unfilled_executors, TradeType.SELL, sell_price)
        if sell_executor_config is not None:
            create_actions.append(CreateExecutorAction(executor_config=sell_executor_config))

        buy_executor_config = self.get_executor_config(unfilled_executors, TradeType.BUY, buy_price)
        if buy_executor_config is not None:
            create_actions.append(CreateExecutorAction(executor_config=buy_executor_config))

        return create_actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        stop_actions = []

        for unfilled_executor in self.get_active_executors(self.config.connector_name, True):
            if has_order_expired(unfilled_executor, self.config.unfilled_order_expiration, self.market_data_provider.time()):
                stop_actions.append(StopExecutorAction(executor_id=unfilled_executor.id))

        # TODO: Unfilled ask orders should expire sooner when bollinger_bands_pct is close to 1
        # TODO: Unfilled ask orders should expire sooner when bollinger_bands_pct is close to 0

        return stop_actions

    #
    # Custom functions
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

    def adjust_ref_price(self, mid_price: Decimal, bbp: float) -> Decimal:
        normalized_bb = 2 * bbp - 1  # Between -1 and 1
        adjustment_factor = 1 + (self.config.bbp_ref_price_adjustment_pct / 100 * normalized_bb)

        return mid_price * Decimal(adjustment_factor)

    def get_position_quote_amount(self) -> Decimal:
        _, quote_currency = split_hb_trading_pair(self.config.trading_pair)
        trade_connector = self.get_trade_connector()

        if trade_connector is None:  # When backtesting
            return Decimal(100)

        available_quote_balance = trade_connector.get_available_balance(quote_currency)

        if available_quote_balance < 1:
            return Decimal(0)

        # If balance = 100 USDT with leverage 20x, the quote position should be 1000
        return Decimal(available_quote_balance * self.config.leverage / 2)

    def get_best_ask(self) -> Decimal:
        return self.get_best_ask_or_bid(PriceType.BestAsk)

    def get_best_bid(self) -> Decimal:
        return self.get_best_ask_or_bid(PriceType.BestBid)

    def get_best_ask_or_bid(self, price_type: PriceType) -> Decimal:
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, price_type)

    def get_executor_config(self, executors: List[ExecutorInfo], side: TradeType, ref_price: Decimal) -> Optional[PositionExecutorConfig]:
        unfilled_side_executors = [executor for executor in executors if executor.side == side]

        # Only create a new position if there is none on that side
        if len(unfilled_side_executors) > 0:
            return None

        quote_amount = self.get_position_quote_amount()

        if quote_amount == 0:
            return None

        best_ask = self.get_best_ask()
        best_bid = self.get_best_bid()

        self.logger().info(f"NEW POSITION. Side: {side}, best_ask: {best_ask}, best_bid: {best_bid}, ref_price: {ref_price}")

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
