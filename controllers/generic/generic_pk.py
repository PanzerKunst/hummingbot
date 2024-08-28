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
    connector_name: str = "okx_perpetual"  # Do not rename attribute - used by BacktestingEngineBase
    trading_pair: str = "AAVE-USDT"  # Do not rename attribute - used by BacktestingEngineBase

    leverage: int = 5  # TODO: 20
    position_mode: PositionMode = PositionMode.HEDGE
    total_amount_quote: int = 100  # Unused. Specified here to avoid prompt

    # Triple Barrier
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 1.0
    filled_order_expiration_min: int = 60

    # TODO: dymanic SL, TP?

    # Technical analysis
    bollinger_bands_length: int = 7
    bollinger_bands_std_dev: float = 2.0

    # Candles
    candles_interval: str = "1m"
    candles_length: int = bollinger_bands_length
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor

    # Maker orders settings
    unfilled_order_expiration_min: int = 5
    min_spread_pct: float = 0.5
    normalized_bbp_mult: float = 0.02

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
        bbp_series = candles_df[f"BBP_{self.config.bollinger_bands_length}_{self.config.bollinger_bands_std_dev}"]
        self.processed_data["normalized_bbp"] = bbp_series.apply(self.get_normalized_bbp)

        self.processed_data["features"] = candles_df

    def determine_executor_actions(self) -> List[ExecutorAction]:
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def create_actions_proposal(self) -> List[ExecutorAction]:
        create_actions = []

        mid_price = self.get_mid_price()
        latest_normalized_bbp = self.get_latest_normalized_bbp()

        # TODO: remove
        self.logger().info(f"mid_price: {mid_price}, latest_normalized_bbp: {latest_normalized_bbp}")

        sell_price = self.adjust_sell_price(mid_price, latest_normalized_bbp)
        buy_price = self.adjust_buy_price(mid_price, latest_normalized_bbp)

        unfilled_executors = self.get_active_executors(self.config.connector_name, True)

        # TODO: remove
        for ue in unfilled_executors:
            summary = {
                "status": ue.status,
                "side": ue.config.side,
                "is_active": ue.is_active,
                "is_trading": ue.is_trading,
                "filled_amount_quote": ue.filled_amount_quote
            }
            self.logger().info(f"unfilled_executor: {summary}")

        sell_executor_config = self.get_executor_config(unfilled_executors, TradeType.SELL, sell_price)
        if sell_executor_config is not None:
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=sell_executor_config))

        buy_executor_config = self.get_executor_config(unfilled_executors, TradeType.BUY, buy_price)
        if buy_executor_config is not None:
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=buy_executor_config))

        return create_actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        stop_actions = []

        for unfilled_executor in self.get_active_executors(self.config.connector_name, True):
            if self.should_stop_unfilled_executor(unfilled_executor):
                self.logger().info("Stopping unfilled executor")
                stop_actions.append(StopExecutorAction(executor_id=unfilled_executor.id))

        return stop_actions

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
            return Decimal(100)

        available_quote_balance = trade_connector.get_available_balance(quote_currency)

        if available_quote_balance < 1:
            return Decimal(0)

        # If balance = 100 USDT with leverage 20x, the quote position should be 1000
        return Decimal(available_quote_balance * self.config.leverage / 2)

    def get_best_ask(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestAsk)

    def get_best_bid(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestBid)

    def _get_best_ask_or_bid(self, price_type: PriceType) -> Decimal:
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, price_type)

    def get_executor_config(self, unfilled_executors: List[ExecutorInfo], side: TradeType, ref_price: Decimal) -> Optional[PositionExecutorConfig]:
        unfilled_side_executors = [e for e in unfilled_executors if e.side == side]

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

    def should_stop_unfilled_executor(self, unfilled_executor: ExecutorInfo) -> bool:
        is_volatility_too_high = abs(self.get_latest_normalized_bbp()) > 1

        if is_volatility_too_high:
            self.logger().info(f"is_volatility_too_high. latest_normalized_bbp: {self.get_latest_normalized_bbp()}")
            return True

        return has_order_expired(unfilled_executor, self.config.unfilled_order_expiration_min * 60, self.market_data_provider.time())

    #
    # Custom functions specific to this controller
    #

    @staticmethod
    def get_normalized_bbp(bbp: float) -> float:
        return 2 * bbp - 1  # Between -1 and 1

    def get_latest_normalized_bbp(self) -> float:
        return self.processed_data["normalized_bbp"].iloc[-1]

    def adjust_sell_price(self, mid_price: Decimal, latest_normalized_bbp: float) -> Decimal:
        price_with_spread: Decimal = mid_price * Decimal(1 + self.config.min_spread_pct / 100)

        if latest_normalized_bbp < 0:
            return price_with_spread

        spread_mult = latest_normalized_bbp * self.config.normalized_bbp_mult + 1
        return price_with_spread * Decimal(spread_mult)

    def adjust_buy_price(self, mid_price: Decimal, latest_normalized_bbp: float) -> Decimal:
        price_with_spread: Decimal = mid_price * Decimal(1 - self.config.min_spread_pct / 100)

        if latest_normalized_bbp > 0:
            return price_with_spread

        spread_div = abs(latest_normalized_bbp * self.config.normalized_bbp_mult - 1)
        return price_with_spread / Decimal(spread_div)
