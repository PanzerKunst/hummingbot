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


class GenericPkConfig(ControllerConfigBase):
    controller_name: str = "generic_pk"
    connector_name: str = "okx_perpetual"  # Do not rename attribute - used by BacktestingEngineBase
    trading_pair: str = "AAVE-USDT"  # Do not rename attribute - used by BacktestingEngineBase

    leverage: int = 20
    position_mode: PositionMode = PositionMode.HEDGE
    total_amount_quote: int = 100  # Specified here primarily to avoid prompt. Used only when backtesting

    # Triple Barrier
    stop_loss_pct: float = 2.5
    take_profit_pct: float = 1.0
    filled_order_expiration_min: int = 120

    # TODO: dymanic SL, TP?

    # Technical analysis
    bollinger_bands_length: int = 7
    bollinger_bands_std_dev: float = 2.0
    bollinger_bands_bandwidth_threshold: float = 1.5

    # Candles
    candles_connector: str = "okx_perpetual"
    candles_interval: str = "1m"
    candles_length: int = bollinger_bands_length * 2
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor

    # Maker orders settings
    min_spread_pct: float = 0.5
    normalized_bbp_mult: float = 0.05
    normalized_bbb_mult: float = 0.1

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
        # Add indicators
        candles_df.ta.bbands(length=self.config.bollinger_bands_length, std=self.config.bollinger_bands_std_dev, append=True)
        candles_df["timestamp_iso"] = pd.to_datetime(candles_df["timestamp"], unit="s")
        candles_df["normalized_bbp"] = candles_df[f"BBP_{self.config.bollinger_bands_length}_{self.config.bollinger_bands_std_dev}"].apply(self.get_normalized_bbp)

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
        latest_normalized_bbp = self.get_latest_normalized_bbp()
        latest_bbb = self.get_latest_bbb()

        unfilled_executors = self.get_active_executors(self.config.connector_name, True)
        unfilled_sell_executors = [e for e in unfilled_executors if e.side == TradeType.SELL]  # TODO: by_side

        if len(unfilled_sell_executors) == 0:
            sell_price = self.adjust_sell_price(mid_price, latest_normalized_bbp, latest_bbb)
            sell_executor_config = self.get_executor_config(TradeType.SELL, sell_price, quote_amount)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=sell_executor_config))

        unfilled_buy_executors = [e for e in unfilled_executors if e.side == TradeType.BUY]

        if len(unfilled_buy_executors) == 0:
            buy_price = self.adjust_buy_price(mid_price, latest_normalized_bbp, latest_bbb)
            buy_executor_config = self.get_executor_config(TradeType.BUY, buy_price, quote_amount)
            create_actions.append(CreateExecutorAction(controller_id=self.config.id, executor_config=buy_executor_config))

        return create_actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        stop_actions = []

        if self.should_stop_unfilled_executors():
            self.logger().info("##### Stopping unfilled executors #####")
            for unfilled_executor in self.get_active_executors(self.config.connector_name, True):
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
            f"BBB_{self.config.bollinger_bands_length}_{self.config.bollinger_bands_std_dev}"
        ]

        return [format_df_for_printout(features_df[columns_to_display].tail(self.config.bollinger_bands_length), table_format="psql",)]

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

    def should_stop_unfilled_executors(self) -> bool:
        is_volatility_too_high = (
            abs(self.get_latest_normalized_bbp()) > 0.3 and
            self.get_latest_bbb() > self.config.bollinger_bands_bandwidth_threshold
        )

        if is_volatility_too_high:
            self.logger().info(f"is_volatility_too_high. latest_normalized_bbp: {self.get_latest_normalized_bbp()}. latest_bbb: {self.get_latest_bbb()}")
            return True

        return False

    #
    # Custom functions specific to this controller
    #

    @staticmethod
    def get_normalized_bbp(bbp: float) -> float:
        return bbp - 0.5

    def get_latest_normalized_bbp(self) -> float:
        return self.processed_data["features"]["normalized_bbp"].iloc[-1]

    def get_latest_bbb(self) -> float:
        return self.processed_data["features"][f"BBB_{self.config.bollinger_bands_length}_{self.config.bollinger_bands_std_dev}"].iloc[-1]

    def adjust_sell_price(self, mid_price: Decimal, latest_normalized_bbp: float, latest_bbb: float) -> Decimal:
        default_adjustment = self.config.min_spread_pct / 100

        bbp_adjustment: float = 0.0

        if latest_normalized_bbp > 0:
            bbp_adjustment = latest_normalized_bbp * self.config.normalized_bbp_mult

        bbb_adjustment: float = 0.0

        if latest_bbb > 1:
            decimals = latest_bbb - 1  # Ex: 0.5, 1.0
            bbb_adjustment = decimals * self.config.normalized_bbb_mult  # Ex: 0.025, 0.05

        total_adjustment = default_adjustment + bbp_adjustment + bbb_adjustment

        ref_price = mid_price * Decimal(1 + total_adjustment)

        self.logger().info(f"Adjusting SELL price. mid:{mid_price}, norm_bbp:{latest_normalized_bbp}, bbb:{latest_bbb}")
        self.logger().info(f"Adjusting SELL price. def_adj:{default_adjustment}, bbp_adj:{bbp_adjustment}, bbb_adj:{bbb_adjustment}")
        self.logger().info(f"Adjusting SELL price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        return ref_price

    def adjust_buy_price(self, mid_price: Decimal, latest_normalized_bbp: float, latest_bbb: float) -> Decimal:
        default_adjustment = self.config.min_spread_pct / 100

        bbp_adjustment: float = 0.0

        if latest_normalized_bbp < 0:
            bbp_adjustment = abs(latest_normalized_bbp) * self.config.normalized_bbp_mult

        bbb_adjustment: float = 0.0

        if latest_bbb > 1:
            decimals = latest_bbb - 1  # Ex: 0.5, 1.0
            bbb_adjustment = decimals * self.config.normalized_bbb_mult  # Ex: 0.025, 0.05

        total_adjustment = default_adjustment + bbp_adjustment + bbb_adjustment

        ref_price = mid_price * Decimal(1 - total_adjustment)

        self.logger().info(f"Adjusting BUY price. mid:{mid_price}, norm_bbp:{latest_normalized_bbp}, bbb:{latest_bbb}")
        self.logger().info(f"Adjusting BUY price. def_adj:{default_adjustment}, bbp_adj:{bbp_adjustment}, bbb_adj:{bbb_adjustment}")
        self.logger().info(f"Adjusting BUY price. total_adj:{total_adjustment}, ref_price:{ref_price}")

        return ref_price
