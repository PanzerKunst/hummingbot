import os
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType, PositionMode
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


class MerlinConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []
    markets: Dict[str, Set[str]] = {}
    config_update_interval: int = Field(10, client_data=ClientFieldData(is_updatable=True))

    connectors_and_pairs: Dict[str, Tuple[str, Decimal]] = {
        # connector_name: (trading_pair, price_adjustment_for_normalization)
        # "bitget_perpetual": "WIF-USDT",  Fails to connect
        "gate_io_perpetual": ("WIF-USDT", Decimal(0.002)),
        # "hyperliquid_perpetual": ("WIF-USD", Decimal(-0.0025)),  Balance zero for now
        "kucoin_perpetual": ("WIF-USDT", Decimal(0.0)),
        "okx_perpetual": ("WIF-USDT", Decimal(0.0))
    }

    leverage: int = 20  # Max 5 on Hyperliquid
    total_amount_quote: int = Field(5, client_data=ClientFieldData(is_updatable=True))
    cooldown_time_min: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    # unfilled_order_expiration: int = 1 TODO: try alternative with LIMIT orders at mid-price

    # Triple Barrier
    stop_loss_pct: float = Field(0.5, client_data=ClientFieldData(is_updatable=True))
    take_profit_pct: float = Field(10.0, client_data=ClientFieldData(is_updatable=True))
    filled_order_expiration_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))

    # Trading algo
    min_ask_bid_price_delta_to_open_bps: int = Field(8, client_data=ClientFieldData(is_updatable=True))  # 0.2% TODO: 20
    max_mid_price_delta_to_close_bps: int = Field(5, client_data=ClientFieldData(is_updatable=True))  # 0.05%

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=Decimal(self.stop_loss_pct / 100),
            # take_profit=Decimal(self.take_profit_pct / 100),
            time_limit=self.filled_order_expiration_min * 60,
            open_order_type=OrderType.MARKET,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,  # Only market orders are supported for time_limit and stop_loss
            time_limit_order_type=OrderType.MARKET  # Only market orders are supported for time_limit and stop_loss
        )


# Generate config file: create --script-config v2_merlin
# Start the bot: start --script v2_merlin.py --conf conf_v2_merlin_WIF.yml
# Quickstart script: -p=a -f v2_merlin.py -c conf_v2_merlin_WIF.yml


class Merlin(StrategyV2Base):
    @classmethod
    def init_markets(cls, config: MerlinConfig):
        markets = {}
        for connector_name, (trading_pair, _) in config.connectors_and_pairs.items():
            markets[connector_name] = [trading_pair]
        cls.markets = markets

    def __init__(self, connectors: Dict[str, ConnectorBase], config: MerlinConfig):
        super().__init__(connectors, config)
        self.config = config

        self.last_terminated_executor_timestamp: float = 0.0

        self.best_asks_and_bids: Dict[str, Tuple[Decimal, Decimal]] = {}
        self.best_arbitrage = None

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(PositionMode.ONEWAY)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        create_actions = []

        self.update_best_asks_and_bids()

        # Loop through the connectors, and find where:
        # - Best-bid is above best-ask (best-bid - best-ask > 0)
        # - That difference is the largest

        self.best_arbitrage = self.find_best_arbitrage()

        if self.best_arbitrage:
            best_ask_connector_name, best_ask_price, best_bid_connector_name, best_bid_price = self.best_arbitrage
            delta_bps = (best_bid_price - best_ask_price) / best_bid_price * 10000

            if delta_bps > self.config.min_ask_bid_price_delta_to_open_bps:
                # If that difference is above the threshold:
                # - Make a MARKET Short order at the best-bid
                # - And a MARKET Long order at the best-ask

                all_active_sell_executors, all_active_buy_executors = self.combine_active_executors_by_side()
                all_active_executors = all_active_sell_executors + all_active_buy_executors

                # TODO: remove
                self.logger().info(f"{delta_bps:.2f} > self.config.min_ask_bid_price_delta_to_open_bps")
                self.logger().info(f"len(all_active_executors):{len(all_active_executors)}")

                if self.can_create_executor(all_active_executors):
                    sell_executor_config = self.get_executor_config(best_bid_connector_name, TradeType.SELL, best_bid_price)
                    create_actions.append(CreateExecutorAction(executor_config=sell_executor_config))

                if self.can_create_executor(all_active_executors):
                    buy_executor_config = self.get_executor_config(best_ask_connector_name, TradeType.BUY, best_ask_price)
                    create_actions.append(CreateExecutorAction(executor_config=buy_executor_config))

                # TODO: try alternative with LIMIT orders at mid-price

        return create_actions

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        stop_actions = []

        active_sell_executor: Optional[ExecutorInfo] = None
        connector_name_where_shorting: str = ""

        active_buy_executor: Optional[ExecutorInfo] = None
        connector_name_where_longing: str = ""

        for connector_name, (trading_pair, _) in self.config.connectors_and_pairs.items():
            active_sell_executors, active_buy_executors = self.get_active_executors_by_side(connector_name)

            if len(active_sell_executors) > 0:
                # TODO: remove
                self.logger().info(f"len(active_sell_executors) > 0: {len(active_sell_executors)}")

                active_sell_executor = active_sell_executors[0]
                connector_name_where_shorting = connector_name

            if len(active_buy_executors) > 0:
                # TODO: remove
                self.logger().info(f"len(active_buy_executors) > 0: {len(active_buy_executors)}")

                active_buy_executor = active_buy_executors[0]
                connector_name_where_longing = connector_name

            if active_sell_executor and active_buy_executor:
                break

        if active_sell_executor and not active_buy_executor:
            # TODO: remove
            self.logger().error(f"ERROR: active_sell_executor and not active_buy_executor | active_sell_executor.is_trading: {active_sell_executor.is_trading}")

            if active_sell_executor.is_trading:
                stop_actions.append(StopExecutorAction(executor_id=active_sell_executor.id))

        if active_buy_executor and not active_sell_executor:
            # TODO: remove
            self.logger().error(f"ERROR: active_buy_executor and not active_sell_executor | active_buy_executor.is_trading: {active_buy_executor.is_trading}")

            if active_buy_executor.is_trading:
                stop_actions.append(StopExecutorAction(executor_id=active_buy_executor.id))

        if active_sell_executor and active_buy_executor and active_sell_executor.is_trading and active_buy_executor.is_trading:
            # If mid-prices are too close to each-other, we close both positions at MARKET price
            mid_price_where_shorting = self.get_mid_price_custom(connector_name_where_shorting)
            mid_price_where_longing = self.get_mid_price_custom(connector_name_where_longing)
            mid_price_delta_bps = (mid_price_where_shorting - mid_price_where_longing) / mid_price_where_shorting * 10000

            self.logger().info(f"mid_price_delta_bps for closing: {mid_price_delta_bps:.2f}")

            if mid_price_delta_bps < self.config.max_mid_price_delta_to_close_bps:
                stop_actions.extend([
                    StopExecutorAction(executor_id=active_sell_executor.id),
                    StopExecutorAction(executor_id=active_buy_executor.id)
                ])

        if len(stop_actions) > 0:
            self.last_terminated_executor_timestamp = self.market_data_provider.time()

        return stop_actions

    def format_status(self) -> str:
        original_status = super().format_status()
        custom_status = []

        if self.ready_to_trade:
            custom_status.append("\nOrder books:")
            order_books_df = pd.DataFrame()

            for connector_name, ask_and_bid in self.best_asks_and_bids.items():
                best_ask, best_bid = ask_and_bid
                row = pd.DataFrame([[connector_name, best_ask, best_bid]])
                order_books_df = pd.concat([order_books_df, row], ignore_index=True)

            if not order_books_df.empty:
                order_books_df.columns = ["Connector", "Best ask", "Best bid"]
                custom_status.append(format_df_for_printout(df=order_books_df, table_format="psql"))

            custom_status.append("\nBest arbitrage:")

            if self.best_arbitrage:
                best_ask_connector_name, best_ask_price, best_bid_connector_name, best_bid_price = self.best_arbitrage
                delta_bps = (best_bid_price - best_ask_price) / best_bid_price * 10000
                custom_status.append(f"Short {best_bid_connector_name} at {best_bid_price} | Long {best_ask_connector_name} at {best_ask_price} | delta_bps:{delta_bps:.2f}")
            else:
                custom_status.append("None")

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_executor(self, active_executors: List[ExecutorInfo]) -> bool:
        if self.get_position_quote_amount() == 0 or len(active_executors) > 0:
            return False

        if self.last_terminated_executor_timestamp + self.config.cooldown_time_min * 60 > self.market_data_provider.time():
            self.logger().info("Cooldown not passed yet")
            return False

        return True

    def get_connector(self, connector_name: str) -> Optional[ConnectorBase]:
        return self.market_data_provider.get_connector(connector_name)

    def get_mid_price(self, connector_name):
        trading_pair, _ = self.config.connectors_and_pairs.get(connector_name)
        return self.market_data_provider.get_price_by_type(connector_name, trading_pair, PriceType.MidPrice)

    def get_position_quote_amount(self) -> Decimal:
        # If balance = 100 USDT with leverage 20x, the quote position should be 500
        return Decimal(self.config.total_amount_quote * self.config.leverage / 4)

    def get_best_ask(self, connector_name: str) -> Decimal:
        return self._get_best_ask_or_bid(connector_name, PriceType.BestAsk)

    def get_best_bid(self, connector_name: str) -> Decimal:
        return self._get_best_ask_or_bid(connector_name, PriceType.BestBid)

    def _get_best_ask_or_bid(self, connector_name: str, price_type: PriceType) -> Decimal:
        trading_pair, _ = self.config.connectors_and_pairs.get(connector_name)
        return self.market_data_provider.get_price_by_type(connector_name, trading_pair, price_type)

    def get_executor_config(self, connector_name: str, side: TradeType, ref_price: Decimal) -> PositionExecutorConfig:
        trading_pair, _ = self.config.connectors_and_pairs.get(connector_name)

        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=connector_name,
            trading_pair=trading_pair,
            side=side,
            entry_price=ref_price,
            amount=self.get_position_quote_amount() / ref_price,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage
        )

    def get_active_executors_by_side(self, connector_name: str) -> Tuple[List[ExecutorInfo], List[ExecutorInfo]]:
        active_executors = self.filter_executors(
            executors=self.get_all_executors(),
            filter_func=lambda e: e.connector_name == connector_name and e.is_active
        )

        active_sell_executors = [e for e in active_executors if e.side == TradeType.SELL]
        active_buy_executors = [e for e in active_executors if e.side == TradeType.BUY]

        return active_sell_executors, active_buy_executors

    def get_unfilled_executors_by_side(self, connector_name: str) -> Tuple[List[ExecutorInfo], List[ExecutorInfo]]:
        active_sell_executors, active_buy_executors = self.get_active_executors_by_side(connector_name)

        unfilled_sell_executors = [e for e in active_sell_executors if not e.is_trading]
        unfilled_buy_executors = [e for e in active_buy_executors if not e.is_trading]

        return unfilled_sell_executors, unfilled_buy_executors

    def get_trading_executors_by_side(self, connector_name: str) -> Tuple[List[ExecutorInfo], List[ExecutorInfo]]:
        active_sell_executors, active_buy_executors = self.get_active_executors_by_side(connector_name)

        trading_sell_executors = [e for e in active_sell_executors if e.is_trading]
        trading_buy_executors = [e for e in active_buy_executors if e.is_trading]

        return trading_sell_executors, trading_buy_executors

    def get_trading_executors_on_side(self, connector_name: str, side: TradeType) -> List[ExecutorInfo]:
        trading_sell_executors, trading_buy_executors = self.get_trading_executors_by_side(connector_name)
        return trading_sell_executors if side == TradeType.SELL else trading_buy_executors

    def combine_active_executors_by_side(self) -> Tuple[List[ExecutorInfo], List[ExecutorInfo]]:
        all_sell_executors = []
        all_buy_executors = []

        for connector_name, _ in self.config.connectors_and_pairs.items():
            sell_executors, buy_executors = self.get_active_executors_by_side(connector_name)
            all_sell_executors.extend(sell_executors)
            all_buy_executors.extend(buy_executors)

        return all_sell_executors, all_buy_executors

    #
    # Custom functions specific to this controller
    #

    def update_best_asks_and_bids(self):
        for connector_name, (_, adjustment) in self.config.connectors_and_pairs.items():
            best_ask = self.get_best_ask(connector_name) + adjustment
            best_bid = self.get_best_bid(connector_name) + adjustment
            self.best_asks_and_bids[connector_name] = best_ask, best_bid

    def find_best_arbitrage(self) -> Optional[Tuple[str, Decimal, str, Decimal]]:
        best_spread = 0
        best_ask_connector_name: Optional[str] = None
        best_bid_connector_name: Optional[str] = None
        best_ask_price = 0
        best_bid_price = 0

        for ask_connector_name, _ in self.best_asks_and_bids.items():
            for bid_connector_name, _ in self.best_asks_and_bids.items():
                if ask_connector_name != bid_connector_name:
                    best_ask, _ = self.best_asks_and_bids.get(ask_connector_name)
                    _, best_bid = self.best_asks_and_bids.get(bid_connector_name)
                    spread = best_bid - best_ask

                    if spread > best_spread:
                        best_spread = spread
                        best_ask_connector_name = ask_connector_name
                        best_bid_connector_name = bid_connector_name
                        best_ask_price = best_ask
                        best_bid_price = best_bid

        if best_ask_connector_name and best_bid_connector_name:
            return best_ask_connector_name, best_ask_price, best_bid_connector_name, best_bid_price
        else:
            return None  # No arbitrage opportunity found

    def get_mid_price_custom(self, connector_name: str) -> Decimal:
        best_ask, best_bid = self.best_asks_and_bids.get(connector_name)
        return (best_ask + best_bid) * Decimal(0.5)
