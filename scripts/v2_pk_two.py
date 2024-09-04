from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
import pandas_ta as ta  # noqa: F401

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from scripts.utility.my_utils import calculate_delta_bps, has_order_expired
from scripts.v2_pk_two_config import PkTwoConfig


class PkTwo(StrategyV2Base):
    """
    TODO: describe
    """

    account_config_set = False
    oracle_price_before_position_creation: Decimal = 0
    oracle_price_df = pd.DataFrame(columns=["timestamp", "price", "price_delta_bps"])

    @classmethod
    def init_markets(cls, config: PkTwoConfig):
        cls.markets = {
            config.binance_exchange: {config.binance_pair},
            config.kucoin_exchange: {config.kucoin_pair}
        }

    def __init__(self, connectors: Dict[str, ConnectorBase], config: PkTwoConfig):
        super().__init__(connectors, config)
        self.config = config
        self.binance_connector: ConnectorBase = self.connectors[config.binance_exchange]
        self.kucoin_connector: ConnectorBase = self.connectors[config.kucoin_exchange]

    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        :param clock: Clock to use.
        :param timestamp: Current time.
        """
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        if not self.account_config_set:
            for connector_name, connector in self.connectors.items():
                if self.is_perpetual(connector_name):
                    connector.set_position_mode(self.config.position_mode)
                    for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                        self.init_leverage(connector_name, trading_pair)
            self.account_config_set = True

    def init_leverage(self, connector_name: str, trading_pair: str):
        leverage = self.get_connector_leverage(connector_name)
        self.connectors[connector_name].set_leverage(trading_pair, leverage)

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        create_actions = []
        signal = self.get_signal()

        if signal is None:
            return []

        if signal < 0:
            # Place a limit buy order right above best bid price - opposite of original idea
            position_config = self.get_executor_config(self.config.kucoin_exchange, TradeType.BUY, signal)

            if position_config is not None:
                create_actions.append(CreateExecutorAction(executor_config=position_config))

        elif signal > 0:
            # Place a limit sell order right below best ask price - opposite of original idea
            position_config = self.get_executor_config(self.config.kucoin_exchange, TradeType.SELL, signal)

            if position_config is not None:
                create_actions.append(CreateExecutorAction(executor_config=position_config))

        return create_actions

    def get_executor_config(self, connector_name: str, side: TradeType, price_delta_bps: Decimal) -> Optional[PositionExecutorConfig]:
        active_executors = self.get_active_executors(connector_name)

        # Only create a new position if none is active on that connector
        if len(active_executors) > 0:
            return None

        best_ask = self.get_best_ask(connector_name)
        best_bid = self.get_best_bid(connector_name)

        ref_price: Decimal = (
            # If side == BUY, place a limit buy order slightly below best bid
            best_bid * Decimal(1 - self.config.delta_with_best_bid_or_ask_bps / 10000) if side == TradeType.BUY else
            # If side == SELL, place a limit sell order slightly above best ask
            best_ask * Decimal(1 + self.config.delta_with_best_bid_or_ask_bps / 10000)
        )

        quote_amount = self.get_position_quote_amount(connector_name)

        if quote_amount == 0:
            return None

        self.oracle_price_before_position_creation = self.get_latest_oracle_price() * Decimal(1 - price_delta_bps / 10000)

        self.logger().info(
            f"NEW POSITION. Side: {side}, latest_oracle_price: {self.get_latest_oracle_price()}, before that: {self.oracle_price_before_position_creation}")
        self.logger().info(
            f"best_ask: {best_ask}, best_bid: {best_bid}, ref_price: {ref_price}, delta_with_best_bid_or_ask_bps: {self.config.delta_with_best_bid_or_ask_bps}")

        return PositionExecutorConfig(
            timestamp=self.current_timestamp,
            connector_name=connector_name,
            trading_pair=self.get_connector_trading_pair(connector_name),
            side=side,
            entry_price=ref_price,
            amount=quote_amount / ref_price,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.get_connector_leverage(connector_name)
        )

    def get_signal(self) -> Optional[Decimal]:
        new_price = self.binance_connector.get_mid_price(self.config.binance_pair)
        price_delta_bps: Decimal = Decimal(0)

        if len(self.oracle_price_df) > 0:
            previous_price = self.get_latest_oracle_price()
            price_delta_bps = calculate_delta_bps(new_price, previous_price)

        self.logger().info(f"price_delta_bps: {price_delta_bps}")

        new_row = pd.DataFrame({"timestamp": [self.current_timestamp], "price": [new_price], "price_delta_bps": [price_delta_bps]})
        self.oracle_price_df = pd.concat([self.oracle_price_df, new_row]).tail(self.config.oracle_price_history_length)

        if (
                price_delta_bps.is_infinite() or
                abs(price_delta_bps) < self.config.price_delta_threshold_on_leading_exchange_bps
        ):
            return None

        return price_delta_bps

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        stop_actions = []

        for executor in self.get_active_executors(self.config.kucoin_exchange, True):
            if has_order_expired(executor, self.config.unfilled_order_time_limit, self.current_timestamp):
                stop_actions.append(StopExecutorAction(executor_id=executor.id))

        return stop_actions

    def has_trend_reversed(self, executor: ExecutorInfo) -> bool:
        has_reversed = (
            (executor.side == TradeType.BUY and self.get_latest_oracle_price() < self.oracle_price_before_position_creation) or
            (executor.side == TradeType.SELL and self.get_latest_oracle_price() > self.oracle_price_before_position_creation)
        )

        if has_reversed:
            self.logger().info("Trend has reversed!")

        return has_reversed

    def get_latest_oracle_price(self) -> Decimal:
        return Decimal(0) if len(self.oracle_price_df) == 0 else self.oracle_price_df["price"].iloc[-1]

    def get_active_executors(self, connector_name: str, is_non_trading_only: bool = False) -> List[ExecutorInfo]:
        filter_func = (
            lambda e: e.connector_name == connector_name and e.is_active and not e.is_trading
        ) if is_non_trading_only else (
            lambda e: e.connector_name == connector_name and e.is_active
        )

        active_executors = self.filter_executors(
            executors=self.get_all_executors(),
            filter_func=filter_func
        )

        return active_executors

    def get_position_quote_amount(self, connector_name: str) -> Decimal:
        connector: ConnectorBase = self.connectors[connector_name]
        trading_pair: str = self.get_connector_trading_pair(connector_name)
        _, quote_currency = split_hb_trading_pair(trading_pair)
        available_quote_balance = connector.get_available_balance(quote_currency)

        if available_quote_balance < 1:
            return Decimal(0)

        leverage: int = connector.get_leverage(trading_pair)

        # If balance = 100 USDT with leverage 20x, the quote position should be 1000
        return Decimal(available_quote_balance * leverage / 2)

    def get_connector_leverage(self, connector_name: str) -> int:
        if connector_name == "kucoin_perpetual":
            return self.config.kucoin_leverage
        return 1

    def get_connector_trading_pair(self, connector_name: str) -> str:
        trading_pairs = self.markets[connector_name]
        return next(iter(trading_pairs))

    def get_best_ask(self, connector_name: str) -> Decimal:
        return self.get_best_ask_or_bid(connector_name, True)

    def get_best_bid(self, connector_name: str) -> Decimal:
        return self.get_best_ask_or_bid(connector_name, False)

    def get_best_ask_or_bid(self, connector_name: str, is_buy: bool) -> Decimal:
        connector: ConnectorBase = self.connectors[connector_name]
        trading_pair: str = self.get_connector_trading_pair(connector_name)
        return connector.get_price(trading_pair, is_buy)

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    def format_status(self) -> str:
        original_info = super().format_status()
        columns_to_show = ["type", "side", "status", "net_pnl_pct", "net_pnl_quote", "cum_fees_quote",
                           "filled_amount_quote", "is_trading", "close_type", "age"]
        extra_info = []

        # Initialize global performance metrics
        global_realized_pnl_quote = Decimal(0)
        global_unrealized_pnl_quote = Decimal(0)
        global_volume_traded = Decimal(0)
        global_close_type_counts = {}

        # Process each controller
        for controller_id, controller in self.controllers.items():
            extra_info.append(f"\n\nController: {controller_id}")
            # Append controller market data metrics
            extra_info.extend(controller.to_format_status())
            executors_list = self.get_executors_by_controller(controller_id)
            if len(executors_list) == 0:
                extra_info.append("No executors found.")
            else:
                # In memory executors info
                executors_df = self.executors_info_to_df(executors_list)
                executors_df["age"] = self.current_timestamp - executors_df["timestamp"]
                extra_info.extend([format_df_for_printout(executors_df[columns_to_show], table_format="psql")])

            # Generate performance report for each controller
            performance_report = self.executor_orchestrator.generate_performance_report(controller_id)

            # Append performance metrics
            controller_performance_info = [
                f"Realized PNL (Quote): {performance_report.realized_pnl_quote:.2f} | Unrealized PNL (Quote): {performance_report.unrealized_pnl_quote:.2f}"
                f"--> Global PNL (Quote): {performance_report.global_pnl_quote:.2f} | Global PNL (%): {performance_report.global_pnl_pct:.2f}%",
                f"Total Volume Traded: {performance_report.volume_traded:.2f}"
            ]

            # Append close type counts
            if performance_report.close_type_counts:
                controller_performance_info.append("Close Types Count:")
                for close_type, count in performance_report.close_type_counts.items():
                    controller_performance_info.append(f"  {close_type}: {count}")
            extra_info.extend(controller_performance_info)

            # Aggregate global metrics and close type counts
            global_realized_pnl_quote += performance_report.realized_pnl_quote
            global_unrealized_pnl_quote += performance_report.unrealized_pnl_quote
            global_volume_traded += performance_report.volume_traded
            for close_type, value in performance_report.close_type_counts.items():
                global_close_type_counts[close_type] = global_close_type_counts.get(close_type, 0) + value

        main_executors_list = self.get_executors_by_controller("main")
        if len(main_executors_list) > 0:
            extra_info.append("\n\nMain Controller Executors:")
            main_executors_df = self.executors_info_to_df(main_executors_list)
            main_executors_df["age"] = self.current_timestamp - main_executors_df["timestamp"]
            extra_info.extend([format_df_for_printout(main_executors_df[columns_to_show], table_format="psql")])
            main_performance_report = self.executor_orchestrator.generate_performance_report("main")
            # Aggregate global metrics and close type counts
            global_realized_pnl_quote += main_performance_report.realized_pnl_quote
            global_unrealized_pnl_quote += main_performance_report.unrealized_pnl_quote
            global_volume_traded += main_performance_report.volume_traded
            for close_type, value in main_performance_report.close_type_counts.items():
                global_close_type_counts[close_type] = global_close_type_counts.get(close_type, 0) + value

        # Calculate and append global performance metrics
        global_pnl_quote = global_realized_pnl_quote + global_unrealized_pnl_quote
        global_pnl_pct = (global_pnl_quote / global_volume_traded) * 100 if global_volume_traded != 0 else Decimal(0)

        global_performance_summary = [
            "\n\nGlobal Performance Summary:",
            f"Global PNL (Quote): {global_pnl_quote:.2f} | Global PNL (%): {global_pnl_pct:.2f}% | Total Volume Traded (Global): {global_volume_traded:.2f}"
        ]

        # Append global close type counts
        if global_close_type_counts:
            global_performance_summary.append("Global Close Types Count:")
            for close_type, count in global_close_type_counts.items():
                global_performance_summary.append(f"  {close_type}: {count}")

        extra_info.extend(global_performance_summary)

        # Oracle Price DF
        oracle_price_summary = ["\n\nOracle Price DF"]
        oracle_price_summary.extend(self.get_formatted_oracle_price_df().to_string(index=False).split("\n"))
        extra_info.extend(oracle_price_summary)

        # Combine original and extra information
        format_status = f"{original_info}\n\n" + "\n".join(extra_info)
        return format_status

    def get_formatted_oracle_price_df(self):
        # Create a copy of the DataFrame to avoid modifying the original
        formatted_df = self.oracle_price_df.copy()

        formatted_df['timestamp'] = formatted_df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).isoformat())
        formatted_df['price_delta_bps'] = formatted_df['price_delta_bps'].apply(lambda x: f"{x:.2f}")

        return formatted_df.iloc[::-1].reset_index(drop=True)
