import os
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionMode, TradeType, PositionAction, PriceType
from hummingbot.core.event.events import SellOrderCreatedEvent, BuyOrderCreatedEvent, OrderFilledEvent
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase, StrategyV2Base
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig, PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from scripts.pk.tracked_order_details import TrackedOrderDetails


class PkStrategy:
    def __init__(self, logger: HummingbotLogger):
        self.logger = logger

        self.is_a_sell_order_being_created = False
        self.is_a_buy_order_being_created = False

        self.tracked_orders: List[TrackedOrderDetails] = []
        self.processed_data: Dict = {}

    def can_create_order(self, side: TradeType) -> bool:
        if self.get_position_quote_amount(side) == 0 or len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL and self.is_a_sell_order_being_created:
            self.logger.info(f"Another SELL order is being created, avoiding a duplicate")
            return False

        if side == TradeType.BUY and self.is_a_buy_order_being_created:
            self.logger.info(f"Another BUY order is being created, avoiding a duplicate")
            return False

        last_canceled_order = self.find_last_cancelled_tracked_order()

        if not last_canceled_order:
            return True

        if last_canceled_order.cancelled_at + self.config.cooldown_time_min * 60 > self.market_data_provider.time():
            self.logger.info("Cooldown not passed yet")
            return False

        return True

    def did_create_sell_order(self, created_event: SellOrderCreatedEvent):
        position = created_event.position

        if not position or position != PositionAction.OPEN.value:
            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == created_event.order_id:
                tracked_order.exchange_order_id = created_event.exchange_order_id,
                tracked_order.created_at = created_event.creation_timestamp
                break

        # TODO: remove
        self.logger.info(f"did_create_sell_order | self.tracked_orders: {self.tracked_orders}")

    def did_create_buy_order(self, created_event: BuyOrderCreatedEvent):
        position = created_event.position

        if not position or position != PositionAction.OPEN.value:
            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == created_event.order_id:
                tracked_order.exchange_order_id = created_event.exchange_order_id,
                tracked_order.created_at = created_event.creation_timestamp
                break

        # TODO: remove
        self.logger.info(f"did_create_buy_order | self.tracked_orders: {self.tracked_orders}")

    def did_fill_order(self, filled_event: OrderFilledEvent):
        position = filled_event.position

        if not position or position != PositionAction.OPEN.value:
            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == filled_event.order_id:
                tracked_order.filled_at = filled_event.timestamp
                break

        # TODO: remove
        self.logger.info(f"did_fill_order | self.tracked_orders: {self.tracked_orders}")

    def get_mid_price(self):
        return self.market_data_provider.get_price_by_type(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)

    def get_position_quote_amount(self, side: TradeType) -> Decimal:
        amount_quote = Decimal(self.config.total_amount_quote)

        # If amount_quote = 100 USDT with leverage 20x, the quote position should be 500
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
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=side,
            entry_price=ref_price,
            amount=self.get_position_quote_amount(side) / ref_price,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage
        )

    def find_tracked_order_of_id(self, order_id: str) -> Optional[TrackedOrderDetails]:
        orders_of_that_id = [order for order in self.tracked_orders if order.order_id == order_id]
        return None if len(orders_of_that_id) == 0 else orders_of_that_id[0]

    def find_last_cancelled_tracked_order(self) -> Optional[TrackedOrderDetails]:
        cancelled_orders = [order for order in self.tracked_orders if order.cancelled_at]

        if len(cancelled_orders) == 0:
            return None

        return max(cancelled_orders, key=lambda order: order.cancelled_at)

    def get_active_tracked_orders(self) -> List[TrackedOrderDetails]:
        return [order for order in self.tracked_orders if order.created_at and not order.cancelled_at]

    def get_active_tracked_orders_by_side(self) -> Tuple[List[TrackedOrderDetails], List[TrackedOrderDetails]]:
        active_orders = self.get_active_tracked_orders()
        active_sell_orders = [order for order in active_orders if order.side == TradeType.SELL]
        active_buy_orders = [order for order in active_orders if order.side == TradeType.BUY]
        return active_sell_orders, active_buy_orders

    def get_unfilled_tracked_orders_by_side(self) -> Tuple[List[TrackedOrderDetails], List[TrackedOrderDetails]]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()
        unfilled_sell_orders = [order for order in active_sell_orders if not order.filled_at]
        unfilled_buy_orders = [order for order in active_buy_orders if not order.filled_at]
        return unfilled_sell_orders, unfilled_buy_orders

    def create_order(self, executor_config: PositionExecutorConfig):
        connector_name = executor_config.connector_name
        trading_pair = executor_config.trading_pair
        amount = executor_config.amount
        entry_price = executor_config.entry_price

        if executor_config.side == TradeType.SELL:
            self.is_a_sell_order_being_created = True

            order_id = self.sell(connector_name, trading_pair, amount, OrderType.MARKET, entry_price)

            self.tracked_orders.append(TrackedOrderDetails(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.SELL,
                order_id=order_id,
                position=PositionAction.OPEN.value,
                amount=amount
            ))

            self.is_a_sell_order_being_created = False

        else:
            self.is_a_buy_order_being_created = True

            order_id = self.buy(connector_name, trading_pair, amount, OrderType.MARKET, entry_price)

            self.tracked_orders.append(TrackedOrderDetails(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.BUY,
                order_id=order_id,
                position=PositionAction.OPEN.value,
                amount = amount
            ))

            self.is_a_buy_order_being_created = False

        # TODO: remove
        self.logger().info(f"create_order | self.tracked_orders: {self.tracked_orders}")

    # `self.cancel()` only works for unfilled orders
    def close_tracked_order(self, tracked_order: TrackedOrderDetails, current_price: Decimal):
        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair
        amount = tracked_order.amount

        # TODO: remove
        self.logger().info(f"cancel_order | tracked_order: {tracked_order}")

        if tracked_order.side == TradeType.SELL:
            self.buy(
                connector_name,
                trading_pair,
                amount,
                OrderType.MARKET,
                current_price,
                PositionAction.CLOSE
            )
        else:
            self.sell(
                connector_name,
                trading_pair,
                amount,
                OrderType.MARKET,
                current_price,
                PositionAction.CLOSE
            )

        for order in self.tracked_orders:
            if order.order_id == tracked_order.order_id:
                order.cancelled_at = self.market_data_provider.time()
                break

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
        # TODO: remove
        self.logger().info(f"is_high_volatility() | latest_bbb: {self.get_latest_bbb()}")

        return self.get_latest_bbb() > self.config.high_volatility_threshold

    def is_still_trending_up(self) -> bool:
        return self.get_latest_normalized_bbp() > -0.2

    def is_still_trending_down(self) -> bool:
        return self.get_latest_normalized_bbp() < 0.2

    def get_latest_normalized_rsi(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["features"]["normalized_rsi"]
        return Decimal(rsi_series.iloc[-1])
