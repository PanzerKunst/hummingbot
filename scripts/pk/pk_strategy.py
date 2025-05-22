from collections import Counter
from datetime import datetime
from decimal import Decimal
from typing import Dict, Final, List, Tuple

import requests
from requests import RequestException

from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.core.event.events import (
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCreatedEvent,
)
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.pk.close_order import CloseOrder
from scripts.pk.log_level import LogLevel
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import (
    bucket_minute,
    has_current_price_reached_stop_loss,
    has_filled_order_reached_time_limit,
    has_unfilled_order_expired,
)
from scripts.pk.take_profit_order import TakeProfitOrder
from scripts.pk.tracked_order import TrackedOrder
from scripts.thunderfury_config import ExcaliburConfig

TELEGRAM_BOT_TOKEN: Final[str] = "REPLACE_WITH_YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID: Final[str] = "REPLACE_WITH_YOUR_CHAT_ID"
TELEGRAM_API_URL: Final[str] = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


class PkStrategy(StrategyV2Base):
    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)
        self.config = config

        self.is_a_sell_order_being_created = False
        self.is_a_buy_order_being_created = False

        self.tracked_orders: List[TrackedOrder] = []
        self.close_orders: List[CloseOrder] = []
        self.cancel_orders: List[CloseOrder] = []
        self.take_profit_orders: List[TakeProfitOrder] = []

    def get_mid_price(self) -> Decimal:
        connector_name = self.config.connector_name
        trading_pair = self.config.trading_pair

        return self.market_data_provider.get_price_by_type(connector_name, trading_pair, PriceType.MidPrice)

    def get_best_ask(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestAsk)

    def get_best_bid(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestBid)

    def _get_best_ask_or_bid(self, price_type: PriceType) -> Decimal:
        connector_name = self.config.connector_name
        trading_pair = self.config.trading_pair

        return self.market_data_provider.get_price_by_type(connector_name, trading_pair, price_type)

    def get_executor_config(self, side: TradeType, entry_price: Decimal, amount_quote: Decimal) -> PositionExecutorConfig:
        connector_name = self.config.connector_name
        trading_pair = self.config.trading_pair
        leverage = self.config.leverage

        amount: Decimal = amount_quote / entry_price

        return PositionExecutorConfig(
            timestamp=self.get_market_data_provider_time(),
            connector_name=connector_name,
            trading_pair=trading_pair,
            side=side,
            entry_price=entry_price,
            amount=amount,
            leverage=leverage,
            type = "position_executor"
        )

    def find_tracked_order_of_id(self, order_id: str) -> TrackedOrder | None:
        orders_of_that_id: List[TrackedOrder] = [order for order in self.tracked_orders if order.order_id == order_id]
        return None if len(orders_of_that_id) == 0 else orders_of_that_id[0]

    def find_tp_order_of_id(self, order_id: str) -> TakeProfitOrder | None:
        orders_of_that_id: List[TakeProfitOrder] = [order for order in self.take_profit_orders if order.order_id == order_id]
        return None if len(orders_of_that_id) == 0 else orders_of_that_id[0]

    def find_close_order_of_id(self, order_id: str) -> CloseOrder | None:
        orders_of_that_id: List[CloseOrder] = [order for order in self.close_orders if order.order_id == order_id]
        return None if len(orders_of_that_id) == 0 else orders_of_that_id[0]

    def find_cancel_order_of_id(self, order_id: str) -> CloseOrder | None:
        orders_of_that_id: List[CloseOrder] = [order for order in self.cancel_orders if order.order_id == order_id]
        return None if len(orders_of_that_id) == 0 else orders_of_that_id[0]

    def find_last_terminated_filled_order(self, side: TradeType, tag: str | None = None) -> TrackedOrder | None:
        terminated_filled_orders = [order for order in self.tracked_orders if (
            order.side == side and
            order.last_filled_at and
            order.terminated_at
        )]

        if tag:
            terminated_filled_orders = [order for order in terminated_filled_orders if order.tag == tag]

        if len(terminated_filled_orders) == 0:
            return None

        return max(terminated_filled_orders, key=lambda order: order.terminated_at)

    def get_active_tracked_orders(self, tag: str | None = None) -> List[TrackedOrder]:
        active_tracked_orders = [order for order in self.tracked_orders if (
            order.canceling_at is None and
            order.closing_at is None and
            order.terminated_at is None
        )]

        if tag:
            active_tracked_orders = [order for order in active_tracked_orders if order.tag == tag]

        return active_tracked_orders

    def get_closing_tracked_orders(self, tag: str | None = None) -> List[TrackedOrder]:
        closing_tracked_orders = [order for order in self.tracked_orders if (
            order.closing_at and
            order.terminated_at is None
        )]

        if tag:
            closing_tracked_orders = [order for order in closing_tracked_orders if order.tag == tag]

        return closing_tracked_orders

    def get_active_tracked_orders_by_side(self, tag: str | None = None) -> Tuple[List[TrackedOrder], List[TrackedOrder]]:
        active_orders = self.get_active_tracked_orders(tag)
        active_sell_orders = [order for order in active_orders if order.side == TradeType.SELL]
        active_buy_orders = [order for order in active_orders if order.side == TradeType.BUY]
        return active_sell_orders, active_buy_orders

    def get_unfilled_tracked_orders_by_side(self, tag: str | None = None) -> Tuple[List[TrackedOrder], List[TrackedOrder]]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(tag)
        unfilled_sell_orders = [order for order in active_sell_orders if order.last_filled_at is None]
        unfilled_buy_orders = [order for order in active_buy_orders if order.last_filled_at is None]
        return unfilled_sell_orders, unfilled_buy_orders

    def get_filled_tracked_orders_by_side(self, tag: str | None = None) -> Tuple[List[TrackedOrder], List[TrackedOrder]]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(tag)
        filled_sell_orders = [order for order in active_sell_orders if order.last_filled_at is not None]
        filled_buy_orders = [order for order in active_buy_orders if order.last_filled_at]
        return filled_sell_orders, filled_buy_orders

    def get_all_unfilled_tp_orders(self) -> List[TakeProfitOrder]:
        return [order for order in self.take_profit_orders if order.last_filled_at is None]

    def get_unfilled_tp_orders(self, tracked_order: TrackedOrder) -> List[TakeProfitOrder]:
        return [order for order in self.get_all_unfilled_tp_orders() if order.tracked_order.order_id == tracked_order.order_id]

    def get_latest_filled_tp_order(self) -> TakeProfitOrder | None:
        filled_tp_orders: List[TakeProfitOrder] = [
            tp for tp in self.take_profit_orders
            if tp.last_filled_at is not None
        ]

        if len(filled_tp_orders) == 0:
            return None

        return max(filled_tp_orders, key=lambda tp: tp.last_filled_at)

    def create_order(self, side: TradeType, entry_price: Decimal, triple_barrier: TripleBarrier, amount_quote: Decimal, tag: str):
        executor_config = self.get_executor_config(side, entry_price, amount_quote)
        self.create_individual_order(executor_config, triple_barrier, tag)

    # async def create_twap_market_orders(self, side: TradeType, entry_price: Decimal, triple_barrier: TripleBarrier, amount_quote: Decimal, tag: str):
    #     executor_config = self.get_executor_config(side, entry_price, amount_quote, True)
    #
    #     for _ in range(self.config.market_order_twap_count):
    #         is_an_order_being_created: bool = self.is_a_sell_order_being_created if executor_config.side == TradeType.SELL else self.is_a_buy_order_being_created
    #
    #         if is_an_order_being_created:
    #             self.log_error("Cannot create another individual order, as one is being created")
    #         else:
    #             self.create_individual_order(executor_config, triple_barrier, tag)
    #             await asyncio.sleep(self.config.market_order_twap_interval)

    def create_individual_order(self, executor_config: PositionExecutorConfig, triple_barrier: TripleBarrier, tag: str):
        connector_name = executor_config.connector_name
        trading_pair = executor_config.trading_pair
        amount = executor_config.amount
        entry_price = executor_config.entry_price
        open_order_type = triple_barrier.open_order_type

        self.is_a_sell_order_being_created = True

        if executor_config.side == TradeType.SELL:
            order_id = self.sell(connector_name, trading_pair, amount, open_order_type, entry_price)

            self.tracked_orders.append(TrackedOrder(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.SELL,
                order_id=order_id,
                amount=amount,
                entry_price=entry_price,
                triple_barrier=triple_barrier,
                tag=tag,
                created_at=self.get_market_data_provider_time()  # Because some exchanges such as gate_io trigger the `did_create_xxx_order` event after 1s
            ))

        else:
            order_id = self.buy(connector_name, trading_pair, amount, open_order_type, entry_price)

            self.tracked_orders.append(TrackedOrder(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.BUY,
                order_id=order_id,
                amount=amount,
                entry_price=entry_price,
                triple_barrier=triple_barrier,
                tag=tag,
                created_at=self.get_market_data_provider_time()
            ))

        self.is_a_buy_order_being_created = False

        created_order: TrackedOrder = self.tracked_orders[-1]
        self.logger().info(f"create_individual_order(): {executor_config.side} {created_order.tag} {created_order.amount} at {created_order.entry_price} | {created_order.order_id}")

    def create_tp_order(self, tracked_order: TrackedOrder, amount: Decimal, entry_price: Decimal):
        side: TradeType = TradeType.SELL if tracked_order.side == TradeType.BUY else TradeType.BUY
        trading_pair: str = tracked_order.trading_pair

        executor_config = self.get_executor_config(side, entry_price, amount)
        connector_name: str = executor_config.connector_name

        order_id = (
            self.sell(connector_name, trading_pair, amount, OrderType.LIMIT, entry_price, PositionAction.CLOSE) if side == TradeType.SELL else
            self.buy(connector_name, trading_pair, amount, OrderType.LIMIT, entry_price, PositionAction.CLOSE)
        )

        self.take_profit_orders.append(TakeProfitOrder(
            order_id=order_id,
            tracked_order=tracked_order,
            amount=amount,
            entry_price=entry_price,
            created_at=self.get_market_data_provider_time()
        ))

        created_order: TakeProfitOrder = self.take_profit_orders[-1]
        self.logger().info(f"create_tp_order(): {side} {created_order.amount} at {created_order.entry_price} | {created_order.order_id} for TrackedOrder {created_order.tracked_order.tag} {created_order.tracked_order.order_id}")

    def update_order_to_canceling(self, order_id: str, timestamp: float):
        tracked_order: TrackedOrder = self.find_tracked_order_of_id(order_id)

        if tracked_order:
            tracked_order.canceling_at = timestamp

    def update_order_to_closing(self, order_id: str, timestamp: float, close_type: CloseType | None = None):
        tracked_order: TrackedOrder = self.find_tracked_order_of_id(order_id)

        if tracked_order:
            tracked_order.closing_at = timestamp

            if close_type:
                tracked_order.close_type = close_type

    def update_order_to_terminated(self, order_id: str, timestamp: float, close_type: CloseType | None = None):
        tracked_order: TrackedOrder = self.find_tracked_order_of_id(order_id)

        if tracked_order:
            tracked_order.terminated_at = timestamp

            if close_type:
                tracked_order.close_type = close_type

            if tracked_order.closing_at is None:
                tracked_order.closing_at = timestamp

    def close_filled_order(self, filled_order: TrackedOrder, market_or_limit: OrderType, close_type: CloseType):
        self.cancel_take_profit_for_order(filled_order)

        connector_name = filled_order.connector_name
        trading_pair = filled_order.trading_pair
        amount_to_close = filled_order.filled_amount - filled_order.filled_tp_amount

        close_price_sell = self.get_best_bid() * Decimal(1 - self.config.limit_take_profit_price_delta_bps / 10000)
        close_price_buy = self.get_best_ask() * Decimal(1 + self.config.limit_take_profit_price_delta_bps / 10000)

        self.logger().info(f"close_filled_order() | amount_to_close:{amount_to_close} | close_price:{close_price_sell if filled_order.side == TradeType.SELL else close_price_buy}")

        order_id = (
            self.buy(connector_name, trading_pair, amount_to_close, market_or_limit, close_price_sell, PositionAction.CLOSE) if filled_order.side == TradeType.SELL else
            self.sell(connector_name, trading_pair, amount_to_close, market_or_limit, close_price_buy, PositionAction.CLOSE)
        )

        self.close_orders.append(CloseOrder(
            order_id=order_id,
            tracked_order=filled_order,
            created_at=self.get_market_data_provider_time()
        ))

        self.update_order_to_closing(filled_order.order_id, self.get_market_data_provider_time(), close_type)

    def close_filled_orders(self, filled_orders: List[TrackedOrder], market_or_limit: OrderType, close_type: CloseType):
        if len(filled_orders) == 0:
            return

        for filled_order in filled_orders:
            self.close_filled_order(filled_order, market_or_limit, close_type)

    def cancel_unfilled_tracked_order(self, tracked_order: TrackedOrder):
        self.cancel_take_profit_for_order(tracked_order)

        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair
        order_id = tracked_order.order_id

        if tracked_order.exchange_order_id is None:
            self.logger().info("cancel_unfilled_tracked_order() > Not canceling due to empty exchange_order_id")
            return

        self.logger().info(f"cancel_unfilled_tracked_order(): {tracked_order.side} {tracked_order.tag} {tracked_order.amount} | {order_id}")
        self.cancel(connector_name, trading_pair, order_id)

        self.cancel_orders.append(CloseOrder(
            order_id=order_id,
            tracked_order=tracked_order,
            created_at=self.get_market_data_provider_time()
        ))

        self.update_order_to_canceling(order_id, self.get_market_data_provider_time())

    def cancel_take_profit_for_order(self, tracked_order: TrackedOrder):
        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair

        for tp_order in self.get_unfilled_tp_orders(tracked_order):
            order_id = tp_order.order_id
            tracked_order: TrackedOrder = tp_order.tracked_order
            side = TradeType.SELL if tracked_order.side == TradeType.BUY else TradeType.BUY

            self.logger().info(f"cancel_take_profit_for_order(): {side} {tp_order.amount} | {order_id} for TrackedOrder {tracked_order.tag} {tracked_order.order_id}")

            self.cancel(connector_name, trading_pair, order_id)
            self.take_profit_orders.remove(tp_order)

    def did_create_sell_order(self, created_event: SellOrderCreatedEvent):
        tracked_order: TrackedOrder = self.find_tracked_order_of_id(created_event.order_id)

        if tracked_order and created_event.position == PositionAction.OPEN.value:
            tracked_order.amount = created_event.amount
            tracked_order.exchange_order_id = created_event.exchange_order_id
            self.logger().info(f"did_create_sell_order() | tracked_order.amount:{tracked_order.amount} | exchange_order_id:{tracked_order.exchange_order_id}")
            return

        tp_order: TakeProfitOrder = self.find_tp_order_of_id(created_event.order_id)

        if tp_order:
            tp_order.amount = created_event.amount
            tp_order.exchange_order_id = created_event.exchange_order_id
            self.logger().info(f"did_create_sell_order() | tp_order.amount:{tp_order.amount} | exchange_order_id:{tp_order.exchange_order_id}")

    def did_create_buy_order(self, created_event: BuyOrderCreatedEvent):
        tracked_order: TrackedOrder = self.find_tracked_order_of_id(created_event.order_id)

        if tracked_order and created_event.position == PositionAction.OPEN.value:
            tracked_order.amount = created_event.amount
            tracked_order.exchange_order_id = created_event.exchange_order_id
            self.logger().info(f"did_create_buy_order() | tracked_order.amount:{tracked_order.amount} | exchange_order_id:{tracked_order.exchange_order_id}")
            return

        tp_order: TakeProfitOrder = self.find_tp_order_of_id(created_event.order_id)

        if tp_order:
            tp_order.amount = created_event.amount
            tp_order.exchange_order_id = created_event.exchange_order_id
            self.logger().info(f"did_create_buy_order() | tp_order.amount:{tp_order.amount} | exchange_order_id:{tp_order.exchange_order_id}")

    def did_fail_order(self, order_failed_event: MarketOrderFailureEvent):
        self.log_error(f"did_fail_order() | order_id:{order_failed_event.order_id} | order_type:{order_failed_event.order_type}")

        tracked_order: TrackedOrder = self.find_tracked_order_of_id(order_failed_event.order_id)

        if tracked_order:
            self.logger().info("did_fail_order() > Found it in self.tracked_orders")
            self.tracked_orders.remove(tracked_order)
            return

        close_order: CloseOrder = self.find_close_order_of_id(order_failed_event.order_id)

        if close_order:
            self.logger().info("did_fail_order() > Found it in self.close_orders")
            self.close_orders.remove(close_order)

            tracked_order: TrackedOrder = self.find_tracked_order_of_id(close_order.tracked_order.order_id)

            if tracked_order:
                self.logger().info("did_fail_order() > 'Unclosing/unterminating' the tracked order")
                tracked_order.closing_at = None
                tracked_order.terminated_at = None
                tracked_order.close_type = None

            return

        tp_order: TakeProfitOrder = self.find_tp_order_of_id(order_failed_event.order_id)

        if tp_order:
            self.logger().info("did_fail_order() > Found it in self.take_profit_limit_orders")
            self.take_profit_orders.remove(tp_order)

    def did_fill_order(self, filled_event: OrderFilledEvent):
        self.logger().info(f"did_fill_order() | timestamp:{filled_event.timestamp} | position:{filled_event.position} | amount:{filled_event.amount} at price:{filled_event.price}")

        tp_order: TakeProfitOrder = self.find_tp_order_of_id(filled_event.order_id)

        if tp_order:
            tracked_order: TrackedOrder = tp_order.tracked_order
            self.logger().info(f"did_fill_order() > Take Profit price reached for tracked order: {tracked_order.side} {tracked_order.tag} | {tracked_order.order_id}")

            tp_order.last_filled_at = filled_event.timestamp
            tp_order.last_filled_price = filled_event.price

            tracked_order: TrackedOrder = self.find_tracked_order_of_id(tp_order.tracked_order.order_id)

            if tracked_order:
                tracked_order.filled_tp_amount += filled_event.amount
                self.logger().info(f"did_fill_order() > tracked_order.filled_tp_amount increased to: {tracked_order.filled_tp_amount}")

                if tracked_order.filled_tp_amount == tracked_order.amount:
                    self.logger().info("did_fill_order() > tracked_order.filled_tp_amount == tracked_order.amount. Terminating it")
                    self.update_order_to_terminated(tracked_order.order_id, filled_event.timestamp, CloseType.TAKE_PROFIT)

            return

        tracked_order: TrackedOrder = self.find_tracked_order_of_id(filled_event.order_id)

        if tracked_order and filled_event.position == PositionAction.OPEN.value:
            self.logger().info(f"did_fill_order() > tracked_order.filled_amount before update: {tracked_order.filled_amount}")

            tracked_order.filled_amount += filled_event.amount
            tracked_order.last_filled_at = filled_event.timestamp
            tracked_order.last_filled_price = filled_event.price

            self.logger().info(f"did_fill_order() > tracked_order.filled_amount increased to: {tracked_order.filled_amount}")
            return

        close_order: CloseOrder = self.find_close_order_of_id(filled_event.order_id)

        if close_order:
            tracked_order = self.find_tracked_order_of_id(close_order.tracked_order.order_id)

            if tracked_order:
                tracked_order.closed_amount += filled_event.amount
                self.logger().info(f"did_fill_order() > tracked_order.closed_amount increased to: {tracked_order.closed_amount}")

                if tracked_order.closed_amount + tracked_order.filled_tp_amount == tracked_order.amount:
                    self.logger().info("did_fill_order() > tracked_order.closed_amount + tracked_order.filled_tp_amount == tracked_order.amount. Terminating it")
                    self.update_order_to_terminated(tracked_order.order_id, filled_event.timestamp)

    def did_cancel_order(self, cancelled_event: OrderCancelledEvent):
        self.logger().info(f"did_cancel_order() | timestamp:{cancelled_event.timestamp} | order_id:{cancelled_event.order_id}")

        cancel_order: CloseOrder = self.find_cancel_order_of_id(cancelled_event.order_id)

        if cancel_order:
            tracked_order = self.find_tracked_order_of_id(cancel_order.tracked_order.order_id)

            if tracked_order:
                self.update_order_to_terminated(tracked_order.order_id, cancelled_event.timestamp)

    def can_create_order(self, side: TradeType, tag: str, cooldown_time_min: int) -> bool:
        if side == TradeType.SELL and self.is_a_sell_order_being_created:
            self.log_error("Another SELL order is being created, avoiding a duplicate")
            return False

        if side == TradeType.BUY and self.is_a_buy_order_being_created:
            self.log_error("Another BUY order is being created, avoiding a duplicate")
            return False

        last_terminated_filled_order = self.find_last_terminated_filled_order(side, tag)

        if last_terminated_filled_order is None:
            return True

        if last_terminated_filled_order.terminated_at + cooldown_time_min * 60 > self.get_market_data_provider_time():
            self.logger().info(f"Cooldown not passed yet for {side}")
            return False

        return True

    def check_orders(self, max_created_orders_per_min: int):
        self.check_unfilled_orders()
        self.check_trading_orders()
        self.check_infinite_order_loop(max_created_orders_per_min)

    def check_unfilled_orders(self):
        if self.config.unfilled_order_expiration == 0:
            return

        unfilled_sell_orders, unfilled_buy_orders = self.get_unfilled_tracked_orders_by_side()

        for unfilled_order in unfilled_sell_orders + unfilled_buy_orders:
            if has_unfilled_order_expired(unfilled_order, self.config.unfilled_order_expiration, self.get_market_data_provider_time()):
                self.logger().info("unfilled_order_has_expired")
                self.cancel_unfilled_tracked_order(unfilled_order)

    def check_trading_orders(self):
        current_price = self.get_mid_price()
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side()

        for filled_order in filled_sell_orders + filled_buy_orders:
            if has_current_price_reached_stop_loss(filled_order, current_price):
                self.logger().info(f"current_price_has_reached_stop_loss | current_price:{current_price}")
                self.close_filled_order(filled_order, OrderType.MARKET, CloseType.STOP_LOSS)
                continue

            if has_filled_order_reached_time_limit(filled_order, self.get_market_data_provider_time()):
                self.logger().info(f"filled_order_has_reached_time_limit | current_price:{current_price}")
                time_limit_order_type = filled_order.triple_barrier.time_limit_order_type
                self.close_filled_order(filled_order, time_limit_order_type, CloseType.TIME_LIMIT)

    def check_infinite_order_loop(self, max_created_orders_per_min: int):
        # If we have more than `max_created_orders_per_min` created orders during the same minute, it indicates a probable infinite loop,
        # and we should close all orders and exit

        if not self._has_too_many_orders_in_one_minute(max_created_orders_per_min):
            return

        self.log_error("Infinite loop detected!")

        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side()
        self.close_filled_orders(filled_sell_orders + filled_buy_orders, OrderType.MARKET, CloseType.EARLY_STOP)

        unfilled_sell_orders, unfilled_buy_orders = self.get_unfilled_tracked_orders_by_side()

        for unfilled_order in unfilled_sell_orders + unfilled_buy_orders:
            self.cancel_unfilled_tracked_order(unfilled_order)

        HummingbotApplication.main_application().stop()

    def _has_too_many_orders_in_one_minute(self, max_created_orders_per_min: int) -> bool:
        active_orders: List[TrackedOrder] = self.get_active_tracked_orders()

        minute_buckets = [bucket_minute(o) for o in active_orders]
        counts = Counter(minute_buckets)  # Count how many orders fell into each minute

        return any(count > max_created_orders_per_min for count in counts.values())

    def log_error(self, text: str):
        self.logger().error(f"{LogLevel.ERROR.value}: {text}")
        self.telegram(LogLevel.ERROR, text)

    def telegram(self, log_lvl: LogLevel, text: str):
        raise NotImplementedError

    def send_telegram(self, header: str, text: str):
        try:
            resp = requests.post(
                TELEGRAM_API_URL,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": f"{header}\n{text}"
                },
                timeout=10,
            )

            resp.raise_for_status()

        except RequestException as ex:
            self.logger().error(f"{LogLevel.ERROR.value}: Failed to send Telegram message: {ex}")

    @staticmethod
    def get_market_data_provider_time() -> float:
        return datetime.now().timestamp()
