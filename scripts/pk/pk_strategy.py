from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Tuple

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
from scripts.ashbringer_config import ExcaliburConfig
from scripts.pk.close_order import CloseOrder
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import (
    has_current_price_reached_stop_loss,
    has_filled_order_reached_time_limit,
    has_unfilled_order_expired,
)
from scripts.pk.take_profit_order import TakeProfitOrder
from scripts.pk.tracked_order import TrackedOrder


class PkStrategy(StrategyV2Base):
    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)
        self.config = config

        self.is_a_sell_order_being_created = False
        self.is_a_buy_order_being_created = False

        self.tracked_orders: List[TrackedOrder] = []
        self.close_orders: List[CloseOrder] = []
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

    def find_last_terminated_filled_order(self, side: TradeType, ref: str) -> TrackedOrder | None:
        terminated_filled_orders = [order for order in self.tracked_orders if (
            order.side == side and
            order.ref == ref and
            order.last_filled_at and
            order.terminated_at
        )]

        if len(terminated_filled_orders) == 0:
            return None

        return max(terminated_filled_orders, key=lambda order: order.terminated_at)

    def get_active_tracked_orders(self, ref: str | None = None) -> List[TrackedOrder]:
        active_tracked_orders = [order for order in self.tracked_orders if (
            order.created_at and
            not order.closing_at and
            not order.terminated_at
        )]

        if ref:
            active_tracked_orders = [order for order in active_tracked_orders if order.ref == ref]

        return active_tracked_orders

    def get_closing_tracked_orders(self, ref: str | None = None) -> List[TrackedOrder]:
        closing_tracked_orders = [order for order in self.tracked_orders if (
            order.created_at and
            order.closing_at and
            not order.terminated_at
        )]

        if ref:
            closing_tracked_orders = [order for order in closing_tracked_orders if order.ref == ref]

        return closing_tracked_orders

    def get_active_tracked_orders_by_side(self, ref: str | None = None) -> Tuple[List[TrackedOrder], List[TrackedOrder]]:
        active_orders = self.get_active_tracked_orders(ref)
        active_sell_orders = [order for order in active_orders if order.side == TradeType.SELL]
        active_buy_orders = [order for order in active_orders if order.side == TradeType.BUY]
        return active_sell_orders, active_buy_orders

    def get_unfilled_tracked_orders_by_side(self, ref: str | None = None) -> Tuple[List[TrackedOrder], List[TrackedOrder]]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ref)
        unfilled_sell_orders = [order for order in active_sell_orders if not order.last_filled_at]
        unfilled_buy_orders = [order for order in active_buy_orders if not order.last_filled_at]
        return unfilled_sell_orders, unfilled_buy_orders

    def get_filled_tracked_orders_by_side(self, ref: str | None = None) -> Tuple[List[TrackedOrder], List[TrackedOrder]]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ref)
        filled_sell_orders = [order for order in active_sell_orders if order.last_filled_at]
        filled_buy_orders = [order for order in active_buy_orders if order.last_filled_at]
        return filled_sell_orders, filled_buy_orders

    def get_all_unfilled_tp_orders(self) -> List[TakeProfitOrder]:
        return [order for order in self.take_profit_orders if not order.last_filled_at]

    def get_all_filled_tp_orders(self) -> List[TakeProfitOrder]:
        return [order for order in self.take_profit_orders if order.last_filled_at]

    def get_unfilled_tp_orders(self, tracked_order: TrackedOrder) -> List[TakeProfitOrder]:
        return [order for order in self.get_all_unfilled_tp_orders() if order.tracked_order.order_id == tracked_order.order_id]

    def get_latest_filled_tp_order(self) -> TakeProfitOrder | None:
        filled_tp_orders = self.get_all_filled_tp_orders()

        if len(filled_tp_orders) == 0:
            return None

        return filled_tp_orders[-1]  # The latest filled TP is necessarily the last one created

    def create_order(self, side: TradeType, entry_price: Decimal, triple_barrier: TripleBarrier, amount_quote: Decimal, ref: str):
        executor_config = self.get_executor_config(side, entry_price, amount_quote)
        self.create_individual_order(executor_config, triple_barrier, ref)

    # async def create_twap_market_orders(self, side: TradeType, entry_price: Decimal, triple_barrier: TripleBarrier, amount_quote: Decimal, ref: str):
    #     executor_config = self.get_executor_config(side, entry_price, amount_quote, True)
    #
    #     for _ in range(self.config.market_order_twap_count):
    #         is_an_order_being_created: bool = self.is_a_sell_order_being_created if executor_config.side == TradeType.SELL else self.is_a_buy_order_being_created
    #
    #         if is_an_order_being_created:
    #             self.logger().error("ERROR: Cannot create another individual order, as one is being created")
    #         else:
    #             self.create_individual_order(executor_config, triple_barrier, ref)
    #             await asyncio.sleep(self.config.market_order_twap_interval)

    def create_individual_order(self, executor_config: PositionExecutorConfig, triple_barrier: TripleBarrier, ref: str):
        connector_name = executor_config.connector_name
        trading_pair = executor_config.trading_pair
        amount = executor_config.amount
        entry_price = executor_config.entry_price
        open_order_type = triple_barrier.open_order_type

        if executor_config.side == TradeType.SELL:
            self.is_a_sell_order_being_created = True

            order_id = self.sell(connector_name, trading_pair, amount, open_order_type, entry_price)

            self.tracked_orders.append(TrackedOrder(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.SELL,
                order_id=order_id,
                amount=amount,
                entry_price=entry_price,
                triple_barrier=triple_barrier,
                ref=ref,
                created_at=self.get_market_data_provider_time()  # Because some exchanges such as gate_io trigger the `did_create_xxx_order` event after 1s
            ))

            self.is_a_sell_order_being_created = False

        else:
            self.is_a_buy_order_being_created = True

            order_id = self.buy(connector_name, trading_pair, amount, open_order_type, entry_price)

            self.tracked_orders.append(TrackedOrder(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.BUY,
                order_id=order_id,
                amount=amount,
                entry_price=entry_price,
                triple_barrier=triple_barrier,
                ref=ref,
                created_at=self.get_market_data_provider_time()
            ))

            self.is_a_buy_order_being_created = False

        self.logger().info(f"create_order: {self.tracked_orders[-1]}")

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

        self.logger().info(f"create_tp_order: {self.take_profit_orders[-1]}")

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

            if not tracked_order.closing_at:
                tracked_order.closing_at = timestamp

    def close_filled_order(self, filled_order: TrackedOrder, market_or_limit: OrderType, close_type: CloseType):
        self.cancel_take_profit_for_order(filled_order)

        connector_name = filled_order.connector_name
        trading_pair = filled_order.trading_pair
        filled_amount = filled_order.filled_amount

        close_price_sell = self.get_best_bid() * Decimal(1 - self.config.limit_take_profit_price_delta_bps / 10000)
        close_price_buy = self.get_best_ask() * Decimal(1 + self.config.limit_take_profit_price_delta_bps / 10000)

        self.logger().info(f"close_filled_order | close_price:{close_price_sell if filled_order.side == TradeType.SELL else close_price_buy}")

        order_id = (
            self.buy(connector_name, trading_pair, filled_amount, market_or_limit, close_price_sell, PositionAction.CLOSE) if filled_order.side == TradeType.SELL else
            self.sell(connector_name, trading_pair, filled_amount, market_or_limit, close_price_buy, PositionAction.CLOSE)
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
        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair
        order_id = tracked_order.order_id

        self.cancel_take_profit_for_order(tracked_order)

        self.logger().info(f"cancel_unfilled_tracked_order: {tracked_order}")
        self.cancel(connector_name, trading_pair, order_id)

    def cancel_take_profit_for_order(self, tracked_order: TrackedOrder):
        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair

        for tp_order in self.get_unfilled_tp_orders(tracked_order):
            order_id = tp_order.order_id
            self.logger().info(f"cancel_take_profit_for_order: {tp_order}")
            self.cancel(connector_name, trading_pair, order_id)

    def did_create_sell_order(self, created_event: SellOrderCreatedEvent):
        tracked_order: TrackedOrder = self.find_tracked_order_of_id(created_event.order_id)

        if tracked_order and created_event.position == PositionAction.OPEN.value:
            tracked_order.amount = created_event.amount
            tracked_order.exchange_order_id = created_event.exchange_order_id
            self.logger().info(f"did_create_sell_order | tracked_order.amount:{tracked_order.amount} | exchange_order_id:{tracked_order.exchange_order_id}")
            return

        tp_order: TakeProfitOrder = self.find_tp_order_of_id(created_event.order_id)

        if tp_order:
            tp_order.amount = created_event.amount
            tp_order.exchange_order_id = created_event.exchange_order_id
            self.logger().info(f"did_create_sell_order | tp_order.amount:{tp_order.amount} | exchange_order_id:{tp_order.exchange_order_id}")

    def did_create_buy_order(self, created_event: BuyOrderCreatedEvent):
        tracked_order: TrackedOrder = self.find_tracked_order_of_id(created_event.order_id)

        if tracked_order and created_event.position == PositionAction.OPEN.value:
            tracked_order.amount = created_event.amount
            tracked_order.exchange_order_id = created_event.exchange_order_id
            self.logger().info(f"did_create_buy_order | tracked_order.amount:{tracked_order.amount} | exchange_order_id:{tracked_order.exchange_order_id}")
            return

        tp_order: TakeProfitOrder = self.find_tp_order_of_id(created_event.order_id)

        if tp_order:
            tp_order.amount = created_event.amount
            tp_order.exchange_order_id = created_event.exchange_order_id
            self.logger().info(f"did_create_buy_order | tp_order.amount:{tp_order.amount} | exchange_order_id:{tp_order.exchange_order_id}")

    def did_fail_order(self, order_failed_event: MarketOrderFailureEvent):
        self.logger().info(f"did_fail_order | order_id:{order_failed_event.order_id} | order_type:{order_failed_event.order_type}")

        tracked_order: TrackedOrder = self.find_tracked_order_of_id(order_failed_event.order_id)

        if tracked_order:
            self.logger().info("did_fail_order > Found it in self.tracked_orders")
            self.tracked_orders.remove(tracked_order)
            return

        close_order: CloseOrder = self.find_close_order_of_id(order_failed_event.order_id)

        if close_order:
            self.logger().info("did_fail_order > Found it in self.close_orders")
            self.close_orders.remove(close_order)

            tracked_order: TrackedOrder = self.find_tracked_order_of_id(close_order.tracked_order.order_id)

            if tracked_order:
                self.logger().info("did_fail_order > 'Unclosing/unterminating' the tracked order")
                tracked_order.closing_at = None
                tracked_order.terminated_at = None
                tracked_order.close_type = None

            return

        tp_order: TakeProfitOrder = self.find_tp_order_of_id(order_failed_event.order_id)

        if tp_order:
            self.logger().info("did_fail_order > Found it in self.take_profit_limit_orders")
            self.take_profit_orders.remove(tp_order)

    def did_fill_order(self, filled_event: OrderFilledEvent):
        self.logger().info(f"did_fill_order | position:{filled_event.position} | amount:{filled_event.amount} at price:{filled_event.price}")

        tp_order: TakeProfitOrder = self.find_tp_order_of_id(filled_event.order_id)

        if tp_order:
            self.logger().info(f"did_fill_order > Take Profit price reached for tracked order: {tp_order.tracked_order}")

            tp_order.filled_amount += filled_event.amount
            tp_order.last_filled_at = filled_event.timestamp
            tp_order.last_filled_price = filled_event.price

            tracked_order: TrackedOrder = self.find_tracked_order_of_id(tp_order.tracked_order.order_id)

            if tracked_order:
                tracked_order.filled_amount -= filled_event.amount
                self.logger().info(f"did_fill_order > tracked_order.filled_amount reduced to: {tracked_order.filled_amount}")

                if tracked_order.filled_amount == 0:
                    self.logger().info("did_fill_order > tracked_order.filled_amount == 0. Terminating it")
                    self.update_order_to_terminated(tracked_order.order_id, filled_event.timestamp, CloseType.TAKE_PROFIT)

            return

        tracked_order: TrackedOrder = self.find_tracked_order_of_id(filled_event.order_id)

        if tracked_order and filled_event.position == PositionAction.OPEN.value:
            self.logger().info(f"did_fill_order > tracked_order.filled_amount before update: {tracked_order.filled_amount}")

            tracked_order.filled_amount += filled_event.amount
            tracked_order.last_filled_at = filled_event.timestamp
            tracked_order.last_filled_price = filled_event.price

            self.logger().info(f"did_fill_order > tracked_order.filled_amount increased to: {tracked_order.filled_amount}")
            return

        close_order: CloseOrder = self.find_close_order_of_id(filled_event.order_id)

        if close_order:
            tracked_order = self.find_tracked_order_of_id(close_order.tracked_order.order_id)

            if tracked_order:
                tracked_order.filled_amount -= filled_event.amount
                self.logger().info(f"did_fill_order > tracked_order.filled_amount reduced to: {tracked_order.filled_amount}")

                if tracked_order.filled_amount == 0:
                    self.logger().info("did_fill_order > tracked_order.filled_amount == 0. Terminating it")
                    self.update_order_to_terminated(tracked_order.order_id, filled_event.timestamp)

    def did_cancel_order(self, cancelled_event: OrderCancelledEvent):
        self.logger().info(f"did_cancel_order | cancelled_event:{cancelled_event}")
        self.update_order_to_terminated(cancelled_event.order_id, self.get_market_data_provider_time(), CloseType.EXPIRED)

        tp_order: TakeProfitOrder = self.find_tp_order_of_id(cancelled_event.order_id)

        if tp_order:
            self.take_profit_orders.remove(tp_order)

    def can_create_order(self, side: TradeType, ref: str, cooldown_time_min: int) -> bool:
        if side == TradeType.SELL and self.is_a_sell_order_being_created:
            self.logger().error("ERROR: Another SELL order is being created, avoiding a duplicate")
            return False

        if side == TradeType.BUY and self.is_a_buy_order_being_created:
            self.logger().error("ERROR: Another BUY order is being created, avoiding a duplicate")
            return False

        last_terminated_filled_order = self.find_last_terminated_filled_order(side, ref)

        if not last_terminated_filled_order:
            return True

        if last_terminated_filled_order.terminated_at + cooldown_time_min * 60 > self.get_market_data_provider_time():
            self.logger().info(f"Cooldown not passed yet for {side}")
            return False

        return True

    def check_orders(self):
        self.check_unfilled_orders()
        self.check_trading_orders()

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

    @staticmethod
    def get_market_data_provider_time() -> float:
        return datetime.now().timestamp()
