from decimal import Decimal
from typing import List, Optional, Tuple

from hummingbot.core.data_type.common import OrderType, TradeType, PositionAction, PriceType
from hummingbot.core.event.events import SellOrderCreatedEvent, BuyOrderCreatedEvent, OrderFilledEvent
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from scripts.pk.arthur import ArthurStrategy
from scripts.pk.pk_utils import has_current_price_reached_stop_loss, has_current_price_reached_take_profit, has_filled_order_reached_time_limit, \
    has_unfilled_order_expired
from scripts.pk.tracked_order_details import TrackedOrderDetails


class PkStrategy:
    def __init__(self, child: ArthurStrategy):
        self.child = child

        self.is_a_sell_order_being_created = False
        self.is_a_buy_order_being_created = False

        self.tracked_orders: List[TrackedOrderDetails] = []

    def get_mid_price(self) -> Decimal:
        market_data_provider = self.child.market_data_provider
        connector_name = self.child.config.connector_name
        trading_pair = self.child.config.trading_pair

        return market_data_provider.get_price_by_type(connector_name, trading_pair, PriceType.MidPrice)

    def get_position_quote_amount(self, side: TradeType) -> Decimal:
        total_amount_quote = self.child.config.total_amount_quote
        leverage = self.child.config.leverage

        amount_quote = Decimal(total_amount_quote)

        # If amount_quote = 100 USDT with leverage 20x, the quote position should be 500
        position_quote_amount = amount_quote * leverage / 4

        if side == TradeType.SELL:
            position_quote_amount = position_quote_amount * Decimal(0.67)  # Less, because closing a Short position on SL costs significantly more

        return position_quote_amount

    def get_best_ask(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestAsk)

    def get_best_bid(self) -> Decimal:
        return self._get_best_ask_or_bid(PriceType.BestBid)

    def _get_best_ask_or_bid(self, price_type: PriceType) -> Decimal:
        market_data_provider = self.child.market_data_provider
        connector_name = self.child.config.connector_name
        trading_pair = self.child.config.trading_pair

        return market_data_provider.get_price_by_type(connector_name, trading_pair, price_type)

    def get_executor_config(self, side: TradeType, entry_price: Decimal) -> PositionExecutorConfig:
        market_data_provider = self.child.market_data_provider
        connector_name = self.child.config.connector_name
        trading_pair = self.child.config.trading_pair
        leverage = self.child.config.leverage
        triple_barrier_config = self.child.get_triple_barrier_config()

        return PositionExecutorConfig(
            timestamp=market_data_provider.time(),
            connector_name=connector_name,
            trading_pair=trading_pair,
            side=side,
            entry_price=entry_price,
            amount=self.get_position_quote_amount(side) / entry_price,
            triple_barrier_config=triple_barrier_config,
            leverage=leverage
        )

    def find_tracked_order_of_id(self, order_id: str) -> Optional[TrackedOrderDetails]:
        orders_of_that_id = [order for order in self.tracked_orders if order.order_id == order_id]
        return None if len(orders_of_that_id) == 0 else orders_of_that_id[0]

    def find_last_terminated_filled_order(self) -> Optional[TrackedOrderDetails]:
        terminated_filled_orders = [order for order in self.tracked_orders if order.last_filled_at and order.terminated_at]

        if len(terminated_filled_orders) == 0:
            return None

        return max(terminated_filled_orders, key=lambda order: order.terminated_at)

    def get_active_tracked_orders(self) -> List[TrackedOrderDetails]:
        return [order for order in self.tracked_orders if order.created_at and not order.terminated_at]

    def get_active_tracked_orders_by_side(self) -> Tuple[List[TrackedOrderDetails], List[TrackedOrderDetails]]:
        active_orders = self.get_active_tracked_orders()
        active_sell_orders = [order for order in active_orders if order.side == TradeType.SELL]
        active_buy_orders = [order for order in active_orders if order.side == TradeType.BUY]
        return active_sell_orders, active_buy_orders

    def get_unfilled_tracked_orders_by_side(self) -> Tuple[List[TrackedOrderDetails], List[TrackedOrderDetails]]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()
        unfilled_sell_orders = [order for order in active_sell_orders if not order.last_filled_at]
        unfilled_buy_orders = [order for order in active_buy_orders if not order.last_filled_at]
        return unfilled_sell_orders, unfilled_buy_orders

    def get_filled_tracked_orders_by_side(self) -> Tuple[List[TrackedOrderDetails], List[TrackedOrderDetails]]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side()
        filled_sell_orders = [order for order in active_sell_orders if order.last_filled_at]
        filled_buy_orders = [order for order in active_buy_orders if order.last_filled_at]
        return filled_sell_orders, filled_buy_orders

    def create_order(self, side: TradeType, entry_price: Decimal):
        executor_config = self.get_executor_config(side, entry_price)

        connector_name = executor_config.connector_name
        trading_pair = executor_config.trading_pair
        amount = executor_config.amount
        entry_price = executor_config.entry_price
        triple_barrier_config = executor_config.triple_barrier_config
        open_order_type = triple_barrier_config.open_order_type

        if executor_config.side == TradeType.SELL:
            self.is_a_sell_order_being_created = True

            order_id = self.child.sell(connector_name, trading_pair, amount, open_order_type, entry_price)

            self.tracked_orders.append(TrackedOrderDetails(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.SELL,
                order_id=order_id,
                position=PositionAction.OPEN.value,
                amount=amount,
                entry_price=entry_price,
                triple_barrier_config=triple_barrier_config
            ))

            self.is_a_sell_order_being_created = False

        else:
            self.is_a_buy_order_being_created = True

            order_id = self.child.buy(connector_name, trading_pair, amount, open_order_type, entry_price)

            self.tracked_orders.append(TrackedOrderDetails(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.BUY,
                order_id=order_id,
                position=PositionAction.OPEN.value,
                amount = amount,
                entry_price=entry_price,
                triple_barrier_config=triple_barrier_config
            ))

            self.is_a_buy_order_being_created = False

        # TODO: remove
        self.child.logger().info(f"create_order | self.tracked_orders: {self.tracked_orders}")

    def cancel_order(self, tracked_order: TrackedOrderDetails):
        if tracked_order.last_filled_at:
            self.close_filled_order(tracked_order, OrderType.MARKET)
        else:
            self.cancel_unfilled_order(tracked_order)

    def close_filled_order(self, tracked_order: TrackedOrderDetails, order_type: OrderType):
        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair
        filled_amount = tracked_order.filled_amount

        current_price = self.get_mid_price()

        # TODO: remove
        self.child.logger().info(f"close_filled_order | tracked_order: {tracked_order}")

        if tracked_order.side == TradeType.SELL:
            self.child.buy(
                connector_name,
                trading_pair,
                filled_amount,
                order_type,
                current_price,
                PositionAction.CLOSE
            )
        else:
            self.child.sell(
                connector_name,
                trading_pair,
                filled_amount,
                order_type,
                current_price,
                PositionAction.CLOSE
            )

        for order in self.tracked_orders:
            if order.order_id == tracked_order.order_id:
                market_data_provider = self.child.market_data_provider
                order.terminated_at = market_data_provider.time()
                break

    def cancel_unfilled_order(self, tracked_order: TrackedOrderDetails):
        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair
        order_id = tracked_order.order_id

        # TODO: remove
        self.child.logger().info(f"cancel_unfilled_order | tracked_order: {tracked_order}")

        self.child.cancel(connector_name, trading_pair, order_id)

        for order in self.tracked_orders:
            if order.order_id == tracked_order.order_id:
                market_data_provider = self.child.market_data_provider
                order.terminated_at = market_data_provider.time()
                break

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
        self.child.logger().info(f"did_create_sell_order | self.tracked_orders: {self.tracked_orders}")

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
        self.child.logger().info(f"did_create_buy_order | self.tracked_orders: {self.tracked_orders}")

    def did_fill_order(self, filled_event: OrderFilledEvent):
        position = filled_event.position

        if not position or position != PositionAction.OPEN.value:
            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == filled_event.order_id:
                tracked_order.filled_amount += filled_event.amount
                tracked_order.last_filled_at = filled_event.timestamp
                break

        # TODO: remove
        self.child.logger().info(f"did_fill_order | self.tracked_orders: {self.tracked_orders}")

    def can_create_order(self, side: TradeType) -> bool:
        if self.get_position_quote_amount(side) == 0:
            return False

        if side == TradeType.SELL and self.is_a_sell_order_being_created:
            self.child.logger().info(f"Another SELL order is being created, avoiding a duplicate")
            return False

        if side == TradeType.BUY and self.is_a_buy_order_being_created:
            self.child.logger().info(f"Another BUY order is being created, avoiding a duplicate")
            return False

        last_terminated_filled_order = self.find_last_terminated_filled_order()

        if not last_terminated_filled_order:
            return True

        market_data_provider = self.child.market_data_provider
        cooldown_time_min = self.child.config.cooldown_time_min

        if last_terminated_filled_order.terminated_at + cooldown_time_min * 60 > market_data_provider.time():
            self.child.logger().info("Cooldown not passed yet")
            return False

        return True

    def check_orders(self):
        self.check_unfilled_orders()
        self.check_trading_orders()

    def check_unfilled_orders(self):
        unfilled_order_expiration_min = self.child.config.unfilled_order_expiration_min
        market_data_provider = self.child.market_data_provider

        if not unfilled_order_expiration_min:
            return

        unfilled_sell_orders, unfilled_buy_orders = self.get_unfilled_tracked_orders_by_side()

        for unfilled_order in unfilled_sell_orders + unfilled_buy_orders:
            if has_unfilled_order_expired(unfilled_order, unfilled_order_expiration_min, market_data_provider.time()):
                self.child.logger().info("has_unfilled_order_expired")
                self.cancel_unfilled_order(unfilled_order)

    def check_trading_orders(self):
        current_price = self.get_mid_price()
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side()

        for filled_order in filled_sell_orders + filled_buy_orders:
            if has_current_price_reached_stop_loss(filled_order, current_price):
                self.child.logger().info("has_current_price_reached_stop_loss")
                stop_loss_order_type = filled_order.triple_barrier_config.stop_loss_order_type
                self.close_filled_order(filled_order, stop_loss_order_type)
                continue

            if has_current_price_reached_take_profit(filled_order, current_price):
                self.child.logger().info("has_current_price_reached_take_profit")
                take_profit_order_type = filled_order.triple_barrier_config.take_profit_order_type
                self.close_filled_order(filled_order, take_profit_order_type)
                continue

            market_data_provider = self.child.market_data_provider

            if has_filled_order_reached_time_limit(filled_order, market_data_provider.time()):
                self.child.logger().info("has_filled_order_reached_time_limit")
                time_limit_order_type = filled_order.triple_barrier_config.time_limit_order_type
                self.close_filled_order(filled_order, time_limit_order_type)
