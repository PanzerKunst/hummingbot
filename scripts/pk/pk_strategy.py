from decimal import Decimal
from typing import List, Optional, Tuple, Dict

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import PriceType, TradeType, PositionAction, OrderType
from hummingbot.core.event.events import SellOrderCreatedEvent, BuyOrderCreatedEvent, OrderFilledEvent
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.pk.galahad_config import GalahadConfig
from scripts.pk.pk_utils import has_unfilled_order_expired, has_current_price_reached_stop_loss, has_current_price_reached_take_profit, \
    has_filled_order_reached_time_limit
from scripts.pk.tracked_order_details import TrackedOrderDetails


class PkStrategy(StrategyV2Base):
    def __init__(self, connectors: Dict[str, ConnectorBase], config: GalahadConfig):
        super().__init__(connectors, config)
        self.config = config

        self.is_a_sell_order_being_created = False
        self.is_a_buy_order_being_created = False

        self.tracked_orders: List[TrackedOrderDetails] = []

    def get_position_quote_amount(self, side: TradeType) -> Decimal:
        total_amount_quote = self.config.total_amount_quote
        leverage = self.config.leverage

        amount_quote = Decimal(total_amount_quote)

        # If amount_quote = 100 USDT with leverage 20x, the quote position should be 500
        position_quote_amount = amount_quote * leverage / 4

        if side == TradeType.SELL:
            position_quote_amount = position_quote_amount * Decimal(0.67)  # Less, because closing a Short position on SL costs significantly more

        return position_quote_amount

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

    def get_executor_config(self, side: TradeType, entry_price: Decimal, triple_barrier_config: TripleBarrierConfig, amount_multiplier: Decimal) -> PositionExecutorConfig:
        connector_name = self.config.connector_name
        trading_pair = self.config.trading_pair
        leverage = self.config.leverage

        amount: Decimal = self.get_position_quote_amount(side) / entry_price * amount_multiplier

        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=connector_name,
            trading_pair=trading_pair,
            side=side,
            entry_price=entry_price,
            amount=amount,
            triple_barrier_config=triple_barrier_config,
            leverage=leverage
        )

    def find_tracked_order_of_id(self, order_id: str) -> Optional[TrackedOrderDetails]:
        orders_of_that_id = [order for order in self.tracked_orders if order.order_id == order_id]
        return None if len(orders_of_that_id) == 0 else orders_of_that_id[0]

    def find_last_terminated_filled_order(self, side: TradeType) -> Optional[TrackedOrderDetails]:
        terminated_filled_orders = [order for order in self.tracked_orders if order.side == side and order.last_filled_at and order.terminated_at]

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

    def create_order(self, side: TradeType, entry_price: Decimal, triple_barrier_config: TripleBarrierConfig, amount_multiplier: Decimal = 1):
        executor_config = self.get_executor_config(side, entry_price, triple_barrier_config, amount_multiplier)

        connector_name = executor_config.connector_name
        trading_pair = executor_config.trading_pair
        amount = executor_config.amount
        entry_price = executor_config.entry_price
        triple_barrier_config = executor_config.triple_barrier_config
        open_order_type = triple_barrier_config.open_order_type

        if executor_config.side == TradeType.SELL:
            self.is_a_sell_order_being_created = True

            order_id = self.sell(connector_name, trading_pair, amount, open_order_type, entry_price)

            self.tracked_orders.append(TrackedOrderDetails(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.SELL,
                order_id=order_id,
                position=PositionAction.OPEN.value,
                amount=amount,
                entry_price=entry_price,
                triple_barrier_config=triple_barrier_config,
                created_at=self.market_data_provider.time()  # Because some exchanges such as gate_io trigger the `did_create_xxx_order` event after 1s
            ))

            self.is_a_sell_order_being_created = False

        else:
            self.is_a_buy_order_being_created = True

            order_id = self.buy(connector_name, trading_pair, amount, open_order_type, entry_price)

            self.tracked_orders.append(TrackedOrderDetails(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=TradeType.BUY,
                order_id=order_id,
                position=PositionAction.OPEN.value,
                amount = amount,
                entry_price=entry_price,
                triple_barrier_config=triple_barrier_config,
                created_at=self.market_data_provider.time()
            ))

            self.is_a_buy_order_being_created = False

        self.logger().info(f"create_order: {self.tracked_orders[-1]}")

    def cancel_tracked_order(self, tracked_order: TrackedOrderDetails):
        if tracked_order.last_filled_at:
            self.close_filled_order(tracked_order, OrderType.MARKET, CloseType.EARLY_STOP)
        else:
            self.cancel_unfilled_order(tracked_order)

    def close_filled_order(self, tracked_order: TrackedOrderDetails, order_type: OrderType, close_type: CloseType):
        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair
        filled_amount = tracked_order.filled_amount

        close_price = self.get_best_bid() if tracked_order.side == TradeType.SELL else self.get_best_ask()

        self.logger().info(f"close_filled_order:{tracked_order} | close_price:{close_price}")

        if tracked_order.side == TradeType.SELL:
            self.buy(
                connector_name,
                trading_pair,
                filled_amount,
                order_type,
                close_price,
                PositionAction.CLOSE
            )
        else:
            self.sell(
                connector_name,
                trading_pair,
                filled_amount,
                order_type,
                close_price,
                PositionAction.CLOSE
            )

        for order in self.tracked_orders:
            if order.order_id == tracked_order.order_id:
                order.terminated_at = self.market_data_provider.time()
                order.close_type = close_type
                break

    def cancel_unfilled_order(self, tracked_order: TrackedOrderDetails):
        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair
        order_id = tracked_order.order_id

        self.logger().info(f"cancel_unfilled_order: {tracked_order}")

        self.cancel(connector_name, trading_pair, order_id)

        for order in self.tracked_orders:
            if order.order_id == tracked_order.order_id:
                order.terminated_at = self.market_data_provider.time()
                order.close_type = CloseType.EXPIRED
                break

    def did_create_sell_order(self, created_event: SellOrderCreatedEvent):
        position = created_event.position

        if not position or position == PositionAction.CLOSE.value:
            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == created_event.order_id:
                tracked_order.exchange_order_id = created_event.exchange_order_id,
                self.logger().info(f"did_create_sell_order: {tracked_order}")
                break


    def did_create_buy_order(self, created_event: BuyOrderCreatedEvent):
        position = created_event.position

        if not position or position == PositionAction.CLOSE.value:
            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == created_event.order_id:
                tracked_order.exchange_order_id = created_event.exchange_order_id,
                self.logger().info(f"did_create_buy_order: {tracked_order}")
                break

    def did_fill_order(self, filled_event: OrderFilledEvent):
        position = filled_event.position

        if not position or position == PositionAction.CLOSE.value:
            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == filled_event.order_id:
                tracked_order.filled_amount += filled_event.amount
                tracked_order.last_filled_at = filled_event.timestamp
                self.logger().info(f"did_fill_order: {tracked_order}")
                break

    def can_create_order(self, side: TradeType) -> bool:
        if self.get_position_quote_amount(side) == 0:
            return False

        if side == TradeType.SELL and self.is_a_sell_order_being_created:
            self.logger().info("Another SELL order is being created, avoiding a duplicate")
            return False

        if side == TradeType.BUY and self.is_a_buy_order_being_created:
            self.logger().info("Another BUY order is being created, avoiding a duplicate")
            return False

        last_terminated_filled_order = self.find_last_terminated_filled_order(side)

        if not last_terminated_filled_order:
            return True

        cooldown_time_min = self.config.cooldown_time_min

        if last_terminated_filled_order.terminated_at + cooldown_time_min * 60 > self.market_data_provider.time():
            self.logger().info(f"Cooldown not passed yet for {side}")
            return False

        return True

    def check_orders(self):
        self.check_unfilled_orders()
        self.check_trading_orders()

    def check_unfilled_orders(self):
        unfilled_order_expiration_min = self.config.unfilled_order_expiration_min

        if not unfilled_order_expiration_min:
            return

        unfilled_sell_orders, unfilled_buy_orders = self.get_unfilled_tracked_orders_by_side()

        for unfilled_order in unfilled_sell_orders + unfilled_buy_orders:
            if has_unfilled_order_expired(unfilled_order, unfilled_order_expiration_min, self.market_data_provider.time()):
                self.logger().info("unfilled_order_has_expired")
                self.cancel_unfilled_order(unfilled_order)

    def check_trading_orders(self):
        current_price = self.get_mid_price()
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side()

        for filled_order in filled_sell_orders + filled_buy_orders:
            if has_current_price_reached_stop_loss(filled_order, current_price):
                self.logger().info("current_price_has_reached_stop_loss")
                stop_loss_order_type = filled_order.triple_barrier_config.stop_loss_order_type
                self.close_filled_order(filled_order, stop_loss_order_type, CloseType.STOP_LOSS)
                continue

            if has_current_price_reached_take_profit(filled_order, current_price):
                self.logger().info("current_price_has_reached_take_profit")
                take_profit_order_type = filled_order.triple_barrier_config.take_profit_order_type
                self.close_filled_order(filled_order, take_profit_order_type, CloseType.TAKE_PROFIT)
                continue

            if has_filled_order_reached_time_limit(filled_order, self.market_data_provider.time()):
                self.logger().info("filled_order_has_reached_time_limit")
                time_limit_order_type = filled_order.triple_barrier_config.time_limit_order_type
                self.close_filled_order(filled_order, time_limit_order_type, CloseType.TIME_LIMIT)
