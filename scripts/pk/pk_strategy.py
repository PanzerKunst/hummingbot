import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.core.event.events import BuyOrderCreatedEvent, OrderFilledEvent, SellOrderCreatedEvent
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.ashbringer_config import ExcaliburConfig
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import (
    has_current_price_reached_stop_loss,
    has_current_price_reached_take_profit,
    has_filled_order_reached_time_limit,
    has_unfilled_order_expired,
    should_close_trailing_stop,
    update_trailing_stop,
)
from scripts.pk.tracked_order_details import TrackedOrderDetails


class PkStrategy(StrategyV2Base):
    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)
        self.config = config

        self.is_a_sell_order_being_created = False
        self.is_a_buy_order_being_created = False

        self.tracked_orders: List[TrackedOrderDetails] = []

    @staticmethod
    def get_position_quote_amount(side: TradeType, amount_quote: int) -> Decimal:
        if side == TradeType.SELL:
            return amount_quote * Decimal(0.67)  # Less, because closing a Short position on SL costs significantly more

        return Decimal(amount_quote)

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

    def get_executor_config(self, side: TradeType, entry_price: Decimal, amount_quote: int, is_twap: bool = False) -> PositionExecutorConfig:
        connector_name = self.config.connector_name
        trading_pair = self.config.trading_pair
        leverage = self.config.leverage

        amount_divider = self.config.market_order_twap_count if is_twap else 1
        amount: Decimal = self.get_position_quote_amount(side, amount_quote) / entry_price / amount_divider

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

    def find_tracked_order_of_id(self, order_id: str) -> Optional[TrackedOrderDetails]:
        orders_of_that_id = [order for order in self.tracked_orders if order.order_id == order_id]
        return None if len(orders_of_that_id) == 0 else orders_of_that_id[0]

    def find_last_terminated_filled_order(self, side: TradeType, ref: str) -> Optional[TrackedOrderDetails]:
        terminated_filled_orders = [order for order in self.tracked_orders if (
            order.side == side and
            order.ref == ref and
            order.last_filled_at and
            order.terminated_at
        )]

        if len(terminated_filled_orders) == 0:
            return None

        return max(terminated_filled_orders, key=lambda order: order.terminated_at)

    def get_active_tracked_orders(self, ref: Optional[str] = None) -> List[TrackedOrderDetails]:
        active_tracked_orders = [order for order in self.tracked_orders if order.created_at and not order.terminated_at]

        if ref:
            active_tracked_orders = [order for order in active_tracked_orders if order.ref == ref]

        return active_tracked_orders

    def get_active_tracked_orders_by_side(self, ref: Optional[str] = None) -> Tuple[List[TrackedOrderDetails], List[TrackedOrderDetails]]:
        active_orders = self.get_active_tracked_orders(ref)
        active_sell_orders = [order for order in active_orders if order.side == TradeType.SELL]
        active_buy_orders = [order for order in active_orders if order.side == TradeType.BUY]
        return active_sell_orders, active_buy_orders

    def get_unfilled_tracked_orders_by_side(self, ref: Optional[str] = None) -> Tuple[List[TrackedOrderDetails], List[TrackedOrderDetails]]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ref)
        unfilled_sell_orders = [order for order in active_sell_orders if not order.last_filled_at]
        unfilled_buy_orders = [order for order in active_buy_orders if not order.last_filled_at]
        return unfilled_sell_orders, unfilled_buy_orders

    def get_filled_tracked_orders_by_side(self, ref: Optional[str] = None) -> Tuple[List[TrackedOrderDetails], List[TrackedOrderDetails]]:
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ref)
        filled_sell_orders = [order for order in active_sell_orders if order.last_filled_at]
        filled_buy_orders = [order for order in active_buy_orders if order.last_filled_at]
        return filled_sell_orders, filled_buy_orders

    def create_order(self, side: TradeType, entry_price: Decimal, triple_barrier: TripleBarrier, amount_quote: int, ref: str):
        executor_config = self.get_executor_config(side, entry_price, amount_quote)
        self.create_individual_order(executor_config, triple_barrier, ref)

    async def create_twap_market_orders(self, side: TradeType, entry_price: Decimal, triple_barrier: TripleBarrier, amount_quote: int, ref: str):
        executor_config = self.get_executor_config(side, entry_price, amount_quote, True)

        for _ in range(self.config.market_order_twap_count):
            is_an_order_being_created: bool = self.is_a_sell_order_being_created if executor_config.side == TradeType.SELL else self.is_a_buy_order_being_created

            if is_an_order_being_created:
                self.logger().error("ERROR: Cannot create another individual order, as one is being created")
            else:
                self.create_individual_order(executor_config, triple_barrier, ref)
                await asyncio.sleep(self.config.market_order_twap_interval)

    def create_individual_order(self, executor_config: PositionExecutorConfig, triple_barrier: TripleBarrier, ref: str):
        connector_name = executor_config.connector_name
        trading_pair = executor_config.trading_pair
        amount = executor_config.amount
        entry_price = executor_config.entry_price
        open_order_type = triple_barrier.open_order_type

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
                triple_barrier=triple_barrier,
                ref=ref,
                created_at=self.get_market_data_provider_time()  # Because some exchanges such as gate_io trigger the `did_create_xxx_order` event after 1s
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
                triple_barrier=triple_barrier,
                ref=ref,
                created_at=self.get_market_data_provider_time()
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

        close_price_sell = self.get_best_bid() * Decimal(1 - self.config.limit_take_profit_price_delta_bps / 10000)
        close_price_buy = self.get_best_ask() * Decimal(1 + self.config.limit_take_profit_price_delta_bps / 10000)

        self.logger().info(f"close_filled_order:{tracked_order} | close_price:{close_price_sell if tracked_order.side == TradeType.SELL else close_price_buy}")

        if tracked_order.side == TradeType.SELL:
            self.buy(
                connector_name,
                trading_pair,
                filled_amount,
                order_type,
                close_price_sell,
                PositionAction.CLOSE
            )
        else:
            self.sell(
                connector_name,
                trading_pair,
                filled_amount,
                order_type,
                close_price_buy,
                PositionAction.CLOSE
            )

        for order in self.tracked_orders:
            if order.order_id == tracked_order.order_id:
                order.terminated_at = self.get_market_data_provider_time()
                order.close_type = close_type
                break

    def market_close_orders(self, filled_orders: List[TrackedOrderDetails], close_type: CloseType):
        for tracked_order in filled_orders:
            self.close_filled_order(tracked_order, OrderType.MARKET, close_type)

    def cancel_unfilled_order(self, tracked_order: TrackedOrderDetails):
        connector_name = tracked_order.connector_name
        trading_pair = tracked_order.trading_pair
        order_id = tracked_order.order_id

        self.logger().info(f"cancel_unfilled_order: {tracked_order}")

        self.cancel(connector_name, trading_pair, order_id)

        for order in self.tracked_orders:
            if order.order_id == tracked_order.order_id:
                order.terminated_at = self.get_market_data_provider_time()
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
                tracked_order.last_filled_price = filled_event.price
                self.logger().info(f"did_fill_order: {tracked_order}")
                break

    def can_create_order(self, side: TradeType, amount_quote: int, ref: str, cooldown_time_min: int) -> bool:
        if self.get_position_quote_amount(side, amount_quote) == 0:
            return False

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
        unfilled_order_expiration_min = self.config.unfilled_order_expiration_min

        if not unfilled_order_expiration_min:
            return

        unfilled_sell_orders, unfilled_buy_orders = self.get_unfilled_tracked_orders_by_side()

        for unfilled_order in unfilled_sell_orders + unfilled_buy_orders:
            if has_unfilled_order_expired(unfilled_order, unfilled_order_expiration_min, self.get_market_data_provider_time()):
                self.logger().info("unfilled_order_has_expired")
                self.cancel_unfilled_order(unfilled_order)

    def check_trading_orders(self):
        current_price = self.get_mid_price()
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side()

        for filled_order in filled_sell_orders + filled_buy_orders:
            if has_current_price_reached_stop_loss(filled_order, current_price):
                self.logger().info(f"current_price_has_reached_stop_loss | current_price:{current_price}")
                stop_loss_order_type = filled_order.triple_barrier.stop_loss_order_type
                self.close_filled_order(filled_order, stop_loss_order_type, CloseType.STOP_LOSS)
                continue

            if has_current_price_reached_take_profit(filled_order, current_price):
                self.logger().info(f"current_price_has_reached_take_profit | current_price:{current_price}")
                take_profit_order_type = filled_order.triple_barrier.take_profit_order_type
                self.close_filled_order(filled_order, take_profit_order_type, CloseType.TAKE_PROFIT)
                continue

            update_trailing_stop(filled_order, current_price)

            if filled_order.trailing_stop_best_price and filled_order.triple_barrier.time_limit:
                filled_order.triple_barrier.time_limit = None  # We disable the time limit

            if filled_order.trailing_stop_best_price == current_price:
                self.logger().info(f"Updated trailing_stop_best_price to:{filled_order.trailing_stop_best_price}")

            if should_close_trailing_stop(filled_order, current_price):
                self.logger().info(f"should_close_trailing_stop | current_price:{current_price}")
                take_profit_order_type = filled_order.triple_barrier.take_profit_order_type
                self.close_filled_order(filled_order, take_profit_order_type, CloseType.TRAILING_STOP)
                continue

            if has_filled_order_reached_time_limit(filled_order, self.get_market_data_provider_time()):
                self.logger().info(f"filled_order_has_reached_time_limit | current_price:{current_price}")
                time_limit_order_type = filled_order.triple_barrier.time_limit_order_type
                self.close_filled_order(filled_order, time_limit_order_type, CloseType.TIME_LIMIT)

    @staticmethod
    def get_market_data_provider_time() -> float:
        return datetime.now().timestamp()
