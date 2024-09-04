import asyncio
from decimal import Decimal
from typing import Dict, Set

from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.core.data_type.common import OrderType
from hummingbot.core.event.events import BuyOrderCreatedEvent
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

base_currency = "ETH"
quote_currency = "USDT"
trading_pair = f"{base_currency}-{quote_currency}"
connector_name = "kucoin_paper_trade"


class Example3(ScriptStrategyBase):
    order_amount_usd = Decimal(100)
    nb_orders_created = 0
    nb_orders_to_create = 3

    markets: Dict[str, Set[str]] = {connector_name: {trading_pair}}
    rate_oracle = RateOracle.get_instance()

    def on_tick(self):
        if self.nb_orders_created == self.nb_orders_to_create:
            return

        # connector = self.connectors[connector_name]
        # mid_price = connector.get_mid_price(trading_pair)

        oracle_quote_price = self.rate_oracle.get_pair_rate(trading_pair)
        amount_to_buy = self.order_amount_usd / oracle_quote_price

        buy_price = oracle_quote_price * Decimal("0.99")

        order_id = self.buy(connector_name, trading_pair, amount_to_buy, OrderType.LIMIT, buy_price)
        self.logger().info(f"Created buy order {order_id}")

    def did_create_buy_order(self, event: BuyOrderCreatedEvent):
        if event.trading_pair == trading_pair:
            self.logger().info(f"Buy order event with order ID {event.order_id}")
            self.nb_orders_created += 1

            if self.nb_orders_created == self.nb_orders_to_create:
                self.logger().info("All orders created, stopping the bot")
                HummingbotApplication.main_application().stop()

    def get_rate_sync(self, base_token: str) -> Decimal:
        try:
            # Try to get the running loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If there is no running loop, create a new one and run the async function
            loop = None

        if loop and loop.is_running():
            # If there is a running loop, create a new loop to run the async function
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(self.rate_oracle.get_rate(base_token))
            finally:
                asyncio.set_event_loop(loop)
                new_loop.close()
            return result
        else:
            # If there is no running loop, use the existing one
            return asyncio.run(self.rate_oracle.get_rate(base_token))
