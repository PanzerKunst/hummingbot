from typing import Dict, Set

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

# Initialize 3 markets and log the best ask, best bid and mid price every tick

trading_pair: str = "ETH-USDT"


class Example1(ScriptStrategyBase):
    markets: Dict[str, Set[str]] = {
        "binance_paper_trade": {trading_pair},
        "kucoin_paper_trade": {trading_pair},
        "gate_io_paper_trade": {trading_pair}
    }

    def on_tick(self):
        for connector_name, connector in self.connectors.items():
            self.logger().info(f"Connector: {connector_name}")

            best_ask = connector.get_price(trading_pair, True)
            self.logger().info(f"Best ask: {best_ask}")

            best_bid = connector.get_price(trading_pair, False)
            self.logger().info(f"Best bid: {best_bid}")

            mid_price = connector.get_mid_price(trading_pair)
            self.logger().info(f"Mid price: {mid_price}")
