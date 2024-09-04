from typing import Dict, Set

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class InitialExample(ScriptStrategyBase):
    markets: Dict[str, Set[str]] = {"binance_paper_trade": {"BTC-USDT", "ETH-USDT"}}

    def on_tick(self):
        self.logger().info("Tick received")
