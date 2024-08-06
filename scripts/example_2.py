from typing import Dict, List, Set

import pandas as pd

from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

trading_pair: str = "ETH-USDT"


class Example2(ScriptStrategyBase):
    markets: Dict[str, Set[str]] = {
        "binance_paper_trade": {trading_pair},
        "kucoin_paper_trade": {trading_pair},
        "gate_io_paper_trade": {trading_pair}
    }

    def format_status(self) -> str:
        """
        Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self.get_market_trading_pair_tuples()))

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        market_status_df = self.market_status_data_frame(self.get_market_trading_pair_tuples())
        lines.extend(["", "  Market status:"] + ["    " + line for line in market_status_df.to_string(index=False).split("\n")])

        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)

    def market_status_data_frame(self, market_trading_pair_tuples: List[MarketTradingPairTuple]) -> pd.DataFrame:
        markets_data = []
        markets_columns = ["Exchange", "Market", "Best Bid Price", "Best Ask Price", "Mid Price", "Volume (+1%)", "Volume (-1%)"]

        try:
            for market_trading_pair_tuple in market_trading_pair_tuples:
                market, trading_pair, base_asset, quote_asset = market_trading_pair_tuple
                bid_price = market.get_price(trading_pair, False)
                ask_price = market.get_price(trading_pair, True)
                mid_price = float((bid_price + ask_price) / 2)

                mid_price_minus_1pc = 0.99 * mid_price
                mid_price_plus_1pc = 1.01 * mid_price

                market_name = market.display_name.replace("_PaperTrade", "_paper_trade")

                connector = self.connectors[market_name]

                # Ask volume for price 1% above mid price
                volume_midprice_plus_1pc = connector.get_volume_for_price(trading_pair, True, mid_price_plus_1pc).result_volume

                # Bid volume for price 1% below mid price
                volume_midprice_minus_1pc = connector.get_volume_for_price(trading_pair, False, mid_price_minus_1pc).result_volume

                markets_data.append([
                    market_name,
                    trading_pair,
                    float(bid_price),
                    float(ask_price),
                    mid_price,
                    float(volume_midprice_plus_1pc),
                    float(volume_midprice_minus_1pc)
                ])
            return pd.DataFrame(data=markets_data, columns=markets_columns)

        except Exception:
            self.logger().error("Error formatting market stats.", exc_info=True)
