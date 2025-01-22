from decimal import Decimal
from typing import Dict, List, Set

from pydantic import Field

from hummingbot.core.data_type.common import PositionMode
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase


class ExcaliburConfig(StrategyV2ConfigBase):
    # Standard attributes START - avoid renaming
    markets: Dict[str, Set[str]] = Field(default_factory=dict)

    candles_config: List[CandlesConfig] = Field(default_factory=lambda: [
        CandlesConfig(
            connector="binance_perpetual",
            interval="3m",
            max_records=50,
            trading_pair = "GOAT-USDT"
        )
    ])

    controllers_config: List[str] = Field(default_factory=list)
    # Standard attributes END

    # Used by PkStrategy
    connector_name: str = "hyperliquid_perpetual"
    trading_pair: str = "GOAT-USD"
    leverage: int = 5
    unfilled_order_expiration: int = 10
    limit_take_profit_price_delta_bps: int = 0

    position_mode: PositionMode = PositionMode.ONEWAY

    # Triple Barrier

    # Order settings
    amount_quote: Decimal = 20.0
