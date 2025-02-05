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
            interval="1m",
            max_records=1000,  # Need a lot of history to correcly compute EMAs
            trading_pair = "AI16Z-USDT"
        )
    ])

    controllers_config: List[str] = Field(default_factory=list)
    config_update_interval: int = 10
    # Standard attributes END

    # Used by PkStrategy
    connector_name: str = "hyperliquid_perpetual"
    trading_pair: str = "AI16Z-USD"
    leverage: int = 5
    unfilled_order_expiration: int = 60
    limit_take_profit_price_delta_bps: int = 0

    position_mode: PositionMode = PositionMode.ONEWAY

    # Triple Barrier

    # Order settings
    amount_quote: Decimal = 50.0
    should_open_position_at_launch: bool = False
    tp_pct: Decimal = 2.8
    tp_position_pct: int = 24
    tp_expiration_min: int = 5
    tp_cooldown_min: int = 6
