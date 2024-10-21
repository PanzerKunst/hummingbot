import os
from decimal import Decimal
from typing import Dict, Set, List

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import PositionMode
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase


class ExcaliburConfig(StrategyV2ConfigBase):
    # Standard attributes - avoid renaming
    markets: Dict[str, Set[str]] = {}
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor
    controllers_config: List[str] = []
    config_update_interval: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # Used by PkStrategy
    connector_name: str = "hyperliquid_perpetual"
    trading_pair: str = "POPCAT-USD"
    total_amount_quote: int = Field(40, client_data=ClientFieldData(is_updatable=True))
    leverage: int = 5
    cooldown_time_min: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))
    limit_take_profit_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))

    position_mode: PositionMode = PositionMode.ONEWAY

    # Triple Barrier
    stop_loss_pct: Decimal = Field(0.6, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    sma_short: int = Field(5, client_data=ClientFieldData(is_updatable=True))
    sma_long: int = Field(20, client_data=ClientFieldData(is_updatable=True))

    # Candles
    candles_connector: str = "binance_perpetual"
    candles_pair: str = "POPCAT-USDT"
    candles_interval: str = "5m"
    candles_length: int = 40

    # Order settings
    entry_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
