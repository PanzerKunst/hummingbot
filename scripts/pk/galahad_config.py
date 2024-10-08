import os
from decimal import Decimal
from typing import Dict, Set, List

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import PositionMode
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase


class GalahadConfig(StrategyV2ConfigBase):
    # Standard attributes - avoid renaming
    markets: Dict[str, Set[str]] = {}
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor
    controllers_config: List[str] = []
    config_update_interval: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # Used by PkStrategy
    connector_name: str = "okx_perpetual"
    trading_pair: str = "POPCAT-USDT"
    total_amount_quote: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    leverage: int = 20
    cooldown_time_min: int = Field(5, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))

    position_mode: PositionMode = PositionMode.HEDGE

    # Triple Barrier
    stop_loss_pct: Decimal = Field(1.0, client_data=ClientFieldData(is_updatable=True))
    take_profit_pct: Decimal = Field(1.0, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    macd_short: int = Field(12, client_data=ClientFieldData(is_updatable=True))
    macd_long: int = Field(26, client_data=ClientFieldData(is_updatable=True))
    macd_signal: int = Field(9, client_data=ClientFieldData(is_updatable=True))

    # Candles
    candles_connector: str = "okx_perpetual"
    candles_interval: str = "1m"
    candles_length: int = 52

    # Order settings
    delta_with_ref_price_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
