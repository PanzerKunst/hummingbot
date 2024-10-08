import os
from decimal import Decimal
from typing import Dict, Set, List

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import PositionMode
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase


class MerlinConfig(StrategyV2ConfigBase):
    # Standard attributes - avoid renaming
    markets: Dict[str, Set[str]] = {}
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor
    controllers_config: List[str] = []
    config_update_interval: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # Used by PkStrategy
    connector_name: str = "gate_io"
    trading_pair: str = "XDC-USDT"
    total_amount_quote: int = Field(20, client_data=ClientFieldData(is_updatable=True))
    leverage: int = 1
    cooldown_time_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = Field(5, client_data=ClientFieldData(is_updatable=True))

    # Triple Barrier
    stop_loss_pct: Decimal = Field(0.3, client_data=ClientFieldData(is_updatable=True))
    take_profit_pct: Decimal = Field(0.3, client_data=ClientFieldData(is_updatable=True))
    filled_order_expiration_min: int = Field(1000, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    bbands_length_for_volatility: int = Field(2, client_data=ClientFieldData(is_updatable=True))
    bbands_std_dev_for_volatility: Decimal = Field(3.0, client_data=ClientFieldData(is_updatable=True))
    high_volatility_threshold: Decimal = Field(30.0, client_data=ClientFieldData(is_updatable=True))
    bbands_length_for_trend: int = Field(6, client_data=ClientFieldData(is_updatable=True))
    bbands_std_dev_for_trend: Decimal = Field(2.0, client_data=ClientFieldData(is_updatable=True))
    rsi_length: int = Field(50, client_data=ClientFieldData(is_updatable=True))
    rsi_top_edge: int = Field(75, client_data=ClientFieldData(is_updatable=True))
    rsi_bottom_edge: int = Field(25, client_data=ClientFieldData(is_updatable=True))
    volume_for_1_pct_volatility_adjustment: int = Field(100000, client_data=ClientFieldData(is_updatable=True))

    # Candles
    candles_connector: str = "gate_io"
    candles_interval: str = "1m"
    candles_length: int = 70

    # Order settings
    default_spread_pct: Decimal = Field(0.2, client_data=ClientFieldData(is_updatable=True))
