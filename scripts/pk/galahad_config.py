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
    trading_pair: str = "NEIRO-USDT"
    total_amount_quote: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    leverage: int = 20
    cooldown_time_min: int = Field(3, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))
    limit_take_profit_price_delta_bps: int = Field(2, client_data=ClientFieldData(is_updatable=True))

    position_mode: PositionMode = PositionMode.HEDGE

    # Triple Barrier
    filled_order_expiration_min: int = Field(10, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    macd_short: int = Field(12, client_data=ClientFieldData(is_updatable=True))
    macd_long: int = Field(26, client_data=ClientFieldData(is_updatable=True))
    macd_signal: int = Field(9, client_data=ClientFieldData(is_updatable=True))
    rsi_length: int = Field(20, client_data=ClientFieldData(is_updatable=True))
    rsi_top_edge: int = Field(68, client_data=ClientFieldData(is_updatable=True))
    rsi_bottom_edge: int = Field(32, client_data=ClientFieldData(is_updatable=True))
    psar_af: Decimal = Field(0.02, client_data=ClientFieldData(is_updatable=True))
    psar_max_af: Decimal = Field(0.2, client_data=ClientFieldData(is_updatable=True))
    bbands_length: int = Field(6, client_data=ClientFieldData(is_updatable=True))
    bbands_std_dev: Decimal = Field(2.0, client_data=ClientFieldData(is_updatable=True))

    # Candles
    candles_connector: str = "binance_perpetual"
    candles_pair: str = "NEIRO-USDT"
    candles_interval: str = "1m"
    candles_length: int = 52

    # Order settings
    entry_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    min_bbb_instant_volatility: Decimal = Field(3, client_data=ClientFieldData(is_updatable=True))
    min_bbb_past_volatility: Decimal = Field(1.5, client_data=ClientFieldData(is_updatable=True))
