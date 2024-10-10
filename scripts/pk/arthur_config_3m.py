import os
from decimal import Decimal
from typing import Dict, Set, List

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import PositionMode
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase


class ArthurConfig3m(StrategyV2ConfigBase):
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
    cooldown_time_min: int = Field(30, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))
    limit_take_profit_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))

    position_mode: PositionMode = PositionMode.HEDGE

    # Triple Barrier
    filled_order_expiration_min: int = Field(30, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    rsi_length: int = Field(20, client_data=ClientFieldData(is_updatable=True))

    # Candles
    candles_connector: str = "okx_perpetual"
    candles_interval: str = "3m"
    candles_length: int = 40

    # Order settings
    entry_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    trend_reversal_candle_height_threshold_pct: Decimal = Field(1.0, client_data=ClientFieldData(is_updatable=True))
    trend_reversal_rsi_threshold_sell: int = Field(74, client_data=ClientFieldData(is_updatable=True))
    trend_reversal_rsi_threshold_buy: int = Field(26, client_data=ClientFieldData(is_updatable=True))
    trend_reversal_nb_seconds_to_calculate_end_of_trend: int = Field(30, client_data=ClientFieldData(is_updatable=True))
    trend_start_candle_height_threshold_pct: Decimal = Field(1.3, client_data=ClientFieldData(is_updatable=True))
