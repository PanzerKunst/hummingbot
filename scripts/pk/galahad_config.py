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
    psar_start: Decimal = Field(0.02, client_data=ClientFieldData(is_updatable=True))
    psar_increment: Decimal = Field(0.02, client_data=ClientFieldData(is_updatable=True))
    psar_max: Decimal = Field(0.2, client_data=ClientFieldData(is_updatable=True))

    # Candles
    candles_connector: str = "binance_perpetual"
    candles_pair: str = "NEIRO-USDT"
    candles_interval: str = "1m"
    candles_length: int = 52

    # Order settings
    entry_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    trend_start_price_change_threshold_pct: Decimal = Field(0.9, client_data=ClientFieldData(is_updatable=True))
    trend_start_sell_latest_complete_candle_min_rsi: int = Field(45, client_data=ClientFieldData(is_updatable=True))
    trend_start_buy_latest_complete_candle_max_rsi: int = Field(55, client_data=ClientFieldData(is_updatable=True))
