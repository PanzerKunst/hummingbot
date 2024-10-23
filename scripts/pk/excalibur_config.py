import os
from decimal import Decimal
from typing import Dict, Set, List

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import PositionMode
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase


class ExcaliburConfig(StrategyV2ConfigBase):
    # Standard attributes START - avoid renaming
    markets: Dict[str, Set[str]] = {}

    candles_config: List[CandlesConfig] = [
        CandlesConfig(
            connector="binance_perpetual",
            interval="5m",
            max_records=90,
            trading_pair = "POPCAT-USDT"
        )
    ]

    controllers_config: List[str] = []
    config_update_interval: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    # Standard attributes END

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
    stop_loss_pct: Decimal = Field(2, client_data=ClientFieldData(is_updatable=True))
    trailing_stop_activation_pct: Decimal = Field(2, client_data=ClientFieldData(is_updatable=True))
    trailing_stop_close_delta_pct: Decimal = Field(2, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    rsi_length: int = Field(20, client_data=ClientFieldData(is_updatable=True))
    sma_short: int = Field(20, client_data=ClientFieldData(is_updatable=True))
    sma_long: int = Field(80, client_data=ClientFieldData(is_updatable=True))
    rsi_length_for_open_order: int = Field(5, client_data=ClientFieldData(is_updatable=True))

    # Order settings
    entry_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    take_profit_sell_rsi_threshold: int = Field(27, client_data=ClientFieldData(is_updatable=True))
    take_profit_buy_rsi_threshold: int = Field(73, client_data=ClientFieldData(is_updatable=True))
    filled_position_min_duration_min: int = Field(20, client_data=ClientFieldData(is_updatable=True))
    min_rsi_to_open_sell_order: int = Field(60, client_data=ClientFieldData(is_updatable=True))
    max_rsi_to_open_buy_order: int = Field(40, client_data=ClientFieldData(is_updatable=True))
