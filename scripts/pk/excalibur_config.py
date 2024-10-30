import os
from decimal import Decimal
from typing import Dict, List, Set

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
            max_records=70,
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
    market_order_twap_count: int = 3
    market_order_twap_interval: int = 5

    position_mode: PositionMode = PositionMode.ONEWAY

    # Triple Barrier
    stop_loss_pct: Decimal = Field(1.5, client_data=ClientFieldData(is_updatable=True))
    trailing_stop_activation_pct: Decimal = Field(1.5, client_data=ClientFieldData(is_updatable=True))
    trailing_stop_close_delta_pct: Decimal = Field(1.1, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    rsi_length: int = Field(20, client_data=ClientFieldData(is_updatable=True))
    sma_short: int = Field(15, client_data=ClientFieldData(is_updatable=True))
    sma_long: int = Field(60, client_data=ClientFieldData(is_updatable=True))

    # Order settings
    entry_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    min_rsi_to_open_sell_order: int = Field(35, client_data=ClientFieldData(is_updatable=True))
    max_rsi_to_open_buy_order: int = Field(65, client_data=ClientFieldData(is_updatable=True))
