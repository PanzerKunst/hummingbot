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
            interval="1m",
            max_records=330,
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
    total_amount_quote: int = Field(30, client_data=ClientFieldData(is_updatable=True))
    leverage: int = 5
    unfilled_order_expiration_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))
    limit_take_profit_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    market_order_twap_count: int = 1
    market_order_twap_interval: int = 5

    position_mode: PositionMode = PositionMode.ONEWAY

    # Triple Barrier
    sma_cross_stop_loss_pct: Decimal = Field(2.25, client_data=ClientFieldData(is_updatable=True))
    mean_reversion_stop_loss_pct: Decimal = Field(0.75, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    rsi_length: int = Field(20, client_data=ClientFieldData(is_updatable=True))
    rsi_mr_length: int = Field(40, client_data=ClientFieldData(is_updatable=True))
    sma_short: int = Field(75, client_data=ClientFieldData(is_updatable=True))
    sma_long: int = Field(300, client_data=ClientFieldData(is_updatable=True))

    # Order settings
    entry_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    max_price_delta_pct_with_short_sma_to_open: Decimal = Field(1.5, client_data=ClientFieldData(is_updatable=True))
    min_price_delta_pct_for_sudden_reversal_to_short_sma: Decimal = Field(1.0, client_data=ClientFieldData(is_updatable=True))
    rsi_crash_bottom_threshold: Decimal = Field(38.0, client_data=ClientFieldData(is_updatable=True))
    rsi_crash_recovery_threshold: Decimal = Field(39.5, client_data=ClientFieldData(is_updatable=True))
    rsi_spike_peak_threshold: Decimal = Field(64, client_data=ClientFieldData(is_updatable=True))
    rsi_spike_recovery_threshold: Decimal = Field(62.5, client_data=ClientFieldData(is_updatable=True))
