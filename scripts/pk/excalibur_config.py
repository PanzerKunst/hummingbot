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
    sma_cross_stop_loss_pct: Decimal = Field(0.7, client_data=ClientFieldData(is_updatable=True))
    mean_reversion_stop_loss_pct: Decimal = Field(0.3, client_data=ClientFieldData(is_updatable=True))
    mean_reversion_take_profit_pct: Decimal = Field(1.5, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    rsi_length: int = Field(20, client_data=ClientFieldData(is_updatable=True))
    sma_short: int = Field(75, client_data=ClientFieldData(is_updatable=True))
    sma_long: int = Field(300, client_data=ClientFieldData(is_updatable=True))

    # Order settings
    entry_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    max_price_delta_pct_with_sma_to_open_position: Decimal = Field(1.5, client_data=ClientFieldData(is_updatable=True))
    min_rsi_delta_for_sudden_change: int = Field(15, client_data=ClientFieldData(is_updatable=True))
    min_price_delta_pct_for_sudden_reversal_to_short_sma: Decimal = Field(0.75, client_data=ClientFieldData(is_updatable=True))
    first_pnl_pct_for_rsi_crash_or_spike_and_recovery_thresholds: Decimal = Field(2.5, client_data=ClientFieldData(is_updatable=True))
    second_pnl_pct_for_rsi_crash_or_spike_and_recovery_thresholds: Decimal = Field(3.25, client_data=ClientFieldData(is_updatable=True))
