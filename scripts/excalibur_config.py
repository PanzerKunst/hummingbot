import os
from decimal import Decimal
from typing import Dict, List, Set

from pydantic import Field

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
            trading_pair = "GOAT-USDT"
        )
    ]

    controllers_config: List[str] = []
    config_update_interval: int = 10
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    # Standard attributes END

    # Used by PkStrategy
    connector_name: str = "hyperliquid_perpetual"
    trading_pair: str = "GOAT-USD"
    leverage: int = 5
    unfilled_order_expiration_min: int = 1
    limit_take_profit_price_delta_bps: int = 0
    market_order_twap_count: int = 1
    market_order_twap_interval: int = 5

    position_mode: PositionMode = PositionMode.ONEWAY

    # Triple Barrier
    ma_cross_stop_loss_pct: Decimal = 3.0

    # Technical analysis
    rsi_short: int = 20
    rsi_long: int = 40
    sma_short: int = 75
    sma_long: int = 300
    stoch_short_k_length: int = 40
    stoch_short_k_smoothing: int = 8
    stoch_short_d_smoothing: int = 6
    stoch_long_k_length: int = 80
    stoch_long_k_smoothing: int = 16
    stoch_long_d_smoothing: int = 6

    # Order settings
    amount_quote_ma_cross: int = 30
    amount_quote_mr: int = 30
    entry_price_delta_bps: int = 0
    max_price_delta_pct_with_short_ma_to_open: Decimal = 2.5
    min_price_delta_pct_for_sudden_reversal_to_short_ma: Decimal = 2.0
    rsi_peak_threshold_to_open_mr: int = 67
    rsi_bottom_threshold_to_open_mr: int = 34
    stoch_peak_threshold_to_open_mr: int = 90
    stoch_bottom_threshold_to_open_mr: int = 10
