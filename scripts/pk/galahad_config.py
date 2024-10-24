import os
from decimal import Decimal
from typing import Dict, Set, List

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import PositionMode
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase


class GalahadConfig(StrategyV2ConfigBase):
    # Standard attributes START - avoid renaming
    markets: Dict[str, Set[str]] = {}

    candles_config: List[CandlesConfig] = [
        CandlesConfig(
            connector="binance_perpetual",
            interval="5m",
            max_records=52,
            trading_pair = "NEIRO-USDT"
        )
    ]

    controllers_config: List[str] = []
    config_update_interval: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    # Standard attributes END

    # Used by PkStrategy
    connector_name: str = "hyperliquid_perpetual"
    trading_pair: str = "kNEIRO-USD"
    total_amount_quote: int = Field(40, client_data=ClientFieldData(is_updatable=True))
    leverage: int = 5
    cooldown_time_min: int = Field(3, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))
    limit_take_profit_price_delta_bps: int = Field(2, client_data=ClientFieldData(is_updatable=True))

    position_mode: PositionMode = PositionMode.ONEWAY

    # Triple Barrier
    stop_loss_pct: Decimal = Field(1.5, client_data=ClientFieldData(is_updatable=True))
    trailing_stop_activation_pct: Decimal = Field(2, client_data=ClientFieldData(is_updatable=True))
    trailing_stop_close_delta_pct: Decimal = Field(1.5, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    macd_short: int = Field(12, client_data=ClientFieldData(is_updatable=True))
    macd_long: int = Field(26, client_data=ClientFieldData(is_updatable=True))
    macd_signal: int = Field(9, client_data=ClientFieldData(is_updatable=True))
    rsi_length: int = Field(20, client_data=ClientFieldData(is_updatable=True))
    psar_start: Decimal = Field(0.02, client_data=ClientFieldData(is_updatable=True))
    psar_increment: Decimal = Field(0.02, client_data=ClientFieldData(is_updatable=True))
    psar_max: Decimal = Field(0.2, client_data=ClientFieldData(is_updatable=True))

    # Order settings
    entry_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    trend_start_price_change_threshold_pct: Decimal = Field(0.9, client_data=ClientFieldData(is_updatable=True))
    trend_start_sell_latest_complete_candle_min_rsi: int = Field(45, client_data=ClientFieldData(is_updatable=True))
    trend_start_buy_latest_complete_candle_max_rsi: int = Field(55, client_data=ClientFieldData(is_updatable=True))
