import os
from decimal import Decimal
from typing import Dict, Set, List

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import PositionMode
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase


class ArthurConfig(StrategyV2ConfigBase):
    # Standard attributes START - avoid renaming
    markets: Dict[str, Set[str]] = {}

    candles_config: List[CandlesConfig] = [
        CandlesConfig(
            connector="binance_perpetual",
            interval="1m",
            max_records=40,
            trading_pair = "POPCAT-USDT"
        ),
        CandlesConfig(
            connector="bybit_perpetual",
            interval="1m",
            max_records=40,
            trading_pair = "POPCAT-USDT"
        ),
        CandlesConfig(
            connector="okx_perpetual",
            interval="1m",
            max_records=40,
            trading_pair="POPCAT-USDT"
        )
    ]

    controllers_config: List[str] = []
    config_update_interval: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    # Standard attributes END

    # Used by PkStrategy
    connector_name: str = "okx_perpetual"
    trading_pair: str = "POPCAT-USDT"
    total_amount_quote: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    leverage: int = 20
    cooldown_time_min: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))
    limit_take_profit_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))

    position_mode: PositionMode = PositionMode.HEDGE

    # Triple Barrier
    filled_trend_start_order_expiration: int = Field(45, client_data=ClientFieldData(is_updatable=True))
    filled_trend_reversal_order_expiration: int = Field(10 * 60, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    rsi_length: int = Field(20, client_data=ClientFieldData(is_updatable=True))

    # Order settings
    entry_price_delta_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))
    trend_start_price_change_threshold_pct: Decimal = Field(0.9, client_data=ClientFieldData(is_updatable=True))
    trend_start_sell_latest_complete_candle_min_rsi: int = Field(40, client_data=ClientFieldData(is_updatable=True))
    trend_start_buy_latest_complete_candle_max_rsi: int = Field(60, client_data=ClientFieldData(is_updatable=True))
    trend_start_volume_mul: int = Field(4, client_data=ClientFieldData(is_updatable=True))
    trend_reversal_sell_min_rsi: int = Field(75, client_data=ClientFieldData(is_updatable=True))
    trend_reversal_buy_max_rsi: int = Field(25, client_data=ClientFieldData(is_updatable=True))
