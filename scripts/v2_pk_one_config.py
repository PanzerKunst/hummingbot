import os
from typing import Dict, List

import pandas_ta as ta  # noqa: F401
from pydantic import Field, validator

from hummingbot.core.data_type.common import OrderType, PositionMode
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig


class PkOneConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    markets: Dict[str, List[str]] = {}

    # Candles
    candles_exchange: str = "binance"
    candles_pair: str = "CRV-USDT"
    candles_interval: str = "1s"
    candles_length: int = 10

    candles_config: List[CandlesConfig] = [CandlesConfig(
        connector=candles_exchange,
        trading_pair=candles_pair,
        interval=candles_interval,
        max_records=candles_length
    )]

    controllers_config: List[str] = []

    # No account on Binance perp
    bitget_exchange: str = "bitget_perpetual"
    bitget_pair: str = "CRV-USDT"
    bitget_leverage: int = 20
    bybit_exchange: str = "bybit_perpetual"
    bybit_pair: str = "CRV-USDT"
    bybit_leverage: int = 20
    gate_io_exchange: str = "gate_io_perpetual"
    gate_io_pair: str = "CRV-USDT"
    gate_io_leverage: int = 20
    htx_exchange: str = "gate_io_perpetual"
    htx_pair: str = "CRV-USDT"
    htx_leverage: int = 20
    hyperliquid_exchange: str = "hyperliquid_perpetual"
    hyperliquid_pair: str = "CRV-USD"
    hyperliquid_leverage: int = 10
    # Kraken perp connector unsupported
    kucoin_exchange: str = "kucoin_perpetual"
    kucoin_pair: str = "CRV-USDT"
    kucoin_leverage: int = 5  # TODO: 20
    okx_exchange: str = "okx_perpetual"
    okx_pair: str = "CRV-USDT"
    okx_leverage: int = 20

    position_mode: PositionMode = PositionMode.ONEWAY
    candles_price_delta_threshold_bps: int = 15
    candles_base_volume_threshold: int = 40000
    delta_with_best_bid_or_ask_bps: int = 1

    # Triple Barrier Configuration
    time_limit_min: int = 1

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        take_profit_bps = self.candles_price_delta_threshold_bps * 0.8
        stop_loss_bps = take_profit_bps

        return TripleBarrierConfig(
            stop_loss=stop_loss_bps / 10000,
            take_profit=take_profit_bps / 10000,
            time_limit=self.time_limit_min * 60,
            open_order_type=OrderType.LIMIT,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,  # Only market orders are supported for time_limit and stop_loss
            time_limit_order_type=OrderType.MARKET  # Only market orders are supported for time_limit and stop_loss
        )

    @validator("position_mode", pre=True, allow_reuse=True)
    def validate_position_mode(cls, v: str) -> PositionMode:
        if v.upper() in PositionMode.__members__:
            return PositionMode[v.upper()]
        raise ValueError(f"Invalid position mode: {v}. Valid options are: {', '.join(PositionMode.__members__)}")
