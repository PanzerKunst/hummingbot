import os
from typing import Dict, List

import pandas_ta as ta  # noqa: F401
from pydantic import Field, validator

from hummingbot.core.data_type.common import OrderType, PositionMode
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig


class PkTwoConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    markets: Dict[str, List[str]] = {}
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []

    position_mode: PositionMode = PositionMode.ONEWAY

    # Triple Barrier Configuration
    time_limit_min: int = 1

    # No account on Binance perp
    binance_exchange: str = "binance"
    binance_pair: str = "CRV-USDT"
    kucoin_exchange: str = "kucoin_perpetual"
    kucoin_pair: str = "CRV-USDT"
    kucoin_leverage: int = 5  # TODO: 20

    oracle_price_history_length: int = 12
    price_delta_threshold_on_leading_exchange_bps: int = 15
    delta_with_best_bid_or_ask_bps: int = 100  # TODO: 1
    unfilled_order_time_limit: int = 2

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        take_profit_bps = self.price_delta_threshold_on_leading_exchange_bps * 0.8
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
