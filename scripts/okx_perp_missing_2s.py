import os
from decimal import Decimal
from typing import Dict, List

import pandas as pd
import pandas_ta as ta  # noqa: F401
from pydantic import Field

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import PositionMode
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction

# Generate config file: create --script-config okx_perp_missing_2s
# Start the bot: start --script okx_perp_missing_2s.py --conf conf_okx_perp_missing_2s_FARTCOIN_Hyperliquid.yml
#                start --script okx_perp_missing_2s.py --conf conf_okx_perp_missing_2s_FARTCOIN_OKX.yml

MA_LENGTH: int = 20


class OkxMissingTwoSecConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    markets: Dict[str, List[str]] = {}
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []
    exchange: str = "hyperliquid_perpetual"
    trading_pair: str = "FARTCOIN-USD"
    candles_exchange: str = "binance_perpetual"
    candles_pair: str = "FARTCOIN-USDT"
    candles_interval: str = "1m"
    candles_length: int = 60
    order_amount_quote: Decimal = 30
    leverage: int = 3
    position_mode: PositionMode = PositionMode.ONEWAY


class OkxMissingTwoSec(StrategyV2Base):
    account_config_set = False

    @classmethod
    def init_markets(cls, config: OkxMissingTwoSecConfig):
        cls.markets = {config.exchange: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: OkxMissingTwoSecConfig):
        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.candles_exchange,
                trading_pair=config.candles_pair,
                interval=config.candles_interval,
                max_records=config.candles_length + 10
            ))
        super().__init__(connectors, config)
        self.config = config

    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        :param clock: Clock to use.
        :param timestamp: Current time.
        """
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        if not self.account_config_set:
            for connector_name, connector in self.connectors.items():
                if self.is_perpetual(connector_name):
                    connector.set_position_mode(self.config.position_mode)
                    for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                        connector.set_leverage(trading_pair, self.config.leverage)
            self.account_config_set = True

    def update_processed_data(self):
        candles_config = self.config.candles_config[0]

        candles_df = self.market_data_provider.get_candles_df(connector_name=candles_config.connector,
                                                              trading_pair=candles_config.trading_pair,
                                                              interval=candles_config.interval,
                                                              max_records=candles_config.max_records)
        num_rows = candles_df.shape[0]

        if num_rows == 0:
            return

        candles_df["index"] = candles_df["timestamp"]
        candles_df.set_index("index", inplace=True)

        candles_df["timestamp_iso"] = pd.to_datetime(candles_df["timestamp"], unit="s")

        candles_df[f"EMA_{MA_LENGTH}"] = candles_df.ta.ema(length=MA_LENGTH)

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        self.logger().info(f"create_actions_proposal() | close:{self.get_current_close()}")

        create_actions = []
        return create_actions

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        stop_actions = []
        return stop_actions

    def format_status(self) -> str:
        original_status = super().format_status()
        custom_status = ["\n"]

        if self.ready_to_trade:
            if not self.processed_data.empty:
                columns_to_display = [
                    "timestamp_iso",
                    "low",
                    "high",
                    "close",
                    "volume",
                    f"EMA_{MA_LENGTH}"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(20), table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Getters on `self.processed_data[]`
    #

    def get_current_close(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[-1])
