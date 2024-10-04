import os
from decimal import Decimal
from typing import Dict, List, Set, Optional

import pandas as pd
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionMode, TradeType
from hummingbot.core.event.events import SellOrderCreatedEvent, BuyOrderCreatedEvent, OrderFilledEvent
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase, StrategyV2Base
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.tracked_order_details import TrackedOrderDetails


class ArthurConfig(StrategyV2ConfigBase):
    # Standard attributes - avoid renaming
    markets: Dict[str, Set[str]] = {}
    candles_config: List[CandlesConfig] = []  # Initialized in the constructor
    controllers_config: List[str] = []
    config_update_interval: int = Field(10, client_data=ClientFieldData(is_updatable=True))
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))

    # Used by PkStrategy
    connector_name: str = "okx_perpetual"
    trading_pair: str = "POPCAT-USDT"
    total_amount_quote: int = Field(5, client_data=ClientFieldData(is_updatable=True))
    leverage: int = 20
    cooldown_time_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))
    unfilled_order_expiration_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))

    position_mode: PositionMode = PositionMode.HEDGE

    # Triple Barrier
    stop_loss_pct: Decimal = Field(0.7, client_data=ClientFieldData(is_updatable=True))
    take_profit_pct: Decimal = Field(0.7, client_data=ClientFieldData(is_updatable=True))
    filled_order_expiration_min: int = Field(1, client_data=ClientFieldData(is_updatable=True))

    # Technical analysis
    bbands_length_for_volatility: int = Field(2, client_data=ClientFieldData(is_updatable=True))
    bbands_std_dev_for_volatility: Decimal = Field(3.0, client_data=ClientFieldData(is_updatable=True))
    high_volatility_threshold: Decimal = Field(3.0, client_data=ClientFieldData(is_updatable=True))
    rsi_length: int = Field(20, client_data=ClientFieldData(is_updatable=True))

    # Candles
    candles_connector: str = "okx_perpetual"
    candles_interval: str = "1m"
    candles_length: int = 40

    # Maker orders settings
    delta_with_mid_price_bps: int = Field(0, client_data=ClientFieldData(is_updatable=True))


# Generate config file: create --script-config arthur
# Start the bot: start --script arthur.arthur.py --conf conf_arthur_POPCAT.yml
# Quickstart script: -p=a -f arthur.arthur.py -c conf_arthur_POPCAT.yml


class ArthurStrategy(StrategyV2Base):
    @classmethod
    def init_markets(cls, config: ArthurConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ArthurConfig):
        super().__init__(connectors, config)
        self.config = config

        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.trading_pair,
                interval=config.candles_interval,
                max_records=config.candles_length
            ))

        self.processed_data = pd.DataFrame(columns=["timestamp", "timestamp_iso", "bbb_for_volatility", "normalized_rsi"])

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

        self.pk_strat = PkStrategy(self)

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def get_triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=Decimal(self.config.stop_loss_pct / 100),
            take_profit=Decimal(self.config.take_profit_pct / 100),
            time_limit=self.config.filled_order_expiration_min * 60,
            open_order_type=OrderType.LIMIT,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,  # Only market orders are supported for time_limit and stop_loss
            time_limit_order_type=OrderType.MARKET  # Only market orders are supported for time_limit and stop_loss
        )

    def update_processed_data(self):
        candles_config = self.config.candles_config[0]

        candles_df = self.market_data_provider.get_candles_df(connector_name=candles_config.connector,
                                                              trading_pair=candles_config.trading_pair,
                                                              interval=candles_config.interval,
                                                              max_records=candles_config.max_records)
        candles_df["index"] = candles_df["timestamp"]
        candles_df.set_index("index", inplace=True)

        candles_df["timestamp_iso"] = pd.to_datetime(candles_df["timestamp"], unit="s")

        bbands_for_volatility = candles_df.ta.bbands(length=self.config.bbands_length_for_volatility, std=self.config.bbands_std_dev_for_volatility)
        candles_df["bbb_for_volatility"] = bbands_for_volatility[f"BBB_{self.config.bbands_length_for_volatility}_{self.config.bbands_std_dev_for_volatility}"]

        rsi = candles_df.ta.rsi(length=self.config.rsi_length)
        candles_df["normalized_rsi"] = rsi.apply(self.normalize_rsi)

        self.update_indicators(candles_df)

    def update_indicators(self, df: pd.DataFrame):
        rows_to_add = []

        for _, row in df.iterrows():
            timestamp = row["timestamp"]
            timestamp_iso = row["timestamp_iso"]
            bbb_for_volatility = row["bbb_for_volatility"]
            normalized_rsi = row["normalized_rsi"]

            if pd.notna(bbb_for_volatility) and pd.notna(normalized_rsi):
                if self._get_indicators_for_timestamp(timestamp):
                    self.processed_data.loc[timestamp, ["bbb_for_volatility", "normalized_rsi"]] = [bbb_for_volatility, normalized_rsi]
                else:
                    rows_to_add.append({
                        "timestamp": timestamp,
                        "timestamp_iso": timestamp_iso,
                        "bbb_for_volatility": bbb_for_volatility,
                        "normalized_rsi": normalized_rsi
                    })

        if len(rows_to_add) > 0:
            new_rows = pd.DataFrame(rows_to_add)
            self.processed_data = pd.concat([self.processed_data, new_rows], ignore_index=True)
            self.processed_data["index"] = self.processed_data["timestamp"]
            self.processed_data.set_index("index", inplace=True)

    def _get_indicators_for_timestamp(self, timestamp: float) -> Optional:
        matching_row = self.processed_data.query(f"timestamp == {timestamp}")

        if not matching_row.empty:
            return matching_row.iloc[0]
        else:
            return None

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        if self.is_high_volatility():
            return []

        unfilled_sell_orders, unfilled_buy_orders = self.pk_strat.get_unfilled_tracked_orders_by_side()

        if self.can_create_order(TradeType.SELL, unfilled_sell_orders):
            delta_pct = Decimal(3)
            entry_price: Decimal = self.pk_strat.get_mid_price() * (1 + delta_pct / 100)
            self.pk_strat.create_order(TradeType.SELL, entry_price)

        if self.can_create_order(TradeType.BUY, unfilled_buy_orders):
            delta_pct = Decimal(5)
            entry_price: Decimal = self.pk_strat.get_mid_price() * (1 - delta_pct / 100)
            self.pk_strat.create_order(TradeType.BUY, entry_price)

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        self.pk_strat.check_orders()

        if self.is_high_volatility():
            self.logger().info(f"##### is_high_volatility -> Stopping unfilled executors #####")
            unfilled_sell_orders, unfilled_buy_orders = self.pk_strat.get_unfilled_tracked_orders_by_side()

            for unfilled_order in unfilled_sell_orders + unfilled_buy_orders:
                self.pk_strat.cancel_order(unfilled_order)

        return []  # Always return []

    def format_status(self) -> str:
        original_status = super().format_status()
        custom_status = []

        if self.ready_to_trade:
            if not self.processed_data.empty:
                columns_to_display = [
                    "timestamp_iso",
                    "close",
                    "bbb_for_volatility",
                    "normalized_rsi"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(self.config.rsi_length), table_format="psql", ))

        return original_status + "\n".join(custom_status)

    #
    # Custom functions potentially interesting for other controllers
    #

    def can_create_order(self, side: TradeType, unfilled_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.pk_strat.can_create_order(side):
            return False

        if len(unfilled_tracked_orders) > 0:
            return False

        return True

    def did_create_sell_order(self, created_event: SellOrderCreatedEvent):
        self.pk_strat.did_create_sell_order(created_event)

    def did_create_buy_order(self, created_event: BuyOrderCreatedEvent):
        self.pk_strat.did_create_buy_order(created_event)

    def did_fill_order(self, filled_event: OrderFilledEvent):
        self.pk_strat.did_fill_order(filled_event)

    #
    # Custom functions specific to this controller
    #

    @staticmethod
    def normalize_rsi(rsi: float) -> Decimal:
        return Decimal(rsi * 2 - 100)

    def get_latest_bbb(self) -> Decimal:
        bbb_series: pd.Series = self.processed_data["bbb_for_volatility"]
        bbb_previous_full_minute = Decimal(bbb_series.iloc[-2])
        bbb_current_incomplete_minute = Decimal(bbb_series.iloc[-1])
        return max(bbb_previous_full_minute, bbb_current_incomplete_minute)

    def is_high_volatility(self) -> bool:
        # TODO: remove
        self.logger().info(f"is_high_volatility() | latest_bbb: {self.get_latest_bbb()}")

        return self.get_latest_bbb() > self.config.high_volatility_threshold

    def get_latest_normalized_rsi(self) -> Decimal:
        rsi_series: pd.Series = self.processed_data["normalized_rsi"]
        return Decimal(rsi_series.iloc[-1])
