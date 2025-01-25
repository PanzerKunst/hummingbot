from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.ma_x_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import compute_softened_leverage
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Generate config file: create --script-config ma_x
# Start the bot: start --script ma_x.py --conf conf_ma_x_ANIME.yml
#                start --script ma_x.py --conf conf_ma_x_MELANIA.yml
#                start --script ma_x.py --conf conf_ma_x_TRUMP.yml
#                start --script ma_x.py --conf conf_ma_x_VINE.yml
# Quickstart script: -p=a -f ma_x.py -c conf_ma_x_ANIME.yml

ORDER_REF_MA_X: str = "MA-X"
SHORT_MA_LENGTH: int = 15  # 3 * 5
LONG_MA_LENGTH: int = 285  # 3 * 95


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()
        self.has_opened_at_launch: bool = not config.should_open_position_at_launch

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

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

        candles_df[f"SMA_{SHORT_MA_LENGTH}"] = candles_df.ta.sma(length=SHORT_MA_LENGTH)
        candles_df[f"EMA_{LONG_MA_LENGTH}"] = candles_df.ta.ema(length=LONG_MA_LENGTH)

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        # if not self.is_coin_still_tradable():
        #     self.logger().info("create_actions_proposal() > Stopping the bot as the coin is no longer tradable")
        #     HummingbotApplication.main_application().stop()

        self.create_actions_proposal_ma_x()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()
        self.stop_actions_proposal_ma_x()

        return []  # Always return []

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
                    f"SMA_{SHORT_MA_LENGTH}",
                    f"EMA_{LONG_MA_LENGTH}"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(20), table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Quote amount and Triple Barrier
    #

    def get_position_quote_amount(self, side: TradeType) -> Decimal:
        softened_leverage: int = compute_softened_leverage(self.config.leverage)
        amount_quote: Decimal = self.config.amount_quote * softened_leverage

        if side == TradeType.BUY:
            return amount_quote * Decimal(1.25)  # More, because closing an unprofitable Less position costs significantly less

        return amount_quote

    @staticmethod
    def get_triple_barrier() -> TripleBarrier:
        return TripleBarrier(
            open_order_type=OrderType.MARKET
        )

    #
    # MA-X start/stop action proposals
    #

    def create_actions_proposal_ma_x(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MA_X)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_ma_x_order(TradeType.SELL, active_orders):
            triple_barrier = self.get_triple_barrier()
            amount_quote = self.get_position_quote_amount(TradeType.SELL)
            self.create_order(TradeType.SELL, self.get_current_close(), triple_barrier, amount_quote, ORDER_REF_MA_X)

        if self.can_create_ma_x_order(TradeType.BUY, active_orders):
            triple_barrier = self.get_triple_barrier()
            amount_quote = self.get_position_quote_amount(TradeType.BUY)
            self.create_order(TradeType.BUY, self.get_current_close(), triple_barrier, amount_quote, ORDER_REF_MA_X)

    def can_create_ma_x_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        amount_quote = self.get_position_quote_amount(side)

        if not self.can_create_order(side, amount_quote, ORDER_REF_MA_X, 0):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if not self.has_opened_at_launch and not self.is_latest_short_ma_over_long():
                self.has_opened_at_launch = True
                self.logger().info(f"can_create_ma_x_order() > Opening initial MA-X Sell at {self.get_current_close()}")
                return True

            elif self.did_short_ma_cross_under_long():
                self.logger().info(f"can_create_ma_x_order() > Opening MA-X Sell at {self.get_current_close()}")
                return True

            return False

        if not self.has_opened_at_launch and self.is_latest_short_ma_over_long():
            self.has_opened_at_launch = True
            self.logger().info(f"can_create_ma_x_order() > Opening initial MA-X Buy at {self.get_current_close()}")
            return True

        elif self.did_short_ma_cross_over_long():
            self.logger().info(f"can_create_ma_x_order() > Opening MA-X Buy at {self.get_current_close()}")
            return True

        return False

    def stop_actions_proposal_ma_x(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MA_X)

        if len(filled_sell_orders) > 0:
            if self.did_short_ma_cross_over_long():
                self.logger().info(f"stop_actions_proposal_ma_x() > Closing MA-X Sell at {self.get_current_close()}")
                self.close_filled_orders(filled_sell_orders, OrderType.MARKET, CloseType.COMPLETED)

        if len(filled_buy_orders) > 0:
            if self.did_short_ma_cross_under_long():
                self.logger().info(f"stop_actions_proposal_ma_x() > Closing MA-X Buy at {self.get_current_close()}")
                self.close_filled_orders(filled_buy_orders, OrderType.MARKET, CloseType.COMPLETED)

    #
    # Getters on `self.processed_data[]`
    #

    def get_current_close(self) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[-1])

    def get_current_open(self) -> Decimal:
        open_series: pd.Series = self.processed_data["open"]
        return Decimal(open_series.iloc[-1])

    def get_current_low(self) -> Decimal:
        low_series: pd.Series = self.processed_data["low"]
        return Decimal(low_series.iloc[-1])

    def get_current_high(self) -> Decimal:
        high_series: pd.Series = self.processed_data["high"]
        return Decimal(high_series.iloc[-1])

    def get_current_ma(self, length: int, s_or_e: str) -> Decimal:
        return self._get_ma_at_index(length, -1, s_or_e)

    def get_latest_ma(self, length: int, s_or_e: str) -> Decimal:
        return self._get_ma_at_index(length, -2, s_or_e)

    def get_previous_ma(self, length: int, s_or_e: str) -> Decimal:
        return self._get_ma_at_index(length, -3, s_or_e)

    def _get_ma_at_index(self, length: int, index: int, s_or_e: str) -> Decimal:
        sma_series: pd.Series = self.processed_data[f"{s_or_e}MA_{length}"]
        return Decimal(sma_series.iloc[index])

    #
    # MA-X functions
    #

    # def is_coin_still_tradable(self) -> bool:
    #     launch_timestamp: float = iso_to_timestamp(self.config.coin_launch_date)
    #     start_of_today_timestamp = normalize_timestamp_to_midnight(self.get_market_data_provider_time())
    #     max_trade_duration = self.config.nb_days_trading_post_launch * 24 * 60 * 60  # seconds
    #
    #     return start_of_today_timestamp <= launch_timestamp + max_trade_duration

    def did_short_ma_cross_under_long(self) -> bool:
        return not self.is_latest_short_ma_over_long() and self.is_previous_short_ma_over_long()

    def did_short_ma_cross_over_long(self) -> bool:
        return self.is_latest_short_ma_over_long() and not self.is_previous_short_ma_over_long()

    def is_latest_short_ma_over_long(self) -> bool:
        latest_short_minus_long: Decimal = self.get_latest_ma(SHORT_MA_LENGTH, "S") - self.get_latest_ma(LONG_MA_LENGTH, "E")

        # TODO: remove
        self.logger().info(f"is_latest_short_ma_over_long() | latest_short_ma:{self.get_latest_ma(SHORT_MA_LENGTH, 'S')} | latest_long_ma:{self.get_latest_ma(LONG_MA_LENGTH, 'E')} | result:{latest_short_minus_long > 0}")

        return latest_short_minus_long > 0

    def is_previous_short_ma_over_long(self) -> bool:
        previous_short_minus_long: Decimal = self.get_previous_ma(SHORT_MA_LENGTH, "S") - self.get_previous_ma(LONG_MA_LENGTH, "E")

        # TODO: remove
        self.logger().info(f"is_previous_short_ma_over_long() | previous_short_ma:{self.get_previous_ma(SHORT_MA_LENGTH, 'S')} | previous_long_ma:{self.get_previous_ma(LONG_MA_LENGTH, 'E')} | result:{previous_short_minus_long > 0}")

        return previous_short_minus_long > 0

    # def is_current_price_over_short_ma(self) -> bool:
    #     current_price_minus_short_ma: Decimal = self.get_current_close() - self.get_current_ma(SHORT_MA_LENGTH, "S")
    #     return current_price_minus_short_ma > 0
