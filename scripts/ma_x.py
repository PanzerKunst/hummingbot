from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.ma_x_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import compute_take_profit_price
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Generate config file: create --script-config ma_x
# Start the bot: start --script ma_x.py --conf conf_ma_x_ANIME.yml
#                start --script ma_x.py --conf conf_ma_x_MELANIA.yml
#                start --script ma_x.py --conf conf_ma_x_RUNE.yml
#                start --script ma_x.py --conf conf_ma_x_TRUMP.yml
#                start --script ma_x.py --conf conf_ma_x_VINE.yml
#                start --script ma_x.py --conf conf_ma_x_VVV.yml
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

        candles_df[f"EMA_{SHORT_MA_LENGTH}"] = candles_df.ta.ema(length=SHORT_MA_LENGTH)
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
                    f"EMA_{SHORT_MA_LENGTH}",
                    f"EMA_{LONG_MA_LENGTH}"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display].tail(20), table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Quote amount and Triple Barrier
    #

    def get_position_quote_amount(self, side: TradeType) -> Decimal:
        if side == TradeType.SELL:
            return self.config.amount_quote * Decimal(0.75)  # Less, because closing an unprofitable Short position costs significantly more

        return self.config.amount_quote

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
            self.tp_count = 0

        if self.can_create_ma_x_order(TradeType.BUY, active_orders):
            triple_barrier = self.get_triple_barrier()
            amount_quote = self.get_position_quote_amount(TradeType.BUY)
            self.create_order(TradeType.BUY, self.get_current_close(), triple_barrier, amount_quote, ORDER_REF_MA_X)
            self.tp_count = 0

    def can_create_ma_x_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        amount_quote = self.get_position_quote_amount(side)

        if not self.can_create_order(side, amount_quote, ORDER_REF_MA_X, 0):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if self.is_price_too_far_from_long_ma():
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

    def did_fill_order(self, filled_event: OrderFilledEvent):
        position = filled_event.position

        if not position or position == PositionAction.CLOSE.value:
            self.logger().info(f"did_fill_order | position:{position}")

            for take_profit_limit_order in self.take_profit_limit_orders:
                if take_profit_limit_order.order_id == filled_event.order_id:
                    self.logger().info(f"did_fill_order | Take Profit price reached for tracked order:{take_profit_limit_order.tracked_order}")

                    take_profit_limit_order.filled_amount = filled_event.amount
                    take_profit_limit_order.filled_at = filled_event.timestamp
                    take_profit_limit_order.filled_price = filled_event.price

                    self.logger().info(f"did_fill_order | amount:{filled_event.amount} at price:{filled_event.price}")

                    self.tp_count += 1

                    for tracked_order in self.tracked_orders:
                        if tracked_order.order_id == take_profit_limit_order.tracked_order.order_id:
                            tracked_order.filled_amount -= filled_event.amount
                            self.logger().info(f"did_fill_order | tracked_order.filled_amount reduced to:{tracked_order.filled_amount}")

                            if tracked_order.filled_amount == 0:
                                self.logger().info("did_fill_order > tracked_order.filled_amount == 0! Closing it")
                                tracked_order.terminated_at = filled_event.timestamp
                                tracked_order.close_type = CloseType.TAKE_PROFIT

                            if not tracked_order.terminated_at and self.tp_count < self.config.max_take_profits:
                                # We create the next TP Limit order
                                take_profit_delta = self.config.tp_threshold_pct / 100
                                take_profit_amount: Decimal = filled_event.amount  # Same amount as the TP order which was just filled

                                take_profit_price = compute_take_profit_price(tracked_order.side, filled_event.price, take_profit_delta)
                                self.logger().info(f"did_fill_order > Creating Limit TP order number {self.tp_count + 1}")
                                self.logger().info(f"did_fill_order | take_profit_delta:{take_profit_delta} | take_profit_price:{take_profit_price} | take_profit_amount:{take_profit_amount}")
                                self.create_tp_limit_order(tracked_order, take_profit_amount, take_profit_price)

                            break

                    break

            return

        for tracked_order in self.tracked_orders:
            if tracked_order.order_id == filled_event.order_id:
                tracked_order.filled_amount += filled_event.amount
                tracked_order.last_filled_at = filled_event.timestamp
                tracked_order.last_filled_price = filled_event.price

                self.logger().info(f"did_fill_order | amount:{filled_event.amount} at price:{filled_event.price}")

                take_profit_delta = self.config.tp_threshold_pct / 100
                take_profit_amount: Decimal = filled_event.amount * Decimal(self.config.max_take_profits / 100)

                take_profit_price = compute_take_profit_price(tracked_order.side, filled_event.price, take_profit_delta)
                self.logger().info(f"did_fill_order > Creating Limit TP order number {self.tp_count + 1}")
                self.logger().info(f"did_fill_order | take_profit_delta:{take_profit_delta} | take_profit_price:{take_profit_price} | take_profit_amount:{take_profit_amount}")
                self.create_tp_limit_order(tracked_order, take_profit_amount, take_profit_price)

                break

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

    def get_current_ma(self, length: int) -> Decimal:
        return self._get_ma_at_index(length, -1)

    def get_latest_ma(self, length: int) -> Decimal:
        return self._get_ma_at_index(length, -2)

    def get_previous_ma(self, length: int) -> Decimal:
        return self._get_ma_at_index(length, -3)

    def _get_ma_at_index(self, length: int, index: int) -> Decimal:
        sma_series: pd.Series = self.processed_data[f"EMA_{length}"]
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

    def is_price_too_far_from_long_ma(self) -> bool:
        current_price: Decimal = self.get_current_close()
        current_long_ma: Decimal = self.get_current_ma(LONG_MA_LENGTH)

        price_delta_pct: Decimal = abs(current_long_ma - current_price) / current_price * 100
        is_too_far: bool = price_delta_pct > self.config.max_delta_pct_between_price_and_long_ma

        if is_too_far:
            self.logger().info(f"is_price_too_far_from_long_ma() | price_delta_pct:{price_delta_pct}")

        return is_too_far

    def did_short_ma_cross_under_long(self) -> bool:
        return not self.is_latest_short_ma_over_long() and self.is_previous_short_ma_over_long()

    def did_short_ma_cross_over_long(self) -> bool:
        return self.is_latest_short_ma_over_long() and not self.is_previous_short_ma_over_long()

    def is_latest_short_ma_over_long(self) -> bool:
        latest_short_minus_long: Decimal = self.get_latest_ma(SHORT_MA_LENGTH) - self.get_latest_ma(LONG_MA_LENGTH)
        return latest_short_minus_long > 0

    def is_previous_short_ma_over_long(self) -> bool:
        previous_short_minus_long: Decimal = self.get_previous_ma(SHORT_MA_LENGTH) - self.get_previous_ma(LONG_MA_LENGTH)
        return previous_short_minus_long > 0

    # def is_current_price_over_short_ma(self) -> bool:
    #     current_price_minus_short_ma: Decimal = self.get_current_close() - self.get_current_ma(SHORT_MA_LENGTH, "S")
    #     return current_price_minus_short_ma > 0
