import math
from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.ashbringer_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import compute_take_profit_price, has_unfilled_order_expired
from scripts.pk.take_profit_limit_order import TakeProfitLimitOrder
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Generate config file: create --script-config ashbringer
# Start the bot: start --script ashbringer.py --conf conf_ashbringer_AI16Z.yml
#                start --script ashbringer.py --conf conf_ashbringer_FARTCOIN.yml
#                start --script ashbringer.py --conf conf_ashbringer_VINE.yml
# Quickstart script: -p=a -f ashbringer.py -c conf_ashbringer_AI16Z.yml

ORDER_REF_TF: str = "TrendFollowing"
CANDLE_DURATION_MINUTES: int = 1
SHORT_MA_LENGTH: int = 15  # 3 * 5
LONG_MA_LENGTH: int = 285  # 3 * 95


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()
        self.latest_saved_candles_timestamp: float = 0

        self.has_opened_at_launch: bool = not config.should_open_position_at_launch
        self.nb_take_profits_left: int = self.get_max_take_profits()
        self.latest_filled_tp_order: TakeProfitLimitOrder | None = None

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

        self.check_if_candles_missed_beats(candles_df["timestamp"])

        candles_df["index"] = candles_df["timestamp"]
        candles_df.set_index("index", inplace=True)

        candles_df["timestamp_iso"] = pd.to_datetime(candles_df["timestamp"], unit="s")

        candles_df[f"EMA_{SHORT_MA_LENGTH}"] = candles_df.ta.ema(length=SHORT_MA_LENGTH)
        candles_df[f"EMA_{LONG_MA_LENGTH}"] = candles_df.ta.ema(length=LONG_MA_LENGTH)

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def check_if_candles_missed_beats(self, timestamp_series: pd.Series):
        current_timestamp: float = timestamp_series.iloc[-1]

        if self.latest_saved_candles_timestamp == 0:
            self.latest_saved_candles_timestamp = current_timestamp

        delta: int = int(current_timestamp - self.latest_saved_candles_timestamp)

        if delta > CANDLE_DURATION_MINUTES * 60:
            self.logger().error(f"check_if_candles_missed_beats() | missed {delta/60} minutes between the last two candles fetch")

        self.latest_saved_candles_timestamp = current_timestamp

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        # if not self.is_coin_still_tradable():
        #     self.logger().info("create_actions_proposal() > Stopping the bot as the coin is no longer tradable")
        #     HummingbotApplication.main_application().stop()

        self.check_for_newly_filled_tp()
        self.create_actions_proposal_tf()
        self.create_actions_proposal_tp()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()
        self.stop_actions_proposal_tf()
        self.stop_actions_proposal_tp()

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
    # TF start/stop action proposals
    #

    def create_actions_proposal_tf(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_TF)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_tf_order(TradeType.SELL, active_orders):
            triple_barrier = self.get_triple_barrier()
            amount_quote = self.get_position_quote_amount(TradeType.SELL)
            self.create_order(TradeType.SELL, self.get_current_close(), triple_barrier, amount_quote, ORDER_REF_TF)
            self.nb_take_profits_left = self.get_max_take_profits()

            # TODO: remove
            self.logger().info(f"create_actions_proposal_tf() > created SELL order | self.nb_take_profits_left:{self.nb_take_profits_left}")

        if self.can_create_tf_order(TradeType.BUY, active_orders):
            triple_barrier = self.get_triple_barrier()
            amount_quote = self.get_position_quote_amount(TradeType.BUY)
            self.create_order(TradeType.BUY, self.get_current_close(), triple_barrier, amount_quote, ORDER_REF_TF)
            self.nb_take_profits_left = self.get_max_take_profits()

            # TODO: remove
            self.logger().info(f"create_actions_proposal_tf() > created BUY order | self.nb_take_profits_left:{self.nb_take_profits_left}")

    def can_create_tf_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        amount_quote = self.get_position_quote_amount(side)

        if not self.can_create_order(side, amount_quote, ORDER_REF_TF, 0):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if not self.has_opened_at_launch and not self.is_latest_short_ma_over_long():
                self.has_opened_at_launch = True
                self.logger().info(f"can_create_tf_order() > Opening initial TF Sell at {self.get_current_close()}")
                return True

            elif self.did_short_ma_cross_under_long():
                self.logger().info(f"can_create_tf_order() > Opening TF Sell at {self.get_current_close()}")
                return True

            return False

        if not self.has_opened_at_launch and self.is_latest_short_ma_over_long():
            self.has_opened_at_launch = True
            self.logger().info(f"can_create_tf_order() > Opening initial TF Buy at {self.get_current_close()}")
            return True

        elif self.did_short_ma_cross_over_long():
            self.logger().info(f"can_create_tf_order() > Opening TF Buy at {self.get_current_close()}")
            return True

        return False

    def stop_actions_proposal_tf(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_TF)

        if len(filled_sell_orders) > 0:
            if self.did_short_ma_cross_over_long():
                self.logger().info(f"stop_actions_proposal_tf() > Closing TF Sell at {self.get_current_close()}")
                self.close_filled_orders(filled_sell_orders, OrderType.MARKET, CloseType.COMPLETED)

        if len(filled_buy_orders) > 0:
            if self.did_short_ma_cross_under_long():
                self.logger().info(f"stop_actions_proposal_tf() > Closing TF Buy at {self.get_current_close()}")
                self.close_filled_orders(filled_buy_orders, OrderType.MARKET, CloseType.COMPLETED)

    #
    # TP start/stop action proposals
    #

    def check_for_newly_filled_tp(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_TF)

        for filled_tf_order in filled_sell_orders + filled_buy_orders:
            latest_filled_tp_order: TakeProfitLimitOrder | None = self.get_latest_filled_tp_limit_order(filled_tf_order)

            if not latest_filled_tp_order:
                break

            self.logger().info(f"check_for_newly_filled_tp | latest_filled_tp_order:{latest_filled_tp_order}")

            if not self.latest_filled_tp_order or self.latest_filled_tp_order.order_id != latest_filled_tp_order.order_id:
                self.logger().info("check_for_newly_filled_tp > we got a new one!")
                self.nb_take_profits_left -= 1
                self.latest_filled_tp_order = latest_filled_tp_order

    def create_actions_proposal_tp(self):
        if self.nb_take_profits_left == 0:

            # TODO: remove
            self.logger().info("create_actions_proposal_tp > self.nb_take_profits_left == 0")

            return

        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_TF)

        if self._is_tp_cooling_down(filled_sell_orders + filled_buy_orders):
            return

        if len(filled_sell_orders) > 0:
            for filled_tf_order in filled_sell_orders:
                unfilled_tp_orders = self.get_unfilled_tp_limit_orders(filled_tf_order)

                if len(unfilled_tp_orders) == 0:
                    tp_amount: Decimal = filled_tf_order.amount * self.config.tp_position_pct / 100
                    tp_price: Decimal = compute_take_profit_price(TradeType.SELL, self.get_current_close(), self.config.tp_pct / 100)
                    self.create_tp_limit_order(filled_tf_order, tp_amount, tp_price)

        if len(filled_buy_orders) > 0:
            for filled_tf_order in filled_buy_orders:
                unfilled_tp_orders = self.get_unfilled_tp_limit_orders(filled_tf_order)

                if len(unfilled_tp_orders) == 0:
                    tp_amount: Decimal = filled_tf_order.amount * self.config.tp_position_pct / 100
                    tp_price: Decimal = compute_take_profit_price(TradeType.BUY, self.get_current_close(), self.config.tp_pct / 100)
                    self.create_tp_limit_order(filled_tf_order, tp_amount, tp_price)

    def _is_tp_cooling_down(self, filled_orders) -> bool:
        for filled_tf_order in filled_orders:
            latest_filled_tp_order: TakeProfitLimitOrder | None = self.get_latest_filled_tp_limit_order(filled_tf_order)

            if not latest_filled_tp_order:
                continue

            is_cooling_down: bool = latest_filled_tp_order.last_filled_at + self.config.tp_cooldown_min * 60 > self.get_market_data_provider_time()

            if is_cooling_down:
                self.logger().info("_is_tp_cooling_down > TRUE")
                return True

        return False

    def stop_actions_proposal_tp(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_TF)

        for filled_tf_order in filled_sell_orders + filled_buy_orders:
            for unfilled_tp_order in self.get_unfilled_tp_limit_orders(filled_tf_order):
                if has_unfilled_order_expired(unfilled_tp_order, self.config.tp_expiration_min * 60, self.get_market_data_provider_time()):
                    self.logger().info("Unfilled TP order has expired")
                    self.cancel_take_profit_for_order(unfilled_tp_order.tracked_order)

        self._check_unfilled_tps_which_shouldnt_be_there()

    # TODO: remove
    def _check_unfilled_tps_which_shouldnt_be_there(self):
        if self.nb_take_profits_left > 0:
            return

        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_TF)

        for filled_order in filled_sell_orders + filled_buy_orders:
            unfilled_limit_take_profit_orders = self.get_unfilled_tp_limit_orders(filled_order)

            if len(unfilled_limit_take_profit_orders) > 0:
                self.logger().info(f"_check_unfilled_tps_which_shouldnt_be_there > There is {len(unfilled_limit_take_profit_orders)} unfilled TPs left which shouldn't be there")

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
    # TF functions
    #

    # def is_coin_still_tradable(self) -> bool:
    #     launch_timestamp: float = iso_to_timestamp(self.config.coin_launch_date)
    #     start_of_today_timestamp = normalize_timestamp_to_midnight(self.get_market_data_provider_time())
    #     max_trade_duration = self.config.nb_days_trading_post_launch * 24 * 60 * 60  # seconds
    #
    #     return start_of_today_timestamp <= launch_timestamp + max_trade_duration

    def get_max_take_profits(self) -> int:
        return math.floor(1 / self.config.tp_position_pct * 100)  # If tp_position_pct = 24%, we want maxTps = 4. 1 / 24 * 100 = 4.16

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
