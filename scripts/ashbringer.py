import asyncio
from decimal import Decimal
from typing import Dict, List, Tuple

import pandas as pd
from pandas_ta import stoch

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.ashbringer_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import timestamp_to_iso
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Generate config file: create --script-config ashbringer_config
# Start the bot: start --script ashbringer.py --conf conf_ashbringer_GOAT.yml
#                start --script ashbringer.py --conf conf_ashbringer_AI16Z.yml
#                start --script ashbringer.py --conf conf_ashbringer_CHILLGUY.yml
#                start --script ashbringer.py --conf conf_ashbringer_FARTCOIN.yml
#                start --script ashbringer.py --conf conf_ashbringer_MOODENG.yml
#                start --script ashbringer.py --conf conf_ashbringer_PENGU.yml
#                start --script ashbringer.py --conf conf_ashbringer_PNUT.yml
#                start --script ashbringer.py --conf conf_ashbringer_WIF.yml
# Quickstart script: -p=a -f ashbringer.py -c conf_ashbringer_GOAT.yml

ORDER_REF_TREND_REVERSAL: str = "TrendReversal"


class ExcaliburStrategy(PkStrategy):
    @classmethod
    def init_markets(cls, config: ExcaliburConfig):
        cls.markets = {config.connector_name: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: ExcaliburConfig):
        super().__init__(connectors, config)

        self.processed_data = pd.DataFrame()

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                connector.set_position_mode(self.config.position_mode)

                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def get_triple_barrier(self) -> TripleBarrier:
        return TripleBarrier(
            open_order_type=OrderType.MARKET,
            stop_loss=self.compute_trend_reversal_sl_pct() / 100
        )

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

        candles_df["RSI_20"] = candles_df.ta.rsi(length=20)

        # Calling the lower-level function, because the one in core.py has a bug in the argument names
        stoch_40_df = stoch(
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            k=40,
            d=1,
            smooth_k=1
        )

        candles_df["STOCH_40_k"] = stoch_40_df["STOCHk_40_1_1"]

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        self.create_actions_proposal_trend_reversal()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()
        self.stop_actions_proposal_trend_reversal()

        return []  # Always return []

    def format_status(self) -> str:
        original_status = super().format_status()
        custom_status = ["\n"]

        if self.ready_to_trade:
            if not self.processed_data.empty:
                columns_to_display = [
                    "timestamp_iso",
                    "close",
                    "volume",
                    "RSI_20",
                    "STOCH_40_k"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display], table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # Trend Reversal start/stop action proposals
    #

    def create_actions_proposal_trend_reversal(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_TREND_REVERSAL)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_trend_reversal_order(TradeType.BUY, active_orders):
            triple_barrier = self.get_triple_barrier()

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.BUY, self.get_mid_price(), triple_barrier, self.config.amount_quote, ORDER_REF_TREND_REVERSAL)
            )

    def can_create_trend_reversal_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_TREND_REVERSAL, 20):
            return False

        if len(active_tracked_orders) > 0:
            return False

        history_candle_count: int = 25

        if (
            self.is_recent_rsi_low_enough(4) and
            self.is_price_crashing(history_candle_count) and
            self.is_price_bottom_recent(history_candle_count, 4) and
            self.did_price_rebound(history_candle_count)
        ):
            self.reset_tr_context(self.compute_tr_bottom_stoch(4))
            self.logger().info(f"can_create_trend_reversal_order() > Opening Trend Reversal Buy at {self.get_current_close()}")
            return True

        return False

    def stop_actions_proposal_trend_reversal(self):
        _, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_TREND_REVERSAL)

        if len(filled_buy_orders) > 0:
            if self.has_stoch_reversed_for_trend_reversal_buy(3):
                self.logger().info(f"stop_actions_proposal_trend_reversal() > Closing Trend Reversal Buy at {self.get_current_close()}")
                self.market_close_orders(filled_buy_orders, CloseType.COMPLETED)

    #
    # Getters on `self.processed_data[]`
    #

    def get_current_close(self) -> Decimal:
        return self._get_close_at_index(-1)

    def get_latest_close(self) -> Decimal:
        return self._get_close_at_index(-2)

    def _get_close_at_index(self, index: int) -> Decimal:
        close_series: pd.Series = self.processed_data["close"]
        return Decimal(close_series.iloc[index])

    def get_open_at_index(self, index: int) -> Decimal:
        open_series: pd.Series = self.processed_data["open"]
        return Decimal(open_series.iloc[index])

    def get_current_stoch(self, length: int) -> Decimal:
        stoch_series: pd.Series = self.processed_data[f"STOCH_{length}_k"]
        return Decimal(stoch_series.iloc[-1])

    #
    # Trend Reversal Context
    #

    def reset_tr_context(self, bottom_stoch: Decimal):
        self.save_tr_drop_pct(Decimal(0.0), self.get_market_data_provider_time())
        self.save_tr_bottom_price(Decimal(0.0), self.get_market_data_provider_time())
        self.save_tr_bottom_stoch(bottom_stoch, self.get_market_data_provider_time())
        self.save_tr_peak_stoch(bottom_stoch + 60, self.get_market_data_provider_time())

        self.tr_stoch_reversal_counter: int = 0
        self.logger().info("Context is reset")

    def save_tr_drop_pct(self, price_drop_pct: Decimal, timestamp: float):
        self.saved_tr_drop_pct: Tuple[Decimal, float] = price_drop_pct, timestamp

    def save_tr_bottom_price(self, bottom_price: Decimal, timestamp: float):
        self.saved_tr_bottom_price: Tuple[Decimal, float] = bottom_price, timestamp

    def save_tr_bottom_stoch(self, bottom_stoch: Decimal, timestamp: float):
        self.saved_tr_bottom_stoch: Tuple[Decimal, float] = bottom_stoch, timestamp

    def save_tr_peak_stoch(self, peak_stoch: Decimal, timestamp: float):
        self.saved_tr_peak_stoch: Tuple[Decimal, float] = peak_stoch, timestamp

    #
    # Trend Reversal functions
    #

    def is_recent_rsi_low_enough(self, candle_count: int) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_20"]
        recent_rsis = rsi_series.iloc[-candle_count:].reset_index(drop=True)
        bottom_rsi: Decimal = Decimal(recent_rsis.min())

        if bottom_rsi < 30:
            self.logger().info(f"is_recent_rsi_low_enough() | bottom_rsi:{bottom_rsi}")

        return bottom_rsi < 30

    def is_price_crashing(self, candle_count: int) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-candle_count:].reset_index(drop=True)
        bottom_price = Decimal(recent_lows.min())
        bottom_price_index = recent_lows.idxmin()

        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-candle_count:].reset_index(drop=True)
        peak_price = Decimal(recent_highs.max())
        peak_price_index = recent_highs.idxmax()

        if peak_price_index > bottom_price_index:
            return False

        price_delta_pct: Decimal = (peak_price - bottom_price) / bottom_price * 100
        is_crashing = self.config.min_price_delta_pct_to_open_trend_reversal < price_delta_pct

        if is_crashing:
            self.logger().info(f"is_price_crashing() | bottom_price_index:{bottom_price_index} | bottom_price:{bottom_price} | peak_price_index:{peak_price_index} | peak_price:{peak_price}")
            self.logger().info(f"is_price_crashing() | current_price:{self.get_current_close()} | price_delta_pct:{price_delta_pct}")
            self.save_tr_drop_pct(price_delta_pct, self.get_market_data_provider_time())

        return is_crashing

    def is_price_bottom_recent(self, history_candle_count: int, recent_candle_count: int) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        lows = low_series.iloc[-history_candle_count:].reset_index(drop=True)

        bottom_price_index = lows.idxmin()

        self.logger().info(f"is_price_bottom_recent() | bottom_price_index:{bottom_price_index}")

        return bottom_price_index >= history_candle_count - recent_candle_count  # >= 25 - 4

    def did_price_rebound(self, history_candle_count: int) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-history_candle_count:].reset_index(drop=True)
        bottom_price = Decimal(recent_lows.min())

        current_price: Decimal = self.get_current_close()
        saved_tr_drop_pct, _ = self.saved_tr_drop_pct
        price_rebound_pct = (current_price - bottom_price) / current_price * 100

        self.logger().info(f"did_price_rebound() | current_price:{current_price} | saved_tr_drop_pct:{saved_tr_drop_pct} | price_rebound_pct:{price_rebound_pct}")

        is_significant: bool = price_rebound_pct > saved_tr_drop_pct / 10

        if is_significant:
            self.save_tr_bottom_price(bottom_price, self.get_market_data_provider_time())

        return is_significant

    def compute_trend_reversal_sl_pct(self) -> Decimal:
        saved_tr_bottom_price, _ = self.saved_tr_bottom_price
        current_price: Decimal = self.get_current_close()

        price_rebound_pct = (current_price - saved_tr_bottom_price) / current_price * 100

        self.logger().info(f"compute_trend_reversal_sl_pct() | price_rebound_pct:{price_rebound_pct}")

        return price_rebound_pct

    def compute_tr_bottom_stoch(self, candle_count: int):
        stoch_series: pd.Series = self.processed_data["STOCH_40_k"]
        recent_stochs = stoch_series.iloc[-candle_count:].reset_index(drop=True)

        bottom_stoch: Decimal = recent_stochs.min()

        self.logger().info(f"compute_tr_bottom_stoch() | bottom_stoch:{bottom_stoch}")

        return bottom_stoch

    def has_stoch_reversed_for_trend_reversal_buy(self, candle_count: int) -> bool:
        stoch_series: pd.Series = self.processed_data["STOCH_40_k"]
        recent_stochs = stoch_series.iloc[-candle_count:].reset_index(drop=True)

        peak_stoch: Decimal = Decimal(recent_stochs.max())
        saved_peak_stoch, _ = self.saved_tr_peak_stoch
        saved_bottom_stoch, _ = self.saved_tr_bottom_stoch

        if max([peak_stoch, saved_peak_stoch]) <= saved_bottom_stoch + 60:
            return False

        if peak_stoch > saved_peak_stoch:
            timestamp_series: pd.Series = self.processed_data["timestamp"]
            recent_timestamps = timestamp_series.iloc[-candle_count:].reset_index(drop=True)
            peak_stoch_index = recent_stochs.idxmax()

            peak_stoch_timestamp = recent_timestamps.iloc[peak_stoch_index]

            self.logger().info(f"has_stoch_reversed_for_trend_reversal_buy() | peak_stoch_index:{peak_stoch_index} | peak_stoch_timestamp:{timestamp_to_iso(peak_stoch_timestamp)}")
            self.save_tr_peak_stoch(peak_stoch, peak_stoch_timestamp)

        saved_peak_stoch, _ = self.saved_tr_peak_stoch

        stoch_threshold: Decimal = saved_peak_stoch - 3
        current_stoch = self.get_current_stoch(40)

        self.logger().info(f"has_stoch_reversed_for_trend_reversal_buy() | saved_peak_stoch:{saved_peak_stoch} | current_stoch:{current_stoch} | stoch_threshold:{stoch_threshold} | current_price:{self.get_current_close()}")

        if current_stoch > stoch_threshold:
            self.tr_stoch_reversal_counter = 0
            self.logger().info("has_stoch_reversed_for_trend_reversal_buy() | resetting self.tr_stoch_reversal_counter to 0")
            return False

        self.tr_stoch_reversal_counter += 1
        self.logger().info(f"has_stoch_reversed_for_trend_reversal_buy() | incremented self.tr_stoch_reversal_counter to:{self.tr_stoch_reversal_counter}")

        return self.tr_stoch_reversal_counter > 4

    # def are_candles_green(self, candle_count: int) -> bool:
    #     candle_start_index: int = -candle_count - 1
    #
    #     open_series: pd.Series = self.processed_data["open"]
    #     recent_opens = open_series.iloc[candle_start_index:-1].reset_index(drop=True)
    #
    #     close_series: pd.Series = self.processed_data["close"]
    #     recent_closes = close_series.iloc[candle_start_index:-1].reset_index(drop=True)
    #
    #     return all(recent_closes[i] > recent_opens[i] for i in range(len(recent_opens)))
