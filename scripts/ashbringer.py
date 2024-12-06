import asyncio
from decimal import Decimal
from typing import Dict, List

import pandas as pd
from pandas_ta import sma

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from scripts.ashbringer_config import ExcaliburConfig
from scripts.pk.pk_strategy import PkStrategy
from scripts.pk.pk_triple_barrier import TripleBarrier
from scripts.pk.pk_utils import compute_buy_orders_pnl_pct, compute_sell_orders_pnl_pct, was_an_order_recently_opened
from scripts.pk.tracked_order_details import TrackedOrderDetails

# Trend following via successive candles above or below a MA channel
# Generate config file: create --script-config ashbringer
# Start the bot: start --script ashbringer.py --conf conf_ashbringer_GOAT.yml
#                start --script ashbringer.py --conf conf_ashbringer_CHILLGUY.yml
#                start --script ashbringer.py --conf conf_ashbringer_FLOKI.yml
#                start --script ashbringer.py --conf conf_ashbringer_MOODENG.yml
#                start --script ashbringer.py --conf conf_ashbringer_NEIRO.yml
#                start --script ashbringer.py --conf conf_ashbringer_PNUT.yml
#                start --script ashbringer.py --conf conf_ashbringer_POPCAT.yml
# Quickstart script: -p=a -f ashbringer.py -c conf_ashbringer_GOAT.yml

ORDER_REF_MA_CHANNEL = "MaChannel"


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
            take_profit=self.config.take_profit_pct / 100
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

        candles_df["SMA_10_h"] = sma(close=candles_df["high"], length=10)
        candles_df["SMA_10_l"] = sma(close=candles_df["low"], length=10)

        candles_df.dropna(inplace=True)

        self.processed_data = candles_df

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        self.update_processed_data()

        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            self.logger().error("create_actions_proposal() > ERROR: processed_data_num_rows == 0")
            return []

        self.create_actions_proposal_ma_channel()

        return []  # Always return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        processed_data_num_rows = self.processed_data.shape[0]

        if processed_data_num_rows == 0:
            return []

        self.check_orders()

        self.stop_actions_proposal_ma_channel()

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
                    "SMA_10_h",
                    "SMA_10_l"
                ]

                custom_status.append(format_df_for_printout(self.processed_data[columns_to_display], table_format="psql"))

        return original_status + "\n".join(custom_status)

    #
    # MA Channel start/stop action proposals
    #

    def create_actions_proposal_ma_channel(self):
        active_sell_orders, active_buy_orders = self.get_active_tracked_orders_by_side(ORDER_REF_MA_CHANNEL)
        active_orders = active_sell_orders + active_buy_orders

        if self.can_create_ma_channel_order(TradeType.SELL, active_orders):
            entry_price: Decimal = self.get_best_bid() * Decimal(1 - self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier()

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.SELL, entry_price, triple_barrier, self.config.amount_quote, ORDER_REF_MA_CHANNEL)
            )

        if self.can_create_ma_channel_order(TradeType.BUY, active_orders):
            entry_price: Decimal = self.get_best_ask() * Decimal(1 + self.config.entry_price_delta_bps / 10000)
            triple_barrier = self.get_triple_barrier()

            asyncio.get_running_loop().create_task(
                self.create_twap_market_orders(TradeType.BUY, entry_price, triple_barrier, self.config.amount_quote, ORDER_REF_MA_CHANNEL)
            )

    def can_create_ma_channel_order(self, side: TradeType, active_tracked_orders: List[TrackedOrderDetails]) -> bool:
        if not self.can_create_order(side, self.config.amount_quote, ORDER_REF_MA_CHANNEL, 8):
            return False

        if len(active_tracked_orders) > 0:
            return False

        if side == TradeType.SELL:
            if (
                self.are_candles_fully_below_mal() and
                not self.is_recent_rsi_too_low_to_open_sell() and
                self.is_price_drop_significant() and
                not self.did_price_recently_pull_back_up()
            ):
                self.logger().info("can_create_ma_channel_order() > Opening Sell MA-C position")
                return True

            return False

        if (
            self.are_candles_fully_above_mah() and
            not self.is_recent_rsi_too_high_to_open_buy() and
            self.is_price_increase_significant() and
            not self.did_price_recently_pull_back_down()
        ):
            self.logger().info("can_create_ma_channel_order() > Opening Buy MA-C position")
            return True

        return False

    def stop_actions_proposal_ma_channel(self):
        filled_sell_orders, filled_buy_orders = self.get_filled_tracked_orders_by_side(ORDER_REF_MA_CHANNEL)

        if len(filled_sell_orders) > 0:
            if self.has_order_been_open_long_enough(filled_sell_orders):
                if not self.is_sell_order_profitable(filled_sell_orders):
                    self.logger().info("stop_actions_proposal_ma_channel() > Closing Sell MA-C: Negative PnL")
                    self.market_close_orders(filled_sell_orders, CloseType.STOP_LOSS)

        if len(filled_buy_orders) > 0:
            if self.has_order_been_open_long_enough(filled_buy_orders):
                if not self.is_buy_order_profitable(filled_buy_orders):
                    self.logger().info("stop_actions_proposal_ma_channel() > Closing Buy MA-C: Negative PnL")
                    self.market_close_orders(filled_buy_orders, CloseType.STOP_LOSS)

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

    def get_current_rsi(self, length: int) -> Decimal:
        rsi_series: pd.Series = self.processed_data[f"RSI_{length}"]
        return Decimal(rsi_series.iloc[-1])

    def get_current_mah(self) -> Decimal:
        smah_series: pd.Series = self.processed_data["SMA_10_h"]
        return Decimal(smah_series.iloc[-1])

    def get_current_mal(self) -> Decimal:
        smal_series: pd.Series = self.processed_data["SMA_10_l"]
        return Decimal(smal_series.iloc[-1])

    #
    # MA Channel functions
    #

    def are_candles_fully_below_mal(self) -> bool:
        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-5:-1].reset_index(drop=True)

        mal_series: pd.Series = self.processed_data["SMA_10_l"]
        recent_mals = mal_series.iloc[-5:-1].reset_index(drop=True)

        return all(recent_highs[i] < recent_mals[i] for i in range(len(recent_highs)))

    def are_candles_fully_above_mah(self) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-5:-1].reset_index(drop=True)

        mah_series: pd.Series = self.processed_data["SMA_10_h"]
        recent_mahs = mah_series.iloc[-5:-1].reset_index(drop=True)

        return all(recent_lows[i] > recent_mahs[i] for i in range(len(recent_lows)))

    def is_recent_rsi_too_low_to_open_sell(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_20"]
        recent_rsis = rsi_series.iloc[-4:]
        bottom_rsi: Decimal = Decimal(recent_rsis.min())

        if bottom_rsi < 34:
            self.logger().info(f"is_recent_rsi_too_low_to_open_sell() | bottom_rsi:{bottom_rsi}")

        return bottom_rsi < 34

    def is_recent_rsi_too_high_to_open_buy(self) -> bool:
        rsi_series: pd.Series = self.processed_data["RSI_20"]
        recent_rsis = rsi_series.iloc[-4:]
        peak_rsi: Decimal = Decimal(recent_rsis.max())

        if peak_rsi > 66:
            self.logger().info(f"is_recent_rsi_too_high_to_open_buy() | peak_rsi:{peak_rsi}")

        return peak_rsi > 66

    def is_price_drop_significant(self) -> bool:
        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-6:-1]  # 1 more than `are_candles_fully_below/below()`

        peak_price = Decimal(recent_highs.max())
        current_price = self.get_current_close()
        delta_pct: Decimal = (peak_price - current_price) / current_price * 100

        self.logger().info(f"is_price_drop_significant() | peak_price:{peak_price} | current_price:{current_price} | delta_pct:{delta_pct}")

        return delta_pct > Decimal(0.5)

    def is_price_increase_significant(self) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-6:-1]  # 1 more than `are_candles_fully_below/below()`

        bottom_price = Decimal(recent_lows.min())
        current_price = self.get_current_close()
        delta_pct: Decimal = (current_price - bottom_price) / current_price * 100

        self.logger().info(f"is_price_increase_significant() | bottom_price:{bottom_price} | current_price:{current_price} | delta_pct:{delta_pct}")

        return delta_pct > Decimal(0.5)

    def did_price_recently_pull_back_up(self) -> bool:
        low_series: pd.Series = self.processed_data["low"]
        recent_lows = low_series.iloc[-5:-1]

        bottom_price = Decimal(recent_lows.min())
        current_price = self.get_current_close()

        self.logger().info(f"did_price_recently_pull_back_up() | bottom_price:{bottom_price} | current_price:{current_price}")

        return current_price > bottom_price

    def did_price_recently_pull_back_down(self) -> bool:
        high_series: pd.Series = self.processed_data["high"]
        recent_highs = high_series.iloc[-5:-1].reset_index(drop=True)

        peak_price = Decimal(recent_highs.max())
        current_price = self.get_current_close()

        self.logger().info(f"did_price_recently_pull_back_down() | peak_price:{peak_price} | current_price:{current_price}")

        return current_price < peak_price

    def has_order_been_open_long_enough(self, filled_orders: List[TrackedOrderDetails]) -> bool:
        return not was_an_order_recently_opened(filled_orders, 2 * 60, self.get_market_data_provider_time())

    def is_sell_order_profitable(self, filled_sell_orders: List[TrackedOrderDetails]) -> bool:
        pnl_pct: Decimal = compute_sell_orders_pnl_pct(filled_sell_orders, self.get_mid_price())

        return pnl_pct > 0

    def is_buy_order_profitable(self, filled_buy_orders: List[TrackedOrderDetails]) -> bool:
        pnl_pct: Decimal = compute_buy_orders_pnl_pct(filled_buy_orders, self.get_mid_price())

        return pnl_pct > 0
