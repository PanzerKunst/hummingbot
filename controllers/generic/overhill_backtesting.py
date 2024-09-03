import pandas as pd

from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase


class OverhillBacktesting(BacktestingEngineBase):
    def update_processed_data(self, row: pd.Series):
        pass
