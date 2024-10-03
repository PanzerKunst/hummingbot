import asyncio
import os
import sys
from datetime import datetime

import plotly.graph_objects as go
from plotly.offline import plot

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test.backtesting_utils import create_backtesting_figure  # noqa: E402

from controllers.generic.excalibur_backtesting import ExcaliburBacktesting  # noqa: E402


async def backtest():
    backtesting_engine = ExcaliburBacktesting()
    config = backtesting_engine.get_controller_config_instance_from_yml("conf_generic.excalibur_SOL.yml")

    start_time = datetime(2024, 8, 3).timestamp()
    end_time = datetime(2024, 8, 5).timestamp()

    backtesting_results = await backtesting_engine.run_backtesting(
        controller_config=config,
        trade_cost=0.0006,
        start=int(start_time),
        end=int(end_time),
        backtesting_resolution="1m"
    )

    df = backtesting_results["processed_data"]["features"]
    executors = backtesting_results["executors"]
    results = backtesting_results["results"]

    fig = create_backtesting_figure(
        df=df,
        executors=executors,
        config=config.dict())
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_short"]))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_long"]))
    print(results)
    plot(fig)


if __name__ == "__main__":
    asyncio.run(backtest())
