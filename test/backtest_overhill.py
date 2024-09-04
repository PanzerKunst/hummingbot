import asyncio
import os
import sys
from datetime import datetime

import plotly.graph_objects as go
from plotly.offline import plot

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test.backtesting_utils import create_backtesting_figure  # noqa: E402

from controllers.generic.overhill_backtesting import OverhillBacktesting  # noqa: E402


async def backtest():
    backtesting_engine = OverhillBacktesting()
    config = backtesting_engine.get_controller_config_instance_from_yml("conf_generic.overhill_1.yml")

    start_time = datetime(2024, 8, 1).timestamp()
    end_time = datetime(2024, 8, 3).timestamp()

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
    fig.add_trace(go.Scatter(x=df.index,
                             y=df[f"BBU_{config.bbands_length}_{config.bbands_std_dev}"])
                  )
    fig.add_trace(go.Scatter(x=df.index,
                             y=df[f"BBM_{config.bbands_length}_{config.bbands_std_dev}"])
                  )
    fig.add_trace(go.Scatter(x=df.index,
                             y=df[f"BBL_{config.bbands_length}_{config.bbands_std_dev}"])
                  )
    print(results)
    plot(fig)


if __name__ == "__main__":
    asyncio.run(backtest())
