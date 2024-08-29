import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hummingbot.strategy_v2.backtesting.controllers_backtesting.generic_backtesting import (  # noqa: E402
    GenericBacktesting,
)


async def backtest():
    backtesting_engine = GenericBacktesting()
    controller_config = backtesting_engine.get_controller_config_instance_from_yml("conf_generic.generic_pk_1.yml")

    start_time = datetime(2024, 7, 1).timestamp()
    end_time = datetime(2024, 8, 25).timestamp()

    backtesting_results = await backtesting_engine.run_backtesting(
        controller_config=controller_config,
        trade_cost=0.0006,
        start=int(start_time),
        end=int(end_time),
        backtesting_resolution="1m"
    )

    results = backtesting_results["results"]  # noqa: F841
    i = 0  # noqa: F841


if __name__ == "__main__":
    asyncio.run(backtest())
