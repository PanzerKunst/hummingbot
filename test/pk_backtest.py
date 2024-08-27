import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from hummingbot.strategy_v2.backtesting import MarketMakingBacktesting
from hummingbot.strategy_v2.backtesting.controllers_backtesting.generic_backtesting import (  # noqa: E402
    GenericBacktesting,
)


async def backtest():
    backtesting_engine = GenericBacktesting()
    controller_config = backtesting_engine.get_controller_config_instance_from_yml("conf_generic.generic_pk_1.yml")

    # backtesting_engine = MarketMakingBacktesting()
    # controller_config = backtesting_engine.get_controller_config_instance_from_yml("pmmdynamic-okx-wld_0.1.yml")

    start_time_ms = datetime(2024, 8, 10).timestamp() * 1000
    end_time_ms = datetime(2024, 8, 11).timestamp() * 1000

    backtesting_results = await backtesting_engine.run_backtesting(
        controller_config=controller_config,
        trade_cost=0.0006,
        start=int(start_time_ms),
        end=int(end_time_ms),
        backtesting_resolution="1m"
    )

    df = backtesting_results["processed_data"]  # noqa: F841
    executors = backtesting_results["executors"]  # noqa: F841
    results = backtesting_results["results"]  # noqa: F841
    i = 0  # noqa: F841


if __name__ == "__main__":
    asyncio.run(backtest())
