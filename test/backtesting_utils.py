import os
import sys
from decimal import Decimal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hummingbot.core.data_type.common import TradeType  # noqa: E402


def get_bt_candlestick_trace(df):
    df.index = pd.to_datetime(df.timestamp, unit='s')
    return go.Scatter(x=df.index,
                      y=df['close'],
                      mode='lines',
                      line=dict(color="blue"),
                      )


def get_pnl_trace(executors):
    pnl = [e.net_pnl_quote for e in executors]
    cum_pnl = np.cumsum(pnl)
    return go.Scatter(
        x=pd.to_datetime([e.close_timestamp for e in executors], unit="s"),
        y=cum_pnl,
        mode='lines',
        line=dict(color='gold', width=2, dash="dash"),
        name='Cumulative PNL'
    )


def get_default_layout(title=None, height=800, width=1800):
    layout = {
        "template": "plotly_dark",
        "plot_bgcolor": 'rgba(0, 0, 0, 0)',  # Transparent background
        "paper_bgcolor": 'rgba(0, 0, 0, 0.1)',  # Lighter shade for the paper
        "font": {"color": 'white', "size": 12},  # Consistent font color and size
        "height": height,
        "width": width,
        "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
        "xaxis_rangeslider_visible": False,
        "hovermode": "x unified",
        "showlegend": False,
    }
    if title:
        layout["title"] = title
    return layout


def add_executors_trace(fig, executors, row, col):
    for executor in executors:
        entry_time = pd.to_datetime(executor.timestamp, unit='s')
        entry_price = executor.custom_info["current_position_average_price"]
        exit_time = pd.to_datetime(executor.close_timestamp, unit='s')
        exit_price = executor.custom_info["close_price"]
        name = "Buy Executor" if executor.config.side == TradeType.BUY else "Sell Executor"

        if executor.filled_amount_quote == 0:
            fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, entry_price], mode='lines',
                                     line=dict(color='grey', width=2, dash="dash"), name=name), row=row, col=col)
        else:
            if executor.net_pnl_quote > Decimal(0):
                fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, exit_price], mode='lines',
                                         line=dict(color='green', width=3), name=name), row=row, col=col)
            else:
                fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, exit_price], mode='lines',
                                         line=dict(color='red', width=3), name=name), row=row, col=col)

        # print(f"Entry_time: {entry_time} Exit time: {exit_time}, Exit price: {exit_price}")
    return fig


def create_backtesting_figure(df, executors, config):
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, subplot_titles=('Candlestick', 'PNL Quote'),
                        row_heights=[0.7, 0.3])

    # Add candlestick trace
    fig.add_trace(get_bt_candlestick_trace(df), row=1, col=1)

    # Add executors trace
    fig = add_executors_trace(fig, executors, row=1, col=1)

    # Add PNL trace
    fig.add_trace(get_pnl_trace(executors), row=2, col=1)

    # Apply the theme layout
    layout_settings = get_default_layout(f"Trading Pair: {config['trading_pair']}")
    layout_settings["showlegend"] = False
    fig.update_layout(**layout_settings)

    # Update axis properties
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="PNL", row=2, col=1)
    return fig
