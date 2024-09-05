/*
    start_time = datetime(2024, 8, 27).timestamp()
    end_time = datetime(2024, 9, 4).timestamp()
 */

const conf = `
id: conf_generic.mm_bbands_DOGS
controller_name: mm_bbands
controller_type: generic
total_amount_quote: 20
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: DOGS-USDT
leverage: 20
position_mode: HEDGE
cooldown_time_min: 3
unfilled_order_expiration_min: 10
stop_loss_pct: 0.9
take_profit_pct: 0.6
filled_order_expiration_min: 90
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 1.5
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.2
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': -0.07448589442606846,
    'net_pnl_quote': -1.4897178885213689,
    'total_executors': 2425,
    'total_executors_with_position': 725,
    'total_volume': 28992.322314712572,
    'total_long': 366,
    'total_short': 359,
    'close_types': {'EARLY_STOP': 1698, 'STOP_LOSS': 293, 'TAKE_PROFIT': 417, 'TIME_LIMIT': 17},
    'accuracy_long': 0.5573770491803278,
    'accuracy_short': 0.6044568245125348,
    'total_positions': 725,
    'accuracy': 0.5806896551724138,
    'max_drawdown_usd': -4.2635450853775385,
    'max_drawdown_pct': -0.217989402553981,
    'sharpe_ratio': -0.5466187176923408,
    'profit_factor': 0.9789753692802513,
    'win_signals': 421,
    'loss_signals': 304
}