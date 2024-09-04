/*
    start_time = datetime(2024, 8, 27).timestamp()
    end_time = datetime(2024, 9, 4).timestamp()
 */

const conf = `
id: oePNxNXqRNkXBD8397uozUvph2unKvBfkVxeDgoDTM8
controller_name: generic_pk
controller_type: generic
total_amount_quote: 20
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: DOGS-USDT
leverage: 20
position_mode: HEDGE
unfilled_order_expiration_min: 10
stop_loss_pct: 0.9
take_profit_pct: 0.6
filled_order_expiration_min: 1000
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 1.0
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.5
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': -0.24017250226787173,
    'net_pnl_quote': -4.803450045357435,
    'total_executors': 2246,
    'total_executors_with_position': 591,
    'total_volume': 23635.56870893444,
    'total_long': 319,
    'total_short': 272,
    'close_types': {'EARLY_STOP': 1653, 'STOP_LOSS': 246, 'TAKE_PROFIT': 345, 'TIME_LIMIT': 2},
    'accuracy_long': 0.6018808777429467,
    'accuracy_short': 0.5625,
    'total_positions': 591,
    'accuracy': 0.583756345177665,
    'max_drawdown_usd': -7.835407161600937,
    'max_drawdown_pct': -0.4006412386199128,
    'sharpe_ratio': -0.3525203836942172,
    'profit_factor': 0.8942194582476336,
    'win_signals': 345,
    'loss_signals': 246
}