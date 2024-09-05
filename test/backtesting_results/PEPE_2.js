/*
    start_time = datetime(2024, 8, 27).timestamp()
    end_time = datetime(2024, 9, 4).timestamp()
 */

const conf = `
id: conf_generic.mm_bbands_PEPE
controller_name: mm_bbands
controller_type: generic
total_amount_quote: 20
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: PEPE-USDT
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
high_volatility_threshold: 1.0
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.2
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': -0.09913975805994413,
    'net_pnl_quote': -1.9827951611988825,
    'total_executors': 2151,
    'total_executors_with_position': 383,
    'total_volume': 15318.311723904551,
    'total_long': 194,
    'total_short': 189,
    'close_types': {'EARLY_STOP': 1766, 'STOP_LOSS': 126, 'TAKE_PROFIT': 188, 'TIME_LIMIT': 71},
    'accuracy_long': 0.5309278350515464,
    'accuracy_short': 0.582010582010582,
    'total_positions': 383,
    'accuracy': 0.556135770234987,
    'max_drawdown_usd': -3.009601987805311,
    'max_drawdown_pct': -0.14942624997520956,
    'sharpe_ratio': -0.2086129351622208,
    'profit_factor': 0.9341622242089643,
    'win_signals': 213,
    'loss_signals': 170
}