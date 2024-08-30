const conf = `
id: 2gk378dinHVrUTzBF357Ywjor3WMxjkmmqd1GkvaXdWh
controller_name: generic_pk
controller_type: generic
total_amount_quote: 100
manual_kill_switch: null
candles_config: []
connector_name: binance
trading_pair: AAVE-USDT
leverage: 20
position_mode: HEDGE
unfilled_order_expiration_min: 5
stop_loss_pct: 0.5
take_profit_pct: 0.3
filled_order_expiration_min: 1000
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
candles_count_for_trend: 12
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
volatility_threshold_bbb: 1.0
candles_connector: binance
candles_interval: 1m
candles_length: 32
default_spread_pct: 0.5
`

const result = {
    'net_pnl': -0.07920798754871691,
    'net_pnl_quote': -7.920798754871692,
    'total_executors': 1884,
    'total_executors_with_position': 466,
    'total_volume': 93229.91914391042,
    'total_long': 215,
    'total_short': 251,
    'close_types': {'EARLY_STOP': 1416, 'STOP_LOSS': 184, 'TAKE_PROFIT': 281, 'TIME_LIMIT': 3},
    'accuracy_long': 0.6325581395348837,
    'accuracy_short': 0.5776892430278885,
    'total_positions': 466,
    'accuracy': 0.6030042918454935,
    'max_drawdown_usd': -23.4664344717497,
    'max_drawdown_pct': -0.2358625174910152,
    'sharpe_ratio': 0.5303437957295232,
    'profit_factor': 0.9368985978755701,
    'win_signals': 281,
    'loss_signals': 185
}