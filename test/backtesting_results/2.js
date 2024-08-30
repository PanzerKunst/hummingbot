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
unfilled_order_expiration_min: 10
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
    'net_pnl': -0.053405626260237246,
    'net_pnl_quote': -5.340562626023725,
    'total_executors': 1223,
    'total_executors_with_position': 469,
    'total_volume': 93839.79519163081,
    'total_long': 215,
    'total_short': 254,
    'close_types': {'EARLY_STOP': 752, 'STOP_LOSS': 184, 'TAKE_PROFIT': 284, 'TIME_LIMIT': 3},
    'accuracy_long': 0.6139534883720931,
    'accuracy_short': 0.5984251968503937,
    'total_positions': 469,
    'accuracy': 0.605543710021322,
    'max_drawdown_usd': -21.70384059526234,
    'max_drawdown_pct': -0.21814658243820445,
    'sharpe_ratio': 0.742963195655871,
    'profit_factor': 0.9566710399849462,
    'win_signals': 284,
    'loss_signals': 185
}