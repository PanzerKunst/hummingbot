const conf = `
id: C1PkH8bFzG3xasPijGM9cUm4bPxTLXiem35grtJUspiL
controller_name: generic_pk
controller_type: generic
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: AAVE-USDT
leverage: 5
position_mode: HEDGE
stop_loss_pct: 2.0
take_profit_pct: 1.0
filled_order_expiration_min: 60
bollinger_bands_length: 7
bollinger_bands_std_dev: 2.0
candles_interval: 1m
candles_length: 7
unfilled_order_expiration_min: 5
min_spread_pct: 0.6
normalized_bbp_mult: 0.02
`

const result = {
    'net_pnl': -0.07951424697069684,
    'net_pnl_quote': -7.951424697069684,
    'total_executors': 2077,
    'total_executors_with_position': 82,
    'total_volume': 16394.51575856023,
    'total_long': 48,
    'total_short': 34,
    'close_types': {'EARLY_STOP': 1993, 'STOP_LOSS': 11, 'TAKE_PROFIT': 25, 'TIME_LIMIT': 48},
    'accuracy_long': 0.5833333333333334,
    'accuracy_short': 0.47058823529411764,
    'total_positions': 82,
    'accuracy': 0.5365853658536586,
    'max_drawdown_usd': -15.231347438016062,
    'max_drawdown_pct': -0.15030197288221006,
    'sharpe_ratio': 0.28499743818289686,
    'profit_factor': 0.8206981269393454,
    'win_signals': 44,
    'loss_signals': 38
}