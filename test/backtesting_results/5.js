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
unfilled_order_expiration_min: 8
min_spread_pct: 0.5
normalized_bbp_mult: 0.02
`

const result = {
    'net_pnl': -0.08239981553765087,
    'net_pnl_quote': -8.239981553765087,
    'total_executors': 1498,
    'total_executors_with_position': 134,
    'total_volume': 26793.232221477316,
    'total_long': 78,
    'total_short': 56,
    'close_types': {'EARLY_STOP': 1362, 'STOP_LOSS': 14, 'TAKE_PROFIT': 37, 'TIME_LIMIT': 85},
    'accuracy_long': 0.5641025641025641,
    'accuracy_short': 0.5,
    'total_positions': 134,
    'accuracy': 0.5373134328358209,
    'max_drawdown_usd': -17.20885084065234,
    'max_drawdown_pct': -0.17139072451755705,
    'sharpe_ratio': 0.21980724151019576,
    'profit_factor': 0.8634194563863772,
    'win_signals': 72,
    'loss_signals': 62
}