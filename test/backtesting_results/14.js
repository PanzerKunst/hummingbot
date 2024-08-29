const conf = `
id: 6ANjkbnMnQp9oiKkG4XoiAj8madvsym4DgzLSyxPRnVv
controller_name: generic_pk
controller_type: generic
total_amount_quote: 100
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
bollinger_bands_bandwidth_threshold: 1.5
candles_interval: 1m
candles_length: 7
unfilled_order_expiration_min: 5
min_spread_pct: 0.5
normalized_bbp_mult: 0.02
normalized_bbb_mult: 0.05
`

const result = {
    'net_pnl': 0.04130159276434338,
    'net_pnl_quote': 4.130159276434338,
    'total_executors': 2096,
    'total_executors_with_position': 116,
    'total_volume': 23192.323012682868,
    'total_long': 74,
    'total_short': 42,
    'close_types': {'EARLY_STOP': 1978, 'STOP_LOSS': 10, 'TAKE_PROFIT': 37, 'TIME_LIMIT': 71},
    'accuracy_long': 0.581081081081081,
    'accuracy_short': 0.5476190476190477,
    'total_positions': 116,
    'accuracy': 0.5689655172413793,
    'max_drawdown_usd': -9.609816743210239,
    'max_drawdown_pct': -0.09483032229469043,
    'sharpe_ratio': 0.9132043301117949,
    'profit_factor': 1.0840648433283997,
    'win_signals': 66,
    'loss_signals': 50
}