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
min_spread_pct: 0.4
normalized_bbp_mult: 0.02
normalized_bbb_mult: 0.05
`

const result = {
    'net_pnl': -0.06275019753957353,
    'net_pnl_quote': -6.275019753957353,
    'total_executors': 2138,
    'total_executors_with_position': 197,
    'total_volume': 39392.79218696578,
    'total_long': 116,
    'total_short': 81,
    'close_types': {'EARLY_STOP': 1939, 'STOP_LOSS': 20, 'TAKE_PROFIT': 52, 'TIME_LIMIT': 127},
    'accuracy_long': 0.5689655172413793,
    'accuracy_short': 0.5308641975308642,
    'total_positions': 197,
    'accuracy': 0.5532994923857868,
    'max_drawdown_usd': -23.533938470201154,
    'max_drawdown_pct': -0.2344832286405356,
    'sharpe_ratio': 0.7890845281186046,
    'profit_factor': 0.9262995553088568,
    'win_signals': 109,
    'loss_signals': 88
}