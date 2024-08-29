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
min_spread_pct: 0.45
normalized_bbp_mult: 0.02
normalized_bbb_mult: 0.05
`

const result = {
    'net_pnl': -0.09921867919941083,
    'net_pnl_quote': -9.921867919941082,
    'total_executors': 2110,
    'total_executors_with_position': 141,
    'total_volume': 28193.51681134267,
    'total_long': 88,
    'total_short': 53,
    'close_types': {'EARLY_STOP': 1967, 'STOP_LOSS': 17, 'TAKE_PROFIT': 43, 'TIME_LIMIT': 83},
    'accuracy_long': 0.5454545454545454,
    'accuracy_short': 0.5471698113207547,
    'total_positions': 141,
    'accuracy': 0.5460992907801419,
    'max_drawdown_usd': -20.053697036341656,
    'max_drawdown_pct': -0.19817799200597605,
    'sharpe_ratio': 0.3408627183241779,
    'profit_factor': 0.8595342259829208,
    'win_signals': 77,
    'loss_signals': 64
}