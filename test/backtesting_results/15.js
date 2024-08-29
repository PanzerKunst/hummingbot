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
normalized_bbp_mult: 0.52
normalized_bbb_mult: 0.55
`

const result = {
    'net_pnl': 0.09616625190712966,
    'net_pnl_quote': 9.616625190712966,
    'total_executors': 2073,
    'total_executors_with_position': 74,
    'total_volume': 14781.692955990235,
    'total_long': 74,
    'total_short': 0,
    'close_types': {'EARLY_STOP': 1997, 'STOP_LOSS': 3, 'TAKE_PROFIT': 25, 'TIME_LIMIT': 48},
    'accuracy_long': 0.581081081081081,
    'accuracy_short': 0.0,
    'total_positions': 74,
    'accuracy': 0.581081081081081,
    'max_drawdown_usd': -5.957257255517617,
    'max_drawdown_pct': -0.05878661795837737,
    'sharpe_ratio': 1.3686111748554193,
    'profit_factor': 1.3697103792953547,
    'win_signals': 43,
    'loss_signals': 31
}