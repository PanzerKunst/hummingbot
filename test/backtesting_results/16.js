const conf = `
id: AX1hegsjY2Eg1qHH7a5vJFEVQiKuuKMHrJc4dQ698Lhs
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
candles_connector: okx
candles_interval: 1m
candles_length: 14
unfilled_order_expiration_min: 5
min_spread_pct: 1.0
normalized_bbp_mult: 0.02
normalized_bbb_mult: 0.05
`

const result = {
    'net_pnl': 0,
    'net_pnl_quote': 0,
    'total_executors': 0,
    'total_executors_with_position': 0,
    'total_volume': 0,
    'total_long': 0,
    'total_short': 0,
    'close_types': 0,
    'accuracy_long': 0,
    'accuracy_short': 0,
    'total_positions': 0,
    'accuracy': 0,
    'max_drawdown_usd': 0,
    'max_drawdown_pct': 0,
    'sharpe_ratio': 0,
    'profit_factor': 0,
    'win_signals': 0,
    'loss_signals': 0
}