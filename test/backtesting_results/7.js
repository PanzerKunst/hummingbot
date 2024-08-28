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
min_spread_pct: 0.3
normalized_bbp_mult: 0.02
`

const result = {
    'net_pnl': -0.04446457242839127,
    'net_pnl_quote': -4.446457242839126,
    'total_executors': 2218,
    'total_executors_with_position': 361,
    'total_volume': 72192.45837553182,
    'total_long': 230,
    'total_short': 131,
    'close_types': {'EARLY_STOP': 1855, 'STOP_LOSS': 32, 'TAKE_PROFIT': 95, 'TIME_LIMIT': 236},
    'accuracy_long': 0.5826086956521739,
    'accuracy_short': 0.4961832061068702,
    'total_positions': 361,
    'accuracy': 0.5512465373961218,
    'max_drawdown_usd': -28.53858170179961,
    'max_drawdown_pct': -0.28254040186522433,
    'sharpe_ratio': 0.790842293784289,
    'profit_factor': 0.9698823029158131,
    'win_signals': 199,
    'loss_signals': 162
}