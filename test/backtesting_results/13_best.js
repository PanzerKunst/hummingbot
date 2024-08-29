const conf = `
id: 7iXbZ3sikepy6nJEs1S6WpbsqPitPXM6RNThJXjrZwBQ
controller_name: generic_pk
controller_type: generic
total_amount_quote: 100
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: AAVE-USDT
leverage: 20
position_mode: HEDGE
stop_loss_pct: 2.5
take_profit_pct: 1.0
filled_order_expiration_min: 120
bollinger_bands_length: 7
bollinger_bands_std_dev: 2.0
bollinger_bands_bandwidth_threshold: 1.5
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 14
min_spread_pct: 0.5
normalized_bbp_mult: 0.1
normalized_bbb_mult: 0.1
`

const result = {
    'net_pnl': 0.3868712701110456,
    'net_pnl_quote': 38.687127011104565,
    'total_executors': 240,
    'total_executors_with_position': 101,
    'total_volume': 20182.124486055513,
    'total_long': 91,
    'total_short': 10,
    'close_types': {'STOP_LOSS': 3, 'TAKE_PROFIT': 48, 'TIME_LIMIT': 189},
    'accuracy_long': 0.7362637362637363,
    'accuracy_short': 0.7,
    'total_positions': 101,
    'accuracy': 0.7326732673267327,
    'max_drawdown_usd': -5.361771025989,
    'max_drawdown_pct': -0.0532919198448412,
    'sharpe_ratio': 2.3589574397760287,
    'profit_factor': 2.590978290803467,
    'win_signals': 74,
    'loss_signals': 27
}