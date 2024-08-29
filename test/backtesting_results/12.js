const conf = `
id: eMdPCafJRi8BX1rcNAyeZxPCwznpyXbLJWN5if4NmhL
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
filled_order_expiration_min: 1000
bollinger_bands_length: 7
bollinger_bands_std_dev: 2.0
bollinger_bands_bandwidth_threshold: 1.5
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 14
unfilled_order_expiration_min: 1000
min_spread_pct: 0.5
normalized_bbp_mult: 0.1
normalized_bbb_mult: 0.1
`

const result = {
    'net_pnl': 0.17962128123940407,
    'net_pnl_quote': 17.962128123940406,
    'total_executors': 43,
    'total_executors_with_position': 30,
    'total_volume': 5996.19695222179,
    'total_long': 21,
    'total_short': 9,
    'close_types': {'STOP_LOSS': 2, 'TAKE_PROFIT': 24, 'TIME_LIMIT': 17},
    'accuracy_long': 0.8571428571428571,
    'accuracy_short': 0.6666666666666666,
    'total_positions': 30,
    'accuracy': 0.8,
    'max_drawdown_usd': -5.298392815329068,
    'max_drawdown_pct': -0.05317188344824176,
    'sharpe_ratio': 1.3282235881590163,
    'profit_factor': 3.0656037942470906,
    'win_signals': 24,
    'loss_signals': 6
}