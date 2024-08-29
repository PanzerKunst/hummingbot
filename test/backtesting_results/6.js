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
stop_loss_pct: 2.0
take_profit_pct: 1.0
filled_order_expiration_min: 60
bollinger_bands_length: 7
bollinger_bands_std_dev: 2.0
bollinger_bands_bandwidth_threshold: 1.5
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 14
unfilled_order_expiration_min: 100
min_spread_pct: 0.5
normalized_bbp_mult: 0.04
normalized_bbb_mult: 0.1
`

const result = {
    'net_pnl': 0.10339837940832756,
    'net_pnl_quote': 10.339837940832755,
    'total_executors': 428,
    'total_executors_with_position': 157,
    'total_volume': 31380.948789329163,
    'total_long': 120,
    'total_short': 37,
    'close_types': {'STOP_LOSS': 6, 'TAKE_PROFIT': 34, 'TIME_LIMIT': 388},
    'accuracy_long': 0.55,
    'accuracy_short': 0.43243243243243246,
    'total_positions': 157,
    'accuracy': 0.5222929936305732,
    'max_drawdown_usd': -8.238171977495895,
    'max_drawdown_pct': -0.08229355788027619,
    'sharpe_ratio': 1.1286199543259523,
    'profit_factor': 1.2317020862747279,
    'win_signals': 82,
    'loss_signals': 75
}