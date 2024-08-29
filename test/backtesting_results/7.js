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
unfilled_order_expiration_min: 1000
min_spread_pct: 0.5
normalized_bbp_mult: 0.1
normalized_bbb_mult: 0.1
`

const result = {
    'net_pnl': 0.10675041465101967,
    'net_pnl_quote': 10.675041465101968,
    'total_executors': 417,
    'total_executors_with_position': 129,
    'total_volume': 25774.051953757986,
    'total_long': 120,
    'total_short': 9,
    'close_types': {'STOP_LOSS': 5, 'TAKE_PROFIT': 30, 'TIME_LIMIT': 382},
    'accuracy_long': 0.55,
    'accuracy_short': 0.3333333333333333,
    'total_positions': 129,
    'accuracy': 0.5348837209302325,
    'max_drawdown_usd': -7.993828323812737,
    'max_drawdown_pct': -0.07985273621959932,
    'sharpe_ratio': 2.333064352776725,
    'profit_factor': 1.2873554404391188,
    'win_signals': 69,
    'loss_signals': 60
}