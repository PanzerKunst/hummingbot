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
stop_loss_pct: 3.0
take_profit_pct: 1.0
filled_order_expiration_min: 30
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
    'net_pnl': 0.09596578775738819,
    'net_pnl_quote': 9.59657877573882,
    'total_executors': 753,
    'total_executors_with_position': 145,
    'total_volume': 28968.237365377267,
    'total_long': 141,
    'total_short': 4,
    'close_types': {'TAKE_PROFIT': 11, 'TIME_LIMIT': 742},
    'accuracy_long': 0.5531914893617021,
    'accuracy_short': 0.75,
    'total_positions': 145,
    'accuracy': 0.5586206896551724,
    'max_drawdown_usd': -5.964537476433764,
    'max_drawdown_pct': -0.05923554013226237,
    'sharpe_ratio': 1.7030215524177674,
    'profit_factor': 1.3486742412984207,
    'win_signals': 81,
    'loss_signals': 64
}