// start_time = datetime(2024, 8, 2).timestamp()
// end_time = datetime(2024, 8, 9).timestamp()

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
    'net_pnl': 0.0538678368219742,
    'net_pnl_quote': 5.38678368219742,
    'total_executors': 338,
    'total_executors_with_position': 3,
    'total_volume': 595.0850359872712,
    'total_long': 3,
    'total_short': 0,
    'close_types': {'TAKE_PROFIT': 3, 'TIME_LIMIT': 335},
    'accuracy_long': 1.0,
    'accuracy_short': 0.0,
    'total_positions': 3,
    'accuracy': 1.0,
    'max_drawdown_usd': 0.0,
    'max_drawdown_pct': 0.0,
    'sharpe_ratio': 4.129817772095078,
    'profit_factor': 1.0,
    'win_signals': 3,
    'loss_signals': 0
}