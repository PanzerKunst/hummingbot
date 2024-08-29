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
unfilled_order_expiration_min: 10
min_spread_pct: 0.5
normalized_bbp_mult: 0.04
normalized_bbb_mult: 0.1
`

const result = {
    'net_pnl': 0.12347756243649688,
    'net_pnl_quote': 12.347756243649687,
    'total_executors': 1756,
    'total_executors_with_position': 179,
    'total_volume': 35770.58485571133,
    'total_long': 158,
    'total_short': 21,
    'close_types': {'EARLY_STOP': 1576, 'STOP_LOSS': 14, 'TAKE_PROFIT': 57, 'TIME_LIMIT': 109},
    'accuracy_long': 0.569620253164557,
    'accuracy_short': 0.47619047619047616,
    'total_positions': 179,
    'accuracy': 0.5586592178770949,
    'max_drawdown_usd': -7.564931893542653,
    'max_drawdown_pct': -0.07556836180917295,
    'sharpe_ratio': 1.1532806559106628,
    'profit_factor': 1.1825300611785705,
    'win_signals': 100,
    'loss_signals': 79
}