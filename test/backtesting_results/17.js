// start_time = datetime(2024, 7, 1).timestamp()
// end_time = datetime(2024, 8, 25).timestamp()

const conf = `
id: A1psBfboZZ1He9h85baqLqDEVX6mpMouGe3rVzpXnAqr
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
    'net_pnl': 0.20059268231491756,
    'net_pnl_quote': 20.059268231491757,
    'total_executors': 1896,
    'total_executors_with_position': 788,
    'total_volume': 157386.0630041616,
    'total_long': 788,
    'total_short': 0,
    'close_types': {'STOP_LOSS': 82, 'TAKE_PROFIT': 313, 'TIME_LIMIT': 1501},
    'accuracy_long': 0.5901015228426396,
    'accuracy_short': 0.0,
    'total_positions': 788,
    'accuracy': 0.5901015228426396,
    'max_drawdown_usd': -72.1396689853403,
    'max_drawdown_pct': -0.7136970493442887,
    'sharpe_ratio': -0.3745348475732628,
    'profit_factor': 1.0509113319886139,
    'win_signals': 465,
    'loss_signals': 323
}