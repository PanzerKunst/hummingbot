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
filled_order_expiration_min: 60
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
    'net_pnl': 0.14774953539392863,
    'net_pnl_quote': 14.774953539392865,
    'total_executors': 417,
    'total_executors_with_position': 129,
    'total_volume': 25774.051953757986,
    'total_long': 120,
    'total_short': 9,
    'close_types': {'STOP_LOSS': 1, 'TAKE_PROFIT': 31, 'TIME_LIMIT': 385},
    'accuracy_long': 0.5583333333333333,
    'accuracy_short': 0.3333333333333333,
    'total_positions': 129,
    'accuracy': 0.5426356589147286,
    'max_drawdown_usd': -7.473049465796866,
    'max_drawdown_pct': -0.07465052082375367,
    'sharpe_ratio': 3.07975235740127,
    'profit_factor': 1.4272217468720012,
    'win_signals': 70,
    'loss_signals': 59
}