const conf = `
id: 7iXbZ3sikepy6nJEs1S6WpbsqPitPXM6RNThJXjrZwBQ
controller_name: generic_pk
controller_type: generic
total_amount_quote: 100
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: ETH-USDT
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
    'net_pnl': 0.06373075829747724,
    'net_pnl_quote': 6.373075829747723,
    'total_executors': 213,
    'total_executors_with_position': 66,
    'total_volume': 13181.73795760644,
    'total_long': 66,
    'total_short': 0,
    'close_types': {'STOP_LOSS': 5, 'TAKE_PROFIT': 19, 'TIME_LIMIT': 189},
    'accuracy_long': 0.6212121212121212,
    'accuracy_short': 0.0,
    'total_positions': 66,
    'accuracy': 0.6212121212121212,
    'max_drawdown_usd': -13.485349806120116,
    'max_drawdown_pct': -0.1344737667340301,
    'sharpe_ratio': 1.3551256168680699,
    'profit_factor': 1.269854416641614,
    'win_signals': 41,
    'loss_signals': 25
}