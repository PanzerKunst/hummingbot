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
filled_order_expiration_min: 240
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
    'net_pnl': 0.18445954234838385,
    'net_pnl_quote': 18.445954234838386,
    'total_executors': 136,
    'total_executors_with_position': 68,
    'total_volume': 13590.151440887084,
    'total_long': 57,
    'total_short': 11,
    'close_types': {'STOP_LOSS': 3, 'TAKE_PROFIT': 35, 'TIME_LIMIT': 98},
    'accuracy_long': 0.7192982456140351,
    'accuracy_short': 0.36363636363636365,
    'total_positions': 68,
    'accuracy': 0.6617647058823529,
    'max_drawdown_usd': -3.7597632470105595,
    'max_drawdown_pct': -0.037459116282897664,
    'sharpe_ratio': 0.8144660006232702,
    'profit_factor': 1.7428794466163906,
    'win_signals': 45,
    'loss_signals': 23
}