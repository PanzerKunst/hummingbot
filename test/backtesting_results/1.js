const conf = `
id: CE3jnE44ocDRoAcMcU9yzXjN3kUkMF3nQdsRN7H4Bkcf
controller_name: generic_pk
controller_type: generic
total_amount_quote: 100
manual_kill_switch: null
candles_config: []
connector_name: binance
trading_pair: AAVE-USDT
leverage: 20
position_mode: HEDGE
stop_loss_pct: 0.5
take_profit_pct: 0.3
filled_order_expiration_min: 1000
bollinger_bands_length: 12
bollinger_bands_std_dev: 2.0
candles_count_for_trend: 16
volatility_threshold_bbp: 0.5
volatility_threshold_bbb: 1.0
candles_connector: binance
candles_interval: 1m
candles_length: 32
min_spread_pct: 0.2
`

const result = {
    'net_pnl': 0.06940890973848218,
    'net_pnl_quote': 6.9408909738482185,
    'total_executors': 94,
    'total_executors_with_position': 87,
    'total_volume': 17427.481118735097,
    'total_long': 25,
    'total_short': 62,
    'close_types': {'STOP_LOSS': 25, 'TAKE_PROFIT': 61, 'TIME_LIMIT': 8},
    'accuracy_long': 0.68,
    'accuracy_short': 0.7096774193548387,
    'total_positions': 87,
    'accuracy': 0.7011494252873564,
    'max_drawdown_usd': -5.080299581635364,
    'max_drawdown_pct': -0.050618968622591104,
    'sharpe_ratio': 1.1864663509080642,
    'profit_factor': 1.3337733701938967,
    'win_signals': 61,
    'loss_signals': 26
}
