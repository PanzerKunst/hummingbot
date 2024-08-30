const conf = `
id: CE3jnE44ocDRoAcMcU9yzXjN3kUkMF3nQdsRN7H4Bkcf
controller_name: generic_pk
controller_type: generic
total_amount_quote: 100
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
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
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 32
min_spread_pct: 0.1
`

const result = {
    'net_pnl': 0.08519488702047917,
    'net_pnl_quote': 8.519488702047916,
    'total_executors': 131,
    'total_executors_with_position': 124,
    'total_volume': 24832.438020577156,
    'total_long': 41,
    'total_short': 83,
    'close_types': {'STOP_LOSS': 38, 'TAKE_PROFIT': 85, 'TIME_LIMIT': 8},
    'accuracy_long': 0.6341463414634146,
    'accuracy_short': 0.7108433734939759,
    'total_positions': 124,
    'accuracy': 0.6854838709677419,
    'max_drawdown_usd': -7.724314613575521,
    'max_drawdown_pct': -0.07697583830529403,
    'sharpe_ratio': 0.9109207658729521,
    'profit_factor': 1.2812345152896079,
    'win_signals': 85,
    'loss_signals': 39
}