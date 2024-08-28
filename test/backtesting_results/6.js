const conf = `
id: C1PkH8bFzG3xasPijGM9cUm4bPxTLXiem35grtJUspiL
controller_name: generic_pk
controller_type: generic
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: AAVE-USDT
leverage: 5
position_mode: HEDGE
stop_loss_pct: 2.0
take_profit_pct: 1.0
filled_order_expiration_min: 60
bollinger_bands_length: 7
bollinger_bands_std_dev: 1.5
candles_interval: 1m
candles_length: 7
unfilled_order_expiration_min: 5
min_spread_pct: 0.5
normalized_bbp_mult: 0.02
`

const result = {
    'net_pnl': 0.024342825809366992,
    'net_pnl_quote': 2.4342825809366992,
    'total_executors': 2092,
    'total_executors_with_position': 115,
    'total_volume': 22993.25564229289,
    'total_long': 74,
    'total_short': 41,
    'close_types': {'EARLY_STOP': 1975, 'STOP_LOSS': 10, 'TAKE_PROFIT': 36, 'TIME_LIMIT': 71},
    'accuracy_long': 0.581081081081081,
    'accuracy_short': 0.5121951219512195,
    'total_positions': 115,
    'accuracy': 0.5565217391304348,
    'max_drawdown_usd': -8.964309352222434,
    'max_drawdown_pct': -0.0884604116536547,
    'sharpe_ratio': 0.7998445662737201,
    'profit_factor': 1.0482018821281618,
    'win_signals': 64,
    'loss_signals': 51
}