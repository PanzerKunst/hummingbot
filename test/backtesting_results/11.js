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
bollinger_bands_std_dev: 2.0
candles_interval: 1m
candles_length: 7
unfilled_order_expiration_min: 5
min_spread_pct: 0.5
normalized_bbp_mult: 0.03
`

const result = {
    'net_pnl': 0.0056274602015089735,
    'net_pnl_quote': 0.5627460201508974,
    'total_executors': 2090,
    'total_executors_with_position': 108,
    'total_volume': 21592.416821451163,
    'total_long': 74,
    'total_short': 34,
    'close_types': {'EARLY_STOP': 1980, 'STOP_LOSS': 10, 'TAKE_PROFIT': 34, 'TIME_LIMIT': 66},
    'accuracy_long': 0.581081081081081,
    'accuracy_short': 0.5,
    'total_positions': 108,
    'accuracy': 0.5555555555555556,
    'max_drawdown_usd': -9.49944945590202,
    'max_drawdown_pct': -0.0937412104306558,
    'sharpe_ratio': 0.7044669468942524,
    'profit_factor': 1.0114061184878942,
    'win_signals': 60,
    'loss_signals': 48
}