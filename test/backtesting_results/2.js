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
bollinger_bands_std_dev: 2.2
candles_interval: 1m
candles_length: 7
unfilled_order_expiration_min: 5
min_spread_pct: 0.5
normalized_bbp_mult: 0.01
`

const results = {
    'net_pnl': -0.0583335582726938,
    'net_pnl_quote': -5.83335582726938,
    'total_executors': 2106,
    'total_executors_with_position': 136,
    'total_volume': 27196.821809375953,
    'total_long': 74,
    'total_short': 62,
    'close_types': {'EARLY_STOP': 1968, 'STOP_LOSS': 15, 'TAKE_PROFIT': 42, 'TIME_LIMIT': 81},
    'accuracy_long': 0.581081081081081,
    'accuracy_short': 0.5161290322580645,
    'total_positions': 136,
    'accuracy': 0.5514705882352942,
    'max_drawdown_usd': -16.060255452178534,
    'max_drawdown_pct': -0.15848368822862477,
    'sharpe_ratio': 0.5319968507552517,
    'profit_factor': 0.9102538451719188,
    'win_signals': 75,
    'loss_signals': 61
}