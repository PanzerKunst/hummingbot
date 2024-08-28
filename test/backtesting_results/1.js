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
unfilled_order_expiration_min: 10
min_spread_pct: 0.5
normalized_bbp_mult: 0.01
`

const result = {
    'net_pnl': 0.021773685751283318,
    'net_pnl_quote': 2.1773685751283316,
    'total_executors': 1270,
    'total_executors_with_position': 168,
    'total_volume': 33602.295428882346,
    'total_long': 91,
    'total_short': 77,
    'close_types': {'EARLY_STOP': 1100, 'STOP_LOSS': 11, 'TAKE_PROFIT': 47, 'TIME_LIMIT': 112},
    'accuracy_long': 0.5494505494505495,
    'accuracy_short': 0.5194805194805194,
    'total_positions': 168,
    'accuracy': 0.5357142857142857,
    'max_drawdown_usd': -12.24789843349171,
    'max_drawdown_pct': -0.12201892724806596,
    'sharpe_ratio': 0.853577488644082,
    'profit_factor': 1.032614711562655,
    'win_signals': 90,
    'loss_signals': 78
}