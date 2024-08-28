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
normalized_bbp_mult: 0.01
`

const result = {
    'net_pnl': -0.04708068193680664,
    'net_pnl_quote': -4.708068193680664,
    'total_executors': 2104,
    'total_executors_with_position': 132,
    'total_volume': 26396.37166229721,
    'total_long': 74,
    'total_short': 58,
    'close_types': {'EARLY_STOP': 1970, 'STOP_LOSS': 15, 'TAKE_PROFIT': 40, 'TIME_LIMIT': 79},
    'accuracy_long': 0.581081081081081,
    'accuracy_short': 0.5517241379310345,
    'total_positions': 132,
    'accuracy': 0.5681818181818182,
    'max_drawdown_usd': -18.175449676919026,
    'max_drawdown_pct': -0.17935656805640407,
    'sharpe_ratio': 0.6503233984834197,
    'profit_factor': 0.9257065288998405,
    'win_signals': 75,
    'loss_signals': 57
}