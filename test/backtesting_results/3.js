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
normalized_bbp_mult: 0.02
`

const result = {
    'net_pnl': -0.010169159097650331,
    'net_pnl_quote': -1.016915909765033,
    'total_executors': 2096,
    'total_executors_with_position': 119,
    'total_volume': 23792.968413928676,
    'total_long': 74,
    'total_short': 45,
    'close_types': {'EARLY_STOP': 1975, 'STOP_LOSS': 13, 'TAKE_PROFIT': 37, 'TIME_LIMIT': 71},
    'accuracy_long': 0.581081081081081,
    'accuracy_short': 0.5333333333333333,
    'total_positions': 119,
    'accuracy': 0.5630252100840336,
    'max_drawdown_usd': -11.992853366942029,
    'max_drawdown_pct': -0.11834628905110171,
    'sharpe_ratio': 0.7298639606187669,
    'profit_factor': 0.9813884273559867,
    'win_signals': 67,
    'loss_signals': 52
}