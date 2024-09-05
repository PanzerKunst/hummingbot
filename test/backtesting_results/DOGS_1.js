/*
    start_time = datetime(2024, 8, 27).timestamp()
    end_time = datetime(2024, 9, 4).timestamp()
 */

const conf = `
id: oePNxNXqRNkXBD8397uozUvph2unKvBfkVxeDgoDTM8
controller_name: mm_bbands
controller_type: generic
total_amount_quote: 20
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: DOGS-USDT
leverage: 20
position_mode: HEDGE
unfilled_order_expiration_min: 10
stop_loss_pct: 0.9
take_profit_pct: 0.6
filled_order_expiration_min: 120
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 3.0
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.5
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': 0.12633859334837336,
    'net_pnl_quote': 2.5267718669674673,
    'total_executors': 2188,
    'total_executors_with_position': 485,
    'total_volume': 19398.94638497583,
    'total_long': 247,
    'total_short': 238,
    'close_types': {'EARLY_STOP': 1701, 'STOP_LOSS': 190, 'TAKE_PROFIT': 291, 'TIME_LIMIT': 6},
    'accuracy_long': 0.5991902834008097,
    'accuracy_short': 0.6092436974789915,
    'total_positions': 485,
    'accuracy': 0.6041237113402061,
    'max_drawdown_usd': -2.7250134508823227,
    'max_drawdown_pct': -0.1393358049812371,
    'sharpe_ratio': 0.5848999981674761,
    'profit_factor': 1.0554452230977533,
    'win_signals': 293,
    'loss_signals': 192
}