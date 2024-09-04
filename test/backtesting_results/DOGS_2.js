/*
    start_time = datetime(2024, 8, 27).timestamp()
    end_time = datetime(2024, 9, 4).timestamp()
 */

const conf = `
id: oePNxNXqRNkXBD8397uozUvph2unKvBfkVxeDgoDTM8
controller_name: generic_pk
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
filled_order_expiration_min: 1000
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 1.0
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.7
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': 0.08765207792866295,
    'net_pnl_quote': 1.7530415585732588,
    'total_executors': 2131,
    'total_executors_with_position': 405,
    'total_volume': 16200.88154938143,
    'total_long': 206,
    'total_short': 199,
    'close_types': {'EARLY_STOP': 1724, 'STOP_LOSS': 156, 'TAKE_PROFIT': 248, 'TIME_LIMIT': 3},
    'accuracy_long': 0.616504854368932,
    'accuracy_short': 0.6080402010050251,
    'total_positions': 405,
    'accuracy': 0.6123456790123457,
    'max_drawdown_usd': -2.8316476183496,
    'max_drawdown_pct': -0.14061576420674407,
    'sharpe_ratio': 0.744023914959437,
    'profit_factor': 1.0445389932369065,
    'win_signals': 248,
    'loss_signals': 157
}