/*
    start_time = datetime(2024, 8, 1).timestamp()
    end_time = datetime(2024, 8, 15).timestamp()
 */

const conf = `
id: oePNxNXqRNkXBD8397uozUvph2unKvBfkVxeDgoDTM8
controller_name: generic_pk
controller_type: generic
total_amount_quote: 70
manual_kill_switch: null
candles_config: []
connector_name: binance
trading_pair: AAVE-USDT
leverage: 20
position_mode: HEDGE
cooldown_time_min: 3
unfilled_order_expiration_min: 10
stop_loss_pct: 1.6
take_profit_pct: 0.8
filled_order_expiration_min: 1000
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 1.0
candles_connector: binance
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.5
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': -0.4799693039867544,
    'net_pnl_quote': -33.59785127907281,
    'total_executors': 3650,
    'total_executors_with_position': 567,
    'total_volume': 79399.67142744905,
    'total_long': 260,
    'total_short': 307,
    'close_types': {'EARLY_STOP': 3082, 'STOP_LOSS': 215, 'TAKE_PROFIT': 349, 'TIME_LIMIT': 4},
    'accuracy_long': 0.5961538461538461,
    'accuracy_short': 0.6319218241042345,
    'total_positions': 567,
    'accuracy': 0.6155202821869489,
    'max_drawdown_usd': -44.787007904369595,
    'max_drawdown_pct': -0.6346786651287292,
    'sharpe_ratio': -1.4511604554436182,
    'profit_factor': 0.8732253188556346,
    'win_signals': 349,
    'loss_signals': 218
}