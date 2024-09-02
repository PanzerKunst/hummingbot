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
trading_pair: ONE-USDT
leverage: 20
position_mode: HEDGE
unfilled_order_expiration_min: 10
stop_loss_pct: 1.2
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
    'net_pnl': 0.09139464078446669,
    'net_pnl_quote': 6.397624854912668,
    'total_executors': 3665,
    'total_executors_with_position': 607,
    'total_volume': 84971.36443206285,
    'total_long': 308,
    'total_short': 299,
    'close_types': {'EARLY_STOP': 3056, 'STOP_LOSS': 243, 'TAKE_PROFIT': 363, 'TIME_LIMIT': 3},
    'accuracy_long': 0.5746753246753247,
    'accuracy_short': 0.6220735785953178,
    'total_positions': 607,
    'accuracy': 0.5980230642504119,
    'max_drawdown_usd': -27.21574466464898,
    'max_drawdown_pct': -0.3855165408755808,
    'sharpe_ratio': -0.30375090221299106,
    'profit_factor': 1.0273731992016153,
    'win_signals': 363,
    'loss_signals': 244
}