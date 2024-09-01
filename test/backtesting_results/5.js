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
connector_name: okx_perpetual
trading_pair: XCH-USDT
leverage: 20
position_mode: HEDGE
cooldown_time_min: 3
unfilled_order_expiration_min: 10
stop_loss_pct: 1.2
take_profit_pct: 0.8
filled_order_expiration_min: 1000
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 1.0
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.5
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': 0.07694933609727697,
    'net_pnl_quote': 5.386453526809388,
    'total_executors': 3534,
    'total_executors_with_position': 368,
    'total_volume': 51505.71880679545,
    'total_long': 198,
    'total_short': 170,
    'close_types': {'EARLY_STOP': 3164, 'STOP_LOSS': 148, 'TAKE_PROFIT': 219, 'TIME_LIMIT': 3},
    'accuracy_long': 0.5707070707070707,
    'accuracy_short': 0.6235294117647059,
    'total_positions': 368,
    'accuracy': 0.595108695652174,
    'max_drawdown_usd': -14.05691693935143,
    'max_drawdown_pct': -0.20357079909350784,
    'sharpe_ratio': -0.16223078557141643,
    'profit_factor': 1.0367073132112097,
    'win_signals': 219,
    'loss_signals': 149
}