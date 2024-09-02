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
    'net_pnl': 0.22700603869202599,
    'net_pnl_quote': 15.89042270844182,
    'total_executors': 3578,
    'total_executors_with_position': 450,
    'total_volume': 62995.10758095916,
    'total_long': 233,
    'total_short': 217,
    'close_types': {'EARLY_STOP': 3126, 'STOP_LOSS': 174, 'TAKE_PROFIT': 275, 'TIME_LIMIT': 3},
    'accuracy_long': 0.5965665236051502,
    'accuracy_short': 0.6267281105990783,
    'total_positions': 450,
    'accuracy': 0.6111111111111112,
    'max_drawdown_usd': -15.233211641217009,
    'max_drawdown_pct': -0.21578153126839505,
    'sharpe_ratio': 0.7072488786973844,
    'profit_factor': 1.0953652581428501,
    'win_signals': 275,
    'loss_signals': 175
}