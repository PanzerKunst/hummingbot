/*
    start_time = datetime(2024, 8, 1).timestamp()
    end_time = datetime(2024, 8, 15).timestamp()
 */

const conf = `
id: oePNxNXqRNkXBD8397uozUvph2unKvBfkVxeDgoDTM8
controller_name: generic_pk
controller_type: generic
total_amount_quote: 100
manual_kill_switch: null
candles_config: []
connector_name: binance
trading_pair: AAVE-USDT
leverage: 20
position_mode: HEDGE
unfilled_order_expiration_min: 10
stop_loss_pct: 0.5
take_profit_pct: 0.3
filled_order_expiration_min: 1000
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
candles_count_for_trend: 12
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
volatility_threshold_bbb: 1.0
candles_connector: binance
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.5
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': -0.7076989047395008,
    'net_pnl_quote': -70.76989047395008,
    'total_executors': 3965,
    'total_executors_with_position': 1137,
    'total_volume': 227427.02231022244,
    'total_long': 580,
    'total_short': 557,
    'close_types': {'EARLY_STOP': 2826, 'STOP_LOSS': 523, 'TAKE_PROFIT': 614, 'TIME_LIMIT': 2},
    'accuracy_long': 0.5379310344827586,
    'accuracy_short': 0.5421903052064632,
    'total_positions': 1137,
    'accuracy': 0.5400175901495162,
    'max_drawdown_usd': -73.76628425770429,
    'max_drawdown_pct': -0.7349258227862915,
    'sharpe_ratio': -1.6108645226320844,
    'profit_factor': 0.7975644822260041,
    'win_signals': 614,
    'loss_signals': 523
}