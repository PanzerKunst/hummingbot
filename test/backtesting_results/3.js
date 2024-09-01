/*
    start_time = datetime(2024, 8, 1).timestamp()
    end_time = datetime(2024, 8, 15).timestamp()
 */

const conf = `
id: 2gk378dinHVrUTzBF357Ywjor3WMxjkmmqd1GkvaXdWh
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
candles_length: 32
default_spread_pct: 1.0
`

const result = {
    'net_pnl': -0.712183632784376,
    'net_pnl_quote': -71.21836327843761,
    'total_executors': 3785,
    'total_executors_with_position': 848,
    'total_volume': 169631.91899594298,
    'total_long': 415,
    'total_short': 433,
    'close_types': {'EARLY_STOP': 2936, 'STOP_LOSS': 402, 'TAKE_PROFIT': 446, 'TIME_LIMIT': 1},
    'accuracy_long': 0.5132530120481927,
    'accuracy_short': 0.5381062355658198,
    'total_positions': 848,
    'accuracy': 0.5259433962264151,
    'max_drawdown_usd': -74.65500384109203,
    'max_drawdown_pct': -0.7506890190346378,
    'sharpe_ratio': -2.2322846343632725,
    'profit_factor': 0.7418418886605103,
    'win_signals': 446,
    'loss_signals': 402
}