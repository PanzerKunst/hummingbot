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
default_spread_pct: 0.5
`

const result = {
    'net_pnl': -0.9461268155565451,
    'net_pnl_quote': -94.6126815556545,
    'total_executors': 4669,
    'total_executors_with_position': 2170,
    'total_volume': 434028.2297761045,
    'total_long': 1100,
    'total_short': 1070,
    'close_types': {'EARLY_STOP': 2497, 'STOP_LOSS': 953, 'TAKE_PROFIT': 1215, 'TIME_LIMIT': 4},
    'accuracy_long': 0.5590909090909091,
    'accuracy_short': 0.5607476635514018,
    'total_positions': 2170,
    'accuracy': 0.5599078341013825,
    'max_drawdown_usd': -101.23155427469482,
    'max_drawdown_pct': -1.008526781855837,
    'sharpe_ratio': -1.6059357525057467,
    'profit_factor': 0.8482413310708118,
    'win_signals': 1215,
    'loss_signals': 955
}