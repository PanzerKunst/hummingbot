/*
    start_time = datetime(2024, 8, 27).timestamp()
    end_time = datetime(2024, 9, 4).timestamp()
 */

const conf = `
id: conf_generic.mm_bbands_DOGS
controller_name: mm_bbands
controller_type: generic
total_amount_quote: 20
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: DOGS-USDT
leverage: 20
position_mode: HEDGE
cooldown_time_min: 3
unfilled_order_expiration_min: 10
stop_loss_pct: 0.9
take_profit_pct: 0.6
filled_order_expiration_min: 90
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 1.5
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.4
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': -0.026190599493182726,
    'net_pnl_quote': -0.5238119898636545,
    'total_executors': 2321,
    'total_executors_with_position': 654,
    'total_volume': 26152.67662505832,
    'total_long': 334,
    'total_short': 320,
    'close_types': {'EARLY_STOP': 1648, 'STOP_LOSS': 335, 'TAKE_PROFIT': 324, 'TIME_LIMIT': 14},
    'accuracy_long': 0.4940119760479042,
    'accuracy_short': 0.53125,
    'total_positions': 654,
    'accuracy': 0.5122324159021406,
    'max_drawdown_usd': -5.921155181940995,
    'max_drawdown_pct': -0.30275453049520773,
    'sharpe_ratio': -0.27421289329154935,
    'profit_factor': 0.9903691429700389,
    'win_signals': 335,
    'loss_signals': 319
}