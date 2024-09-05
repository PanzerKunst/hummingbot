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
unfilled_order_expiration_min: 10
stop_loss_pct: 0.9
take_profit_pct: 0.6
filled_order_expiration_min: 120
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
    'net_pnl': 0.028543073944970163,
    'net_pnl_quote': 0.5708614788994033,
    'total_executors': 2249,
    'total_executors_with_position': 553,
    'total_volume': 22118.25870509392,
    'total_long': 272,
    'total_short': 281,
    'close_types': {'EARLY_STOP': 1694, 'STOP_LOSS': 218, 'TAKE_PROFIT': 327, 'TIME_LIMIT': 10},
    'accuracy_long': 0.5698529411764706,
    'accuracy_short': 0.6227758007117438,
    'total_positions': 553,
    'accuracy': 0.596745027124774,
    'max_drawdown_usd': -3.4120408807925813,
    'max_drawdown_pct': -0.1744610305173905,
    'sharpe_ratio': -0.33462688117741407,
    'profit_factor': 1.0104995955649974,
    'win_signals': 330,
    'loss_signals': 223
}