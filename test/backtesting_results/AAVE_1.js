/*
    start_time = datetime(2024, 8, 27).timestamp()
    end_time = datetime(2024, 9, 4).timestamp()
 */

const conf = `
id: conf_generic.mm_bbands_AAVE
controller_name: mm_bbands
controller_type: generic
total_amount_quote: 20
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: AAVE-USDT
leverage: 20
position_mode: HEDGE
unfilled_order_expiration_min: 10
stop_loss_pct: 0.6
take_profit_pct: 0.4
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
    'net_pnl': -0.006184221563929799,
    'net_pnl_quote': -0.12368443127859598,
    'total_executors': 1943,
    'total_executors_with_position': 65,
    'total_volume': 2599.786027590415,
    'total_long': 31,
    'total_short': 34,
    'close_types': {'EARLY_STOP': 1876, 'STOP_LOSS': 26, 'TAKE_PROFIT': 39, 'TIME_LIMIT': 2},
    'accuracy_long': 0.6129032258064516,
    'accuracy_short': 0.5882352941176471,
    'total_positions': 65,
    'accuracy': 0.6,
    'max_drawdown_usd': -1.3610530985663063,
    'max_drawdown_pct': -0.06775120338439891,
    'sharpe_ratio': -0.2075058649301959,
    'profit_factor': 0.969174592135079,
    'win_signals': 39,
    'loss_signals': 26
}