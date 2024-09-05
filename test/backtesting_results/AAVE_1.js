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
cooldown_time_min: 3
unfilled_order_expiration_min: 10
stop_loss_pct: 0.5
take_profit_pct: 0.3
filled_order_expiration_min: 90
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 1.0
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.2
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': -0.10496709091215171,
    'net_pnl_quote': -2.099341818243034,
    'total_executors': 1986,
    'total_executors_with_position': 170,
    'total_volume': 6802.420516976788,
    'total_long': 81,
    'total_short': 89,
    'close_types': {'EARLY_STOP': 1814, 'STOP_LOSS': 76, 'TAKE_PROFIT': 94, 'TIME_LIMIT': 2},
    'accuracy_long': 0.48148148148148145,
    'accuracy_short': 0.6179775280898876,
    'total_positions': 170,
    'accuracy': 0.5529411764705883,
    'max_drawdown_usd': -2.368063310687428,
    'max_drawdown_pct': -0.11912869677077466,
    'sharpe_ratio': -1.5025003775542476,
    'profit_factor': 0.7765408864908323,
    'win_signals': 94,
    'loss_signals': 76
}