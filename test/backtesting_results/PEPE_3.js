/*
    start_time = datetime(2024, 8, 27).timestamp()
    end_time = datetime(2024, 9, 4).timestamp()
 */

const conf = `
id: conf_generic.mm_bbands_PEPE
controller_name: mm_bbands
controller_type: generic
total_amount_quote: 20
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: PEPE-USDT
leverage: 20
position_mode: HEDGE
cooldown_time_min: 3
unfilled_order_expiration_min: 10
stop_loss_pct: 0.7
take_profit_pct: 0.4
filled_order_expiration_min: 90
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 1.0
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.3
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': -0.10204563087771235,
    'net_pnl_quote': -2.040912617554247,
    'total_executors': 2142,
    'total_executors_with_position': 405,
    'total_volume': 16198.460154560651,
    'total_long': 208,
    'total_short': 197,
    'close_types': {'EARLY_STOP': 1735, 'STOP_LOSS': 147, 'TAKE_PROFIT': 234, 'TIME_LIMIT': 26},
    'accuracy_long': 0.5769230769230769,
    'accuracy_short': 0.6192893401015228,
    'total_positions': 405,
    'accuracy': 0.5975308641975309,
    'max_drawdown_usd': -3.7632777850553265,
    'max_drawdown_pct': -0.18734901196938733,
    'sharpe_ratio': -0.5432987791445331,
    'profit_factor': 0.9197431032908548,
    'win_signals': 242,
    'loss_signals': 163
}