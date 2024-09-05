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
cooldown_time_min: 30
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
default_spread_pct: 0.3
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': 0.02812550955737609,
    'net_pnl_quote': 0.5625101911475218,
    'total_executors': 2345,
    'total_executors_with_position': 664,
    'total_volume': 26558.66989491748,
    'total_long': 325,
    'total_short': 339,
    'close_types': {'EARLY_STOP': 1679, 'STOP_LOSS': 263, 'TAKE_PROFIT': 386, 'TIME_LIMIT': 17},
    'accuracy_long': 0.5661538461538461,
    'accuracy_short': 0.5988200589970502,
    'total_positions': 664,
    'accuracy': 0.5828313253012049,
    'max_drawdown_usd': -4.508314777923625,
    'max_drawdown_pct': -0.23050937117938652,
    'sharpe_ratio': -0.15475010188270538,
    'profit_factor': 1.0087342788105993,
    'win_signals': 387,
    'loss_signals': 277
}