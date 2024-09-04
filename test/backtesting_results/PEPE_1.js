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
unfilled_order_expiration_min: 10
stop_loss_pct: 0.9
take_profit_pct: 0.6
filled_order_expiration_min: 1000
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 1.0
candles_connector: okx_perpetual
candles_interval: 1m
candles_length: 24
default_spread_pct: 0.7
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': -0.049000984512315014,
    'net_pnl_quote': -0.9800196902463003,
    'total_executors': 1963,
    'total_executors_with_position': 116,
    'total_volume': 4639.9451986620315,
    'total_long': 61,
    'total_short': 55,
    'close_types': {'EARLY_STOP': 1845, 'STOP_LOSS': 50, 'TAKE_PROFIT': 66, 'TIME_LIMIT': 2},
    'accuracy_long': 0.5901639344262295,
    'accuracy_short': 0.5454545454545454,
    'total_positions': 116,
    'accuracy': 0.5689655172413793,
    'max_drawdown_usd': -1.5663939275972933,
    'max_drawdown_pct': -0.07907517603144419,
    'sharpe_ratio': -0.746385741655363,
    'profit_factor': 0.9084771740678129,
    'win_signals': 66,
    'loss_signals': 50
}