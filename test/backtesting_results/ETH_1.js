/*
    start_time = datetime(2024, 8, 27).timestamp()
    end_time = datetime(2024, 9, 4).timestamp()
 */

const conf = `
id: conf_generic.mm_bbands_ETH
controller_name: mm_bbands
controller_type: generic
total_amount_quote: 20
manual_kill_switch: null
candles_config: []
connector_name: okx_perpetual
trading_pair: ETH-USDT
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
    'net_pnl': -0.02190800841761148,
    'net_pnl_quote': -0.43816016835222965,
    'total_executors': 1944,
    'total_executors_with_position': 66,
    'total_volume': 2638.9931578646706,
    'total_long': 35,
    'total_short': 31,
    'close_types': {'EARLY_STOP': 1876, 'STOP_LOSS': 28, 'TAKE_PROFIT': 38, 'TIME_LIMIT': 2},
    'accuracy_long': 0.6571428571428571,
    'accuracy_short': 0.4838709677419355,
    'total_positions': 66,
    'accuracy': 0.5757575757575758,
    'max_drawdown_usd': -0.9912034828944953,
    'max_drawdown_pct': -0.04934797342391364,
    'sharpe_ratio': -0.6549829549540079,
    'profit_factor': 0.8946851001803874,
    'win_signals': 38,
    'loss_signals': 28
}