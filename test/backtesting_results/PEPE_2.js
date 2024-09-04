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
    'net_pnl': -0.13525367780933006,
    'net_pnl_quote': -2.7050735561866013,
    'total_executors': 2024,
    'total_executors_with_position': 249,
    'total_volume': 9958.73719307657,
    'total_long': 133,
    'total_short': 116,
    'close_types': {'EARLY_STOP': 1773, 'STOP_LOSS': 108, 'TAKE_PROFIT': 141, 'TIME_LIMIT': 2},
    'accuracy_long': 0.5413533834586466,
    'accuracy_short': 0.5948275862068966,
    'total_positions': 249,
    'accuracy': 0.5662650602409639,
    'max_drawdown_usd': -3.9946343622809835,
    'max_drawdown_pct': -0.1988788746340035,
    'sharpe_ratio': -1.5991389015323672,
    'profit_factor': 0.8323408151471522,
    'win_signals': 141,
    'loss_signals': 108
}