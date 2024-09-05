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
stop_loss_pct: 0.8
take_profit_pct: 0.5
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
    'net_pnl': -0.1778473149004928,
    'net_pnl_quote': -3.5569462980098563,
    'total_executors': 2272,
    'total_executors_with_position': 588,
    'total_volume': 23517.29081450707,
    'total_long': 292,
    'total_short': 296,
    'close_types': {'EARLY_STOP': 1682, 'STOP_LOSS': 239, 'TAKE_PROFIT': 340, 'TIME_LIMIT': 11},
    'accuracy_long': 0.565068493150685,
    'accuracy_short': 0.6148648648648649,
    'total_positions': 588,
    'accuracy': 0.5901360544217688,
    'max_drawdown_usd': -5.334443399810153,
    'max_drawdown_pct': -0.27275537582404075,
    'sharpe_ratio': -0.40439680110476295,
    'profit_factor': 0.9330653148697646,
    'win_signals': 347,
    'loss_signals': 241
}