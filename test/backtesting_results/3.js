const conf = `
id: 2gk378dinHVrUTzBF357Ywjor3WMxjkmmqd1GkvaXdWh
controller_name: generic_pk
controller_type: generic
total_amount_quote: 100
manual_kill_switch: null
candles_config: []
connector_name: binance
trading_pair: AAVE-USDT
leverage: 20
position_mode: HEDGE
unfilled_order_expiration_min: 3
stop_loss_pct: 0.5
take_profit_pct: 0.3
filled_order_expiration_min: 1000
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
candles_count_for_trend: 12
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
volatility_threshold_bbb: 1.0
candles_connector: binance
candles_interval: 1m
candles_length: 32
default_spread_pct: 0.5
`

const result = {
    'net_pnl': -0.08972756908877089,
    'net_pnl_quote': -8.972756908877088,
    'total_executors': 2517,
    'total_executors_with_position': 457,
    'total_volume': 91444.21643278122,
    'total_long': 193,
    'total_short': 264,
    'close_types': {'EARLY_STOP': 2059, 'STOP_LOSS': 184, 'TAKE_PROFIT': 272, 'TIME_LIMIT': 2},
    'accuracy_long': 0.6424870466321243,
    'accuracy_short': 0.5606060606060606,
    'total_positions': 457,
    'accuracy': 0.5951859956236324,
    'max_drawdown_usd': -12.475271771817457,
    'max_drawdown_pct': -0.12538969267051456,
    'sharpe_ratio': -0.567545857034041,
    'profit_factor': 0.9268108573444126,
    'win_signals': 272,
    'loss_signals': 185
}