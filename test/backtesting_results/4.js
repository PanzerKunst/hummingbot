/*
    start_time = datetime(2024, 8, 1).timestamp()
    end_time = datetime(2024, 8, 15).timestamp()
 */

const conf = `
id: oePNxNXqRNkXBD8397uozUvph2unKvBfkVxeDgoDTM8
controller_name: generic_pk
controller_type: generic
total_amount_quote: 70
manual_kill_switch: null
candles_config: []
connector_name: binance
trading_pair: AAVE-USDT
leverage: 20
position_mode: HEDGE
cooldown_time_min: 3
unfilled_order_expiration_min: 30
stop_loss_pct: 1.4
take_profit_pct: 0.8
filled_order_expiration_min: 1000
bbands_length_for_trend: 12
bbands_std_dev_for_trend: 2.0
bbands_length_for_volatility: 2
bbands_std_dev_for_volatility: 3.0
high_volatility_threshold: 1.0
candles_connector: binance
candles_interval: 1m
candles_length: 24
default_spread_pct: 1.0
price_adjustment_volatility_threshold: 0.5
`

const result = {
    'net_pnl': -0.3766906780526188,
    'net_pnl_quote': -26.368347463683317,
    'total_executors': 1372,
    'total_executors_with_position': 230,
    'total_volume': 32203.012262497745,
    'total_long': 112,
    'total_short': 118,
    'close_types': {'EARLY_STOP': 1140, 'STOP_LOSS': 100, 'TAKE_PROFIT': 130, 'TIME_LIMIT': 2},
    'accuracy_long': 0.5357142857142857,
    'accuracy_short': 0.5932203389830508,
    'total_positions': 230,
    'accuracy': 0.5652173913043478,
    'max_drawdown_usd': -32.22997307256554,
    'max_drawdown_pct': -0.4555114398707625,
    'sharpe_ratio': -1.514309266293118,
    'profit_factor': 0.7693171614432451,
    'win_signals': 130,
    'loss_signals': 100
}