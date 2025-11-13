# Exchange Manager API Reference

The Exchange Manager provides unified multi-exchange coordination and trading execution across NeuroFlux's supported trading platforms.

## Overview

The Exchange Manager serves as the central hub for all exchange interactions, providing a unified interface for order routing, balance management, and cross-exchange arbitrage opportunities.

## Core Functions

### `initialize_exchanges(exchange_configs)`

Initialize and configure multiple exchange connections.

**Parameters:**
- `exchange_configs` (dict): Configuration for each exchange

**Returns:**
```python
{
    'initialized_exchanges': list,    # Successfully initialized exchanges
    'failed_exchanges': list,         # Exchanges that failed to initialize
    'status': str,                    # 'partial_success', 'full_success', 'failed'
    'active_connections': int,        # Number of active connections
    'connection_health': dict         # Health status for each exchange
}
```

**Exchange Config Structure:**
```python
{
    'hyperliquid': {
        'api_key': str,               # API key for authentication
        'secret_key': str,            # Secret key for authentication
        'testnet': bool,              # Whether to use testnet
        'max_leverage': float,        # Maximum allowed leverage
        'rate_limits': dict           # Custom rate limiting settings
    },
    'binance': {
        'api_key': str,
        'secret_key': str,
        'subaccount': str,            # Optional subaccount
        'default_leverage': float
    }
}
```

### `route_order(order_request, routing_strategy='optimal')`

Route orders to the most suitable exchange based on specified strategy.

**Parameters:**
- `order_request` (dict): Order details and requirements
- `routing_strategy` (str): Strategy for order routing

**Returns:**
```python
{
    'order_id': str,                  # Unique order identifier
    'exchange': str,                  # Exchange where order was placed
    'status': str,                    # 'routed', 'pending', 'failed'
    'routing_reason': str,            # Reason for exchange selection
    'estimated_costs': dict,          # Estimated fees and costs
    'execution_time': float,          # Estimated execution time
    'backup_exchanges': list          # Alternative exchanges if needed
}
```

**Routing Strategies:**
- `'optimal'`: Best price and liquidity combination
- `'fastest'`: Lowest latency execution
- `'cheapest'`: Lowest fee structure
- `'balanced'`: Balance between cost and speed
- `'redundant'`: Place on multiple exchanges for redundancy

### `get_unified_balance(asset_filter=None)`

Get consolidated balance across all connected exchanges.

**Parameters:**
- `asset_filter` (list, optional): Specific assets to include

**Returns:**
```python
{
    'total_balance_usd': float,       # Total balance in USD
    'asset_breakdown': dict,         # Balance by asset
    'exchange_breakdown': dict,      # Balance by exchange
    'available_balance': dict,       # Available for trading
    'locked_balance': dict,          # Locked in positions/orders
    'last_updated': str,             # Timestamp of last update
    'sync_status': dict              # Sync status for each exchange
}
```

**Asset Breakdown Structure:**
```python
{
    'BTC': {
        'total': float,               # Total BTC across exchanges
        'available': float,           # Available for trading
        'in_positions': float,        # Locked in positions
        'in_orders': float,           # Reserved in open orders
        'value_usd': float            # Current USD value
    }
}
```

### `execute_cross_exchange_arbitrage(opportunity)`

Execute arbitrage opportunities between exchanges.

**Parameters:**
- `opportunity` (dict): Arbitrage opportunity details

**Returns:**
```python
{
    'arbitrage_id': str,             # Unique arbitrage execution ID
    'status': str,                   # 'executed', 'partial', 'failed'
    'profit_realized': float,        # Realized profit in USD
    'execution_details': dict,       # Details of both legs
    'fees_paid': float,              # Total fees paid
    'execution_time': float,         # Time taken for execution
    'risk_assessment': dict          # Post-execution risk assessment
}
```

### `monitor_exchange_health()`

Monitor health and connectivity of all exchanges.

**Returns:**
```python
{
    'overall_health': str,           # 'healthy', 'degraded', 'critical'
    'exchange_status': dict,         # Status for each exchange
    'connectivity_metrics': dict,    # Latency and uptime metrics
    'api_rate_limits': dict,         # Current rate limit status
    'recommended_actions': list,     # Actions to improve health
    'last_checked': str              # Timestamp of last health check
}
```

**Exchange Status Structure:**
```python
{
    'hyperliquid': {
        'status': str,                # 'online', 'degraded', 'offline'
        'latency_ms': float,          # Average response time
        'uptime_percent': float,      # Uptime percentage (24h)
        'rate_limit_usage': float,    # Current rate limit usage %
        'last_successful_request': str # Timestamp of last successful call
    }
}
```

### `rebalance_portfolio(target_allocation, max_slippage=0.01)`

Rebalance portfolio across exchanges to match target allocation.

**Parameters:**
- `target_allocation` (dict): Target asset allocation percentages
- `max_slippage` (float): Maximum allowed slippage

**Returns:**
```python
{
    'rebalance_id': str,             # Unique rebalance operation ID
    'status': str,                   # 'completed', 'partial', 'failed'
    'orders_executed': list,         # List of executed orders
    'final_allocation': dict,        # Final allocation after rebalance
    'total_cost': float,             # Total transaction costs
    'execution_time': float,         # Time taken for rebalance
    'slippage_analysis': dict        # Slippage analysis
}
```

### `get_market_data_aggregated(symbol, exchanges=None)`

Get aggregated market data from multiple exchanges.

**Parameters:**
- `symbol` (str): Trading symbol
- `exchanges` (list, optional): Specific exchanges to query

**Returns:**
```python
{
    'symbol': str,                   # Trading symbol
    'aggregated_price': float,       # Volume-weighted average price
    'price_range': dict,             # Min/max prices across exchanges
    'total_volume': float,           # Total volume across exchanges
    'liquidity_score': float,        # Aggregate liquidity score (0-1)
    'exchange_data': dict,           # Raw data from each exchange
    'timestamp': str,                # Data timestamp
    'data_freshness': dict           # Freshness of data from each exchange
}
```

## Advanced Features

### `setup_exchange_failover(primary_exchange, backup_exchanges)`

Configure automatic failover between exchanges.

**Parameters:**
- `primary_exchange` (str): Primary exchange for trading
- `backup_exchanges` (list): Backup exchanges in priority order

**Returns:**
```python
{
    'failover_config_id': str,       # Configuration identifier
    'primary_exchange': str,         # Configured primary exchange
    'backup_exchanges': list,        # Configured backup exchanges
    'failover_conditions': dict,     # Conditions that trigger failover
    'test_status': str,              # 'tested', 'untested', 'failed'
    'activation_status': str         # 'active', 'inactive'
}
```

### `optimize_execution_parameters(symbol, order_size, urgency='normal')`

Optimize order execution parameters for best results.

**Parameters:**
- `symbol` (str): Trading symbol
- `order_size` (float): Order size in base currency
- `urgency` (str): Execution urgency level

**Returns:**
```python
{
    'recommended_exchange': str,     # Best exchange for execution
    'order_splitting': dict,         # Recommended order splitting
    'timing_strategy': str,          # Optimal timing approach
    'expected_slippage': float,      # Expected slippage percentage
    'expected_cost': float,          # Expected total cost
    'confidence_score': float        # Confidence in recommendations (0-1)
}
```

### `track_portfolio_performance(benchmark_symbols=None)`

Track comprehensive portfolio performance metrics.

**Parameters:**
- `benchmark_symbols` (list, optional): Benchmark symbols for comparison

**Returns:**
```python
{
    'performance_metrics': dict,     # Key performance indicators
    'benchmark_comparison': dict,    # Comparison to benchmarks
    'risk_metrics': dict,            # Risk-adjusted performance
    'attribution_analysis': dict,    # Performance attribution by exchange/asset
    'period_returns': dict,          # Returns over different time periods
    'volatility_analysis': dict      # Volatility and drawdown analysis
}
```

## Usage Examples

### Multi-Exchange Order Routing

```python
from neuroflux.exchanges.exchange_manager import ExchangeManager

# Initialize exchange manager
manager = ExchangeManager()
exchanges_config = {
    'hyperliquid': {'api_key': 'your_key', 'secret_key': 'your_secret'},
    'binance': {'api_key': 'your_key', 'secret_key': 'your_secret'}
}
manager.initialize_exchanges(exchanges_config)

# Route order to optimal exchange
order_request = {
    'symbol': 'BTC/USD',
    'side': 'buy',
    'quantity': 0.1,
    'type': 'market',
    'requirements': {
        'max_slippage': 0.001,
        'max_fee': 0.001,
        'min_liquidity': 1000000
    }
}

result = manager.route_order(order_request, routing_strategy='optimal')
print(f"Order routed to {result['exchange']} - ID: {result['order_id']}")
```

### Cross-Exchange Arbitrage

```python
# Scan for arbitrage opportunities
opportunities = manager.scan_arbitrage_opportunities()

if opportunities:
    best_opp = opportunities[0]
    print(f"Arbitrage opportunity: {best_opp['symbol']}")
    print(f"Expected profit: ${best_opp['expected_profit']:.2f}")

    # Execute arbitrage
    result = manager.execute_cross_exchange_arbitrage(best_opp)
    print(f"Arbitrage executed - Profit: ${result['profit_realized']:.2f}")
```

### Portfolio Rebalancing

```python
# Define target allocation
target_allocation = {
    'BTC': 0.4,    # 40% BTC
    'ETH': 0.3,    # 30% ETH
    'SOL': 0.2,    # 20% SOL
    'USDC': 0.1    # 10% USDC
}

# Rebalance portfolio
rebalance_result = manager.rebalance_portfolio(target_allocation, max_slippage=0.005)

if rebalance_result['status'] == 'completed':
    print("Portfolio rebalanced successfully")
    print(f"Final allocation: {rebalance_result['final_allocation']}")
else:
    print(f"Rebalance {rebalance_result['status']}")
```

### Health Monitoring

```python
# Monitor exchange health
health = manager.monitor_exchange_health()

print(f"Overall health: {health['overall_health']}")

for exchange, status in health['exchange_status'].items():
    print(f"{exchange}: {status['status']} - Latency: {status['latency_ms']:.0f}ms")
```

## Configuration

Exchange Manager uses configuration from `config.py`:

- `EXCHANGE_TIMEOUTS`: Timeout settings for different operations
- `RATE_LIMIT_BUFFER`: Safety buffer for rate limits
- `DEFAULT_ROUTING_STRATEGY`: Default order routing strategy
- `ARBITRAGE_THRESHOLDS`: Minimum profit thresholds for arbitrage
- `REBALANCE_FREQUENCY`: How often to check for rebalancing needs

## Error Handling

- **Connection Failures**: Automatic retry with exponential backoff
- **Rate Limiting**: Intelligent rate limit management and queuing
- **Exchange Outages**: Automatic failover to backup exchanges
- **Order Failures**: Partial fill handling and position reconciliation
- **Data Inconsistencies**: Cross-exchange data validation and correction

## Performance Considerations

- **Latency Optimization**: Direct API connections with minimal overhead
- **Concurrent Operations**: Parallel processing across multiple exchanges
- **Caching Strategy**: Intelligent caching of market data and account information
- **Resource Management**: Efficient memory usage for large order books
- **Network Resilience**: Automatic handling of network interruptions

## Cross-References

- See [HyperLiquid Adapter](hyperliquid.md) for exchange-specific implementation
- See [Trading Agent API](../agents/trading.md) for order generation integration
- See [Risk Agent API](../agents/risk.md) for risk management coordination
- See [Base Exchange](base_exchange.md) for unified exchange interface
- See [Task Orchestrator](../task_orchestrator.md) for multi-exchange coordination

## File Locations

- **Exchange Manager**: `src/exchanges/exchange_manager.py`
- **Exchange Adapters**: `src/exchanges/` directory
- **Configuration**: `src/exchanges/config/`
- **Output Directory**: `src/data/exchanges/`
- **Dependencies**: Individual exchange SDKs, WebSocket libraries, AsyncIO</content>
<parameter name="filePath">neuroflux/docs/api/exchanges/manager.md