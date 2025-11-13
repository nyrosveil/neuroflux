# Base Exchange API Reference

The Base Exchange provides the unified interface that all NeuroFlux exchange adapters must implement, ensuring consistent behavior across different trading platforms.

## Overview

The Base Exchange defines the abstract interface for all exchange integrations in NeuroFlux. It provides a standardized set of methods for market data retrieval, order management, account operations, and exchange-specific functionality.

## Abstract Interface Methods

### `connect(self, credentials: dict) -> dict`

Establish connection to the exchange.

**Parameters:**
- `credentials` (dict): Authentication credentials for the exchange

**Returns:**
```python
{
    'connected': bool,              # Connection status
    'exchange_id': str,             # Unique exchange identifier
    'account_info': dict,           # Basic account information
    'connection_time': str,         # Connection timestamp
    'status': str                   # 'connected', 'failed', 'restricted'
}
```

### `disconnect(self) -> bool`

Close connection to the exchange.

**Returns:**
- `bool`: True if disconnection successful, False otherwise

### `get_account_balance(self, asset: str = None) -> dict`

Retrieve account balance information.

**Parameters:**
- `asset` (str, optional): Specific asset to query, None for all assets

**Returns:**
```python
{
    'balances': dict,               # Asset balances
    'total_value_usd': float,       # Total account value in USD
    'available_balances': dict,     # Available for trading
    'locked_balances': dict,        # Locked in orders/positions
    'last_updated': str            # Balance update timestamp
}
```

### `get_market_data(self, symbol: str, data_type: str = 'ticker') -> dict`

Retrieve market data for a trading symbol.

**Parameters:**
- `symbol` (str): Trading symbol (e.g., 'BTC/USD', 'ETH/USDT')
- `data_type` (str): Type of data ('ticker', 'orderbook', 'trades', 'klines')

**Returns:**
```python
{
    'symbol': str,                  # Trading symbol
    'data_type': str,               # Type of returned data
    'data': dict,                   # Market data payload
    'timestamp': str,               # Data timestamp
    'exchange': str                 # Exchange identifier
}
```

### `place_order(self, order_params: dict) -> dict`

Place a new order on the exchange.

**Parameters:**
- `order_params` (dict): Order specifications

**Returns:**
```python
{
    'order_id': str,                # Unique order identifier
    'status': str,                  # Order status
    'symbol': str,                  # Trading symbol
    'side': str,                   # 'buy' or 'sell'
    'quantity': float,             # Order quantity
    'price': float,                # Order price (None for market orders)
    'order_type': str,             # Order type
    'timestamp': str               # Order placement timestamp
}
```

### `cancel_order(self, order_id: str) -> dict`

Cancel an existing order.

**Parameters:**
- `order_id` (str): Order identifier to cancel

**Returns:**
```python
{
    'cancelled': bool,              # Cancellation success
    'order_id': str,                # Cancelled order ID
    'symbol': str,                  # Trading symbol
    'remaining_quantity': float,    # Remaining quantity if partially filled
    'status': str                   # Final order status
}
```

### `get_order_status(self, order_id: str) -> dict`

Get the current status of an order.

**Parameters:**
- `order_id` (str): Order identifier to query

**Returns:**
```python
{
    'order_id': str,                # Order identifier
    'status': str,                  # Current order status
    'symbol': str,                  # Trading symbol
    'side': str,                   # Order side
    'quantity': float,             # Original quantity
    'filled_quantity': float,      # Filled quantity
    'remaining_quantity': float,   # Remaining quantity
    'price': float,                # Order price
    'average_fill_price': float,   # Average fill price
    'last_update': str             # Last status update timestamp
}
```

### `get_open_orders(self, symbol: str = None) -> list`

Get all open orders for the account.

**Parameters:**
- `symbol` (str, optional): Filter by specific symbol

**Returns:**
- `list`: List of open order dictionaries (same format as get_order_status)

### `get_trade_history(self, symbol: str = None, limit: int = 100) -> list`

Retrieve historical trade data.

**Parameters:**
- `symbol` (str, optional): Filter by specific symbol
- `limit` (int): Maximum number of trades to retrieve

**Returns:**
- `list`: List of trade dictionaries

### `get_positions(self, symbol: str = None) -> list`

Get current positions (for derivatives/margin trading).

**Parameters:**
- `symbol` (str, optional): Filter by specific symbol

**Returns:**
- `list`: List of position dictionaries

## Standard Data Formats

### Order Parameters Structure

```python
{
    'symbol': str,                  # Trading symbol (required)
    'side': str,                   # 'buy' or 'sell' (required)
    'quantity': float,             # Order quantity (required)
    'order_type': str,             # 'market', 'limit', 'stop', etc. (required)
    'price': float,                # Limit price (required for limit orders)
    'time_in_force': str,          # 'GTC', 'IOC', 'FOK' (optional)
    'client_order_id': str,        # Custom order ID (optional)
    'leverage': float,             # Leverage for derivatives (optional)
    'reduce_only': bool,           # Reduce position only flag (optional)
    'trigger_price': float         # Trigger price for stop orders (optional)
}
```

### Market Data Structures

#### Ticker Data
```python
{
    'symbol': str,                  # Trading symbol
    'price': float,                 # Last price
    'bid': float,                   # Best bid price
    'ask': float,                   # Best ask price
    'volume': float,                # 24h volume
    'high': float,                  # 24h high
    'low': float,                   # 24h low
    'change': float,                # 24h price change
    'change_percent': float         # 24h price change percentage
}
```

#### Orderbook Data
```python
{
    'symbol': str,                  # Trading symbol
    'bids': list,                   # List of [price, quantity] pairs
    'asks': list,                   # List of [price, quantity] pairs
    'timestamp': str,               # Orderbook timestamp
    'update_id': int               # Exchange-specific update identifier
}
```

#### Trade Data
```python
{
    'trade_id': str,                # Unique trade identifier
    'symbol': str,                  # Trading symbol
    'price': float,                 # Trade price
    'quantity': float,              # Trade quantity
    'side': str,                   # 'buy' or 'sell'
    'timestamp': str,               # Trade timestamp
    'is_maker': bool               # Whether trade was maker or taker
}
```

## Error Handling

### Standard Error Format

```python
{
    'error': bool,                  # True if error occurred
    'error_code': str,              # Error code identifier
    'error_message': str,           # Human-readable error message
    'exchange_error_code': str,     # Original exchange error code
    'retryable': bool,              # Whether operation can be retried
    'retry_after': int,             # Seconds to wait before retry
    'request_id': str              # Request identifier for debugging
}
```

### Common Error Codes

- `'INVALID_CREDENTIALS'`: Authentication failed
- `'INSUFFICIENT_BALANCE'`: Not enough funds for operation
- `'INVALID_SYMBOL'`: Trading symbol not found
- `'INVALID_ORDER'`: Order parameters invalid
- `'RATE_LIMIT_EXCEEDED'`: API rate limit exceeded
- `'EXCHANGE_MAINTENANCE'`: Exchange under maintenance
- `'NETWORK_ERROR'`: Network connectivity issues

## Rate Limiting

### Rate Limit Headers

```python
{
    'rate_limit_remaining': int,    # Remaining requests in current window
    'rate_limit_reset': int,        # Timestamp when limit resets
    'rate_limit_total': int,        # Total requests allowed per window
    'retry_after': int             # Seconds to wait if limit exceeded
}
```

### Rate Limit Handling

- **Automatic Backoff**: Exponential backoff for rate-limited requests
- **Request Queuing**: Queue requests when rate limits are approached
- **Priority Queuing**: Critical operations get priority in queues
- **Limit Monitoring**: Real-time monitoring of rate limit usage

## WebSocket Support

### WebSocket Connection

```python
{
    'websocket_url': str,           # WebSocket endpoint URL
    'supported_streams': list,      # Available data streams
    'connection_status': str,       # 'connected', 'disconnected', 'error'
    'last_heartbeat': str,          # Last heartbeat timestamp
    'reconnect_attempts': int       # Number of reconnection attempts
}
```

### Stream Subscriptions

```python
{
    'stream_type': str,             # 'ticker', 'orderbook', 'trades', 'user'
    'symbol': str,                  # Trading symbol (if applicable)
    'subscription_id': str,         # Unique subscription identifier
    'active': bool,                 # Whether stream is active
    'last_update': str             # Last data update timestamp
}
```

## Implementation Requirements

### Required Methods

All exchange adapters must implement these core methods:
- `connect()`
- `disconnect()`
- `get_account_balance()`
- `get_market_data()`
- `place_order()`
- `cancel_order()`
- `get_order_status()`

### Optional Methods

These methods may be implemented based on exchange capabilities:
- `get_open_orders()`
- `get_trade_history()`
- `get_positions()`
- WebSocket streaming methods

### Validation Requirements

- **Input Validation**: All inputs must be validated before API calls
- **Error Handling**: Comprehensive error handling with meaningful messages
- **Data Consistency**: Ensure data consistency across different methods
- **Security**: Secure handling of API keys and sensitive data

## Testing Requirements

### Unit Tests

- Mock API responses for deterministic testing
- Test all error conditions and edge cases
- Validate data format compliance
- Test rate limiting behavior

### Integration Tests

- Test with exchange sandbox/testnet environments
- Validate real API interactions
- Test WebSocket streaming functionality
- Performance and latency testing

## Configuration

Exchange adapters use configuration from `config.py`:

- `EXCHANGE_TIMEOUT`: Default timeout for API calls
- `RATE_LIMIT_BUFFER`: Safety buffer for rate limits
- `RETRY_ATTEMPTS`: Maximum retry attempts for failed requests
- `WEBSOCKET_TIMEOUT`: WebSocket connection timeout
- `VALIDATION_STRICTNESS`: Input validation strictness level

## Usage Examples

### Basic Exchange Operations

```python
from neuroflux.exchanges.base_exchange import BaseExchange

# Note: This is an abstract class - use concrete implementations
# exchange = HyperLiquidAdapter()  # or other concrete adapter

# Connect to exchange
credentials = {
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key'
}
connection = exchange.connect(credentials)

# Get account balance
balance = exchange.get_account_balance()
print(f"Total balance: ${balance['total_value_usd']:.2f}")

# Get market data
ticker = exchange.get_market_data('BTC/USD', 'ticker')
print(f"BTC Price: ${ticker['data']['price']:.2f}")

# Place an order
order = exchange.place_order({
    'symbol': 'BTC/USD',
    'side': 'buy',
    'quantity': 0.001,
    'order_type': 'market'
})
print(f"Order placed: {order['order_id']}")
```

### Error Handling

```python
try:
    order = exchange.place_order(order_params)
    if 'error' in order and order['error']:
        print(f"Order failed: {order['error_message']}")
        if order['retryable']:
            # Implement retry logic
            pass
except Exception as e:
    print(f"Unexpected error: {e}")
    # Implement fallback logic
```

## Cross-References

- See [Exchange Manager](manager.md) for multi-exchange coordination
- See [HyperLiquid Adapter](hyperliquid.md) for concrete implementation example
- See [Trading Agent API](../agents/trading.md) for order generation integration
- See [Risk Agent API](../agents/risk.md) for position risk management

## File Locations

- **Base Exchange**: `src/exchanges/base_exchange.py`
- **Exchange Adapters**: `src/exchanges/` directory
- **Configuration**: `src/exchanges/config/`
- **Tests**: `src/tests/exchanges/`
- **Dependencies**: Exchange-specific SDKs, WebSocket libraries, AsyncIO</content>
<parameter name="filePath">neuroflux/docs/api/exchanges/base_exchange.md