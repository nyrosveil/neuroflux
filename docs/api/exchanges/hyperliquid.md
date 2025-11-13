# HyperLiquid Adapter API Reference

The HyperLiquid Adapter provides seamless integration with HyperLiquid's perpetual futures exchange, enabling high-performance trading with advanced order types and real-time market data.

## Overview

The HyperLiquid Adapter serves as the bridge between NeuroFlux agents and HyperLiquid's decentralized perpetual futures platform, supporting spot trading, perpetual futures, and advanced order management.

## Core Functions

### `connect_hyperliquid(wallet_address, private_key, testnet=False)`

Establish connection to HyperLiquid exchange.

**Parameters:**
- `wallet_address` (str): Ethereum wallet address for authentication
- `private_key` (str): Private key for transaction signing
- `testnet` (bool): Whether to connect to testnet (default: False)

**Returns:**
```python
{
    'connected': bool,              # Connection status
    'wallet_address': str,          # Connected wallet address
    'account_info': dict,           # Account information
    'leverage_info': dict,          # Available leverage settings
    'vault_address': str,           # Vault contract address
    'connection_id': str           # Unique connection identifier
}
```

### `get_market_data(symbol, data_type='perpetual')`

Retrieve real-time market data from HyperLiquid.

**Parameters:**
- `symbol` (str): Trading symbol (e.g., 'BTC', 'ETH')
- `data_type` (str): Data type ('spot', 'perpetual')

**Returns:**
```python
{
    'symbol': str,                  # Trading symbol
    'price': float,                 # Current price
    'bid': float,                   # Best bid price
    'ask': float,                   # Best ask price
    'volume_24h': float,            # 24h trading volume
    'open_interest': float,         # Open interest (for perpetuals)
    'funding_rate': float,          # Current funding rate
    'mark_price': float,            # Mark price
    'index_price': float,           # Index price
    'timestamp': str               # Data timestamp
}
```

### `place_perpetual_order(order_params)`

Place a perpetual futures order on HyperLiquid.

**Parameters:**
- `order_params` (dict): Order specifications

**Returns:**
```python
{
    'order_id': str,                # Unique order identifier
    'status': str,                  # 'placed', 'filled', 'rejected'
    'symbol': str,                  # Trading symbol
    'side': str,                    # 'buy' or 'sell'
    'quantity': float,              # Order quantity
    'price': float,                 # Order price (market orders: None)
    'leverage': float,              # Leverage used
    'order_type': str,              # 'market', 'limit', 'stop'
    'txn_hash': str,                # Blockchain transaction hash
    'gas_used': int                # Gas used for transaction
}
```

**Order Parameters Structure:**
```python
{
    'symbol': str,                  # Trading symbol
    'side': str,                    # 'buy' or 'sell'
    'quantity': float,              # Order size in base currency
    'order_type': str,              # 'market', 'limit', 'stop_market', 'stop_limit'
    'price': float,                 # Limit price (required for limit orders)
    'leverage': float,              # Leverage (1-50x)
    'reduce_only': bool,            # Reduce position only flag
    'time_in_force': str,           # 'gtc', 'ioc', 'fok'
    'trigger_price': float          # Trigger price for stop orders
}
```

### `get_position_info(symbol=None)`

Get current position information.

**Parameters:**
- `symbol` (str, optional): Specific symbol to query

**Returns:**
```python
{
    'positions': list,              # List of open positions
    'total_unrealized_pnl': float,  # Total unrealized P&L
    'total_margin_used': float,     # Total margin used
    'available_balance': float,     # Available balance for new positions
    'maintenance_margin': float,    # Maintenance margin requirement
    'liquidation_price': dict       # Liquidation prices by symbol
}
```

**Position Structure:**
```python
{
    'symbol': str,                  # Position symbol
    'side': str,                    # 'long' or 'short'
    'size': float,                  # Position size
    'entry_price': float,           # Average entry price
    'mark_price': float,            # Current mark price
    'unrealized_pnl': float,        # Unrealized profit/loss
    'leverage': float,              # Leverage used
    'margin_used': float,           # Margin allocated
    'liquidation_price': float      # Liquidation price
}
```

### `manage_leverage(symbol, leverage)`

Adjust leverage for a specific symbol.

**Parameters:**
- `symbol` (str): Trading symbol
- `leverage` (float): New leverage value (1-50x)

**Returns:**
```python
{
    'success': bool,                # Operation success
    'symbol': str,                  # Affected symbol
    'previous_leverage': float,     # Previous leverage
    'new_leverage': float,          # New leverage setting
    'max_leverage': float,          # Maximum allowed leverage
    'txn_hash': str                # Transaction hash
}
```

### `get_funding_history(symbol, hours=24)`

Retrieve funding rate payment history.

**Parameters:**
- `symbol` (str): Trading symbol
- `hours` (int): Hours of history to retrieve

**Returns:**
```python
{
    'symbol': str,                  # Trading symbol
    'funding_payments': list,       # List of funding payments
    'total_paid': float,            # Total funding paid
    'total_received': float,        # Total funding received
    'net_funding': float,           # Net funding flow
    'current_rate': float           # Current funding rate
}
```

### `cancel_order(order_id)`

Cancel an open order.

**Parameters:**
- `order_id` (str): Order identifier to cancel

**Returns:**
```python
{
    'cancelled': bool,              # Cancellation success
    'order_id': str,                # Cancelled order ID
    'symbol': str,                  # Trading symbol
    'remaining_quantity': float,    # Remaining quantity if partially filled
    'txn_hash': str                # Cancellation transaction hash
}
```

### `get_orderbook(symbol, depth=50)`

Get orderbook depth for a symbol.

**Parameters:**
- `symbol` (str): Trading symbol
- `depth` (int): Orderbook depth to retrieve

**Returns:**
```python
{
    'symbol': str,                  # Trading symbol
    'bids': list,                   # Bid orders [[price, quantity], ...]
    'asks': list,                   # Ask orders [[price, quantity], ...]
    'timestamp': str,               # Orderbook timestamp
    'spread': float,                # Bid-ask spread
    'mid_price': float             # Mid price
}
```

## Advanced Features

### `batch_orders(order_list)`

Execute multiple orders in a single transaction.

**Parameters:**
- `order_list` (list): List of order specifications

**Returns:**
```python
{
    'batch_id': str,                # Batch operation identifier
    'orders_placed': int,           # Number of orders successfully placed
    'orders_failed': int,           # Number of failed orders
    'total_gas_used': int,          # Total gas consumption
    'txn_hash': str,                # Batch transaction hash
    'execution_results': list       # Individual order results
}
```

### `set_stop_loss_take_profit(position_params)`

Set stop loss and take profit orders for a position.

**Parameters:**
- `position_params` (dict): Position and order specifications

**Returns:**
```python
{
    'position_symbol': str,         # Position symbol
    'stop_loss_order': dict,        # Stop loss order details
    'take_profit_order': dict,      # Take profit order details
    'trigger_conditions': dict,     # Trigger conditions
    'risk_management_active': bool  # Risk management status
}
```

### `get_trade_history(symbol=None, limit=100)`

Retrieve historical trade data.

**Parameters:**
- `symbol` (str, optional): Specific symbol to query
- `limit` (int): Maximum number of trades to retrieve

**Returns:**
```python
{
    'trades': list,                 # List of trade records
    'total_trades': int,            # Total number of trades
    'date_range': dict,             # Date range of returned data
    'symbols_traded': list,         # Symbols with trading activity
    'pnl_summary': dict            # P&L summary
}
```

## Usage Examples

### Basic Connection and Trading

```python
from neuroflux.exchanges.hyperliquid_adapter import HyperLiquidAdapter

# Initialize adapter
hl = HyperLiquidAdapter()

# Connect to HyperLiquid
connection = hl.connect_hyperliquid(
    wallet_address="0x1234...abcd",
    private_key="your_private_key",
    testnet=True  # Use testnet for development
)

if connection['connected']:
    print(f"Connected to HyperLiquid with wallet: {connection['wallet_address']}")

    # Get market data
    btc_data = hl.get_market_data('BTC')
    print(f"BTC Price: ${btc_data['price']:.2f}")

    # Place a market order
    order = hl.place_perpetual_order({
        'symbol': 'BTC',
        'side': 'buy',
        'quantity': 0.001,
        'order_type': 'market',
        'leverage': 5.0
    })

    print(f"Order placed: {order['order_id']}")
```

### Position Management

```python
# Get current positions
positions = hl.get_position_info()
print(f"Total unrealized P&L: ${positions['total_unrealized_pnl']:.2f}")

# Adjust leverage
leverage_result = hl.manage_leverage('ETH', 10.0)
if leverage_result['success']:
    print(f"Leverage updated to {leverage_result['new_leverage']}x")

# Set risk management
risk_orders = hl.set_stop_loss_take_profit({
    'symbol': 'BTC',
    'stop_loss_price': 45000,
    'take_profit_price': 55000,
    'quantity': 0.001
})
```

### Advanced Order Types

```python
# Place a limit order with stop loss
limit_order = hl.place_perpetual_order({
    'symbol': 'ETH',
    'side': 'buy',
    'quantity': 0.1,
    'order_type': 'limit',
    'price': 2800,
    'leverage': 3.0,
    'time_in_force': 'gtc'
})

# Place a stop market order
stop_order = hl.place_perpetual_order({
    'symbol': 'ETH',
    'side': 'sell',
    'quantity': 0.1,
    'order_type': 'stop_market',
    'trigger_price': 2700,
    'leverage': 3.0,
    'reduce_only': True
})
```

### Batch Operations

```python
# Execute multiple orders in batch
batch_orders = [
    {
        'symbol': 'BTC',
        'side': 'buy',
        'quantity': 0.001,
        'order_type': 'market',
        'leverage': 5.0
    },
    {
        'symbol': 'ETH',
        'side': 'buy',
        'quantity': 0.1,
        'order_type': 'limit',
        'price': 2800,
        'leverage': 3.0
    }
]

batch_result = hl.batch_orders(batch_orders)
print(f"Batch executed: {batch_result['orders_placed']} orders placed")
```

## Configuration

HyperLiquid Adapter uses configuration from `config.py`:

- `HYPERLIQUID_TESTNET`: Testnet configuration
- `HYPERLIQUID_MAINNET`: Mainnet configuration
- `DEFAULT_LEVERAGE`: Default leverage setting
- `MAX_LEVERAGE`: Maximum allowed leverage
- `GAS_LIMIT_BUFFER`: Gas limit safety buffer
- `ORDER_TIMEOUT`: Order execution timeout

## Error Handling

- **Network Errors**: Automatic retry with exponential backoff
- **Insufficient Funds**: Clear error messages with balance information
- **Invalid Orders**: Detailed validation with specific error reasons
- **Rate Limiting**: Intelligent queuing and rate limit management
- **Blockchain Congestion**: Dynamic gas price adjustment

## Performance Considerations

- **Low Latency**: Direct blockchain interaction with optimized RPC calls
- **Gas Optimization**: Efficient transaction batching and gas estimation
- **Memory Management**: Minimal memory footprint for high-frequency operations
- **Concurrent Operations**: Support for parallel order execution
- **Real-time Updates**: WebSocket integration for live market data

## Security Features

- **Private Key Management**: Secure key storage and transaction signing
- **Transaction Validation**: Pre-execution validation of all parameters
- **Slippage Protection**: Configurable slippage limits and warnings
- **Position Limits**: Automatic position size limits based on account balance
- **Emergency Stop**: Manual kill switch for all trading operations

## Cross-References

- See [Exchange Manager](manager.md) for multi-exchange coordination
- See [Trading Agent API](../agents/trading.md) for order generation integration
- See [Risk Agent API](../agents/risk.md) for position risk management
- See [Base Exchange](base_exchange.md) for unified exchange interface

## File Locations

- **HyperLiquid Adapter**: `src/exchanges/hyperliquid_adapter.py`
- **Helper Functions**: `src/nice_funcs_hl.py`
- **Configuration**: `src/exchanges/config/hyperliquid_config.py`
- **Output Directory**: `src/data/exchanges/hyperliquid/`
- **Dependencies**: web3.py, eth-account, hyperliquid-python-sdk</content>
<parameter name="filePath">neuroflux/docs/api/exchanges/hyperliquid.md