# 🔗 NeuroFlux Exchange Setup Guide

## Overview

This comprehensive guide covers the setup and configuration of exchange integrations in NeuroFlux, including API key management, multi-exchange coordination, and production deployment best practices.

**Target Audience:** Traders and developers integrating NeuroFlux with cryptocurrency exchanges

**Prerequisites:**
- Active accounts on target exchanges
- Understanding of exchange APIs and authentication
- Basic knowledge of cryptocurrency trading

---

## 📋 Table of Contents

1. [Security Best Practices](#security-best-practices)
2. [API Key Configuration](#api-key-configuration)
3. [Exchange-Specific Setup](#exchange-specific-setup)
4. [Multi-Exchange Coordination](#multi-exchange-coordination)
5. [Rate Limiting & Error Handling](#rate-limiting--error-handling)
6. [Testing & Validation](#testing--validation)
7. [Production Deployment](#production-deployment)
8. [Monitoring & Maintenance](#monitoring--maintenance)

---

## 🔐 Security Best Practices

### API Key Management

**Never commit API keys to version control:**
```bash
# ❌ BAD: Committing secrets
git add .env
git commit -m "Add API keys"

# ✅ GOOD: Use .env.example template
cp .env.example .env
# Edit .env with your keys (not tracked by git)
echo ".env" >> .gitignore
```

**Environment Variable Structure:**
```bash
# .env file (never commit)
HYPER_LIQUID_ETH_PRIVATE_KEY=0x_your_private_key_here
SOLANA_PRIVATE_KEY=your_solana_private_key_base58
BIRDEYE_API_KEY=your_birdeye_api_key
COINGECKO_API_KEY=your_coingecko_api_key

# Optional: Additional exchanges
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret
```

### Key Permissions

**Principle of Least Privilege:**
- **Read-only keys** for market data and monitoring
- **Trading keys** with restricted permissions
- **Separate keys** for different environments (dev/staging/prod)

```python
# Example: Restricted permissions configuration
EXCHANGE_PERMISSIONS = {
    'hyperliquid': {
        'trading_enabled': True,
        'withdrawals_enabled': False,  # Never enable withdrawals
        'max_order_size': 1000.0,      # USD limit per order
        'allowed_symbols': ['BTC', 'ETH', 'SOL']  # Restrict to specific pairs
    },
    'solana': {
        'trading_enabled': True,
        'withdrawals_enabled': False,
        'max_wallet_balance': 10000.0  # Maximum wallet balance
    }
}
```

### Key Rotation

```python
class APIKeyManager:
    """Automated API key rotation and management"""

    def __init__(self, key_store_path: str = "keys/"):
        self.key_store = key_store_path
        self.rotation_schedule = {}  # key_id -> rotation_date

    async def rotate_key(self, exchange: str, key_type: str):
        """Rotate API key for security"""
        # Generate new key pair
        new_keys = await self.generate_new_keys(exchange)

        # Test new keys
        if await self.test_keys(exchange, new_keys):
            # Update configuration
            await self.update_configuration(exchange, new_keys)

            # Revoke old keys
            await self.revoke_old_keys(exchange)

            # Update rotation schedule
            self.schedule_next_rotation(exchange)

            return True
        return False

    async def schedule_next_rotation(self, exchange: str):
        """Schedule next key rotation (90 days)"""
        next_rotation = datetime.now() + timedelta(days=90)
        self.rotation_schedule[exchange] = next_rotation

        # Set up automated rotation
        # (Use cron job or scheduler)
```

---

## 🔑 API Key Configuration

### Environment Setup

```bash
# 1. Clone and setup NeuroFlux
git clone https://github.com/nyrosveil/neuroflux.git
cd neuroflux

# 2. Create environment
conda create -n neuroflux python=3.10.9
conda activate neuroflux
pip install -r requirements.txt

# 3. Configure environment variables
cp .env_example .env
# Edit .env with your API keys
```

### Key Validation

```python
import os
from typing import Dict, List

class KeyValidator:
    """Validate API key configuration and permissions"""

    REQUIRED_KEYS = {
        'hyperliquid': ['HYPER_LIQUID_ETH_PRIVATE_KEY'],
        'solana': ['SOLANA_PRIVATE_KEY'],
        'birdeye': ['BIRDEYE_API_KEY'],
        'coingecko': ['COINGECKO_API_KEY']
    }

    def validate_all_keys(self) -> Dict[str, bool]:
        """Validate all configured API keys"""
        results = {}

        for exchange, keys in self.REQUIRED_KEYS.items():
            results[exchange] = self.validate_exchange_keys(exchange, keys)

        return results

    def validate_exchange_keys(self, exchange: str, required_keys: List[str]) -> bool:
        """Validate keys for specific exchange"""
        missing_keys = []
        invalid_keys = []

        for key_name in required_keys:
            key_value = os.getenv(key_name)

            if not key_value:
                missing_keys.append(key_name)
                continue

            # Basic format validation
            if not self.validate_key_format(exchange, key_name, key_value):
                invalid_keys.append(key_name)

        if missing_keys or invalid_keys:
            print(f"❌ {exchange.upper()} Key Validation Failed:")
            if missing_keys:
                print(f"  Missing: {missing_keys}")
            if invalid_keys:
                print(f"  Invalid: {invalid_keys}")
            return False

        print(f"✅ {exchange.upper()} keys validated")
        return True

    def validate_key_format(self, exchange: str, key_name: str, key_value: str) -> bool:
        """Validate key format for specific exchange"""

        if exchange == 'hyperliquid':
            if key_name == 'HYPER_LIQUID_ETH_PRIVATE_KEY':
                # Ethereum private key format (64 hex chars)
                return len(key_value) == 64 and all(c in '0123456789abcdefABCDEF' for c in key_value)

        elif exchange == 'solana':
            if key_name == 'SOLANA_PRIVATE_KEY':
                # Base58 encoded private key (88 chars for 64-byte key)
                return len(key_value) == 88 and self.is_base58(key_value)

        elif exchange in ['birdeye', 'coingecko']:
            # API keys are typically 32-128 characters
            return 32 <= len(key_value) <= 128

        return True  # Unknown exchange/key type

    def is_base58(self, s: str) -> bool:
        """Check if string is valid base58"""
        base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        return all(c in base58_chars for c in s)
```

### Key Testing

```python
class ExchangeTester:
    """Test exchange connectivity and key permissions"""

    def __init__(self, exchange_manager):
        self.manager = exchange_manager

    async def test_all_exchanges(self):
        """Test connectivity to all configured exchanges"""

        test_results = {}

        for exchange_name in self.manager.supported_exchanges:
            try:
                result = await self.test_exchange_connection(exchange_name)
                test_results[exchange_name] = result
                status = "✅" if result['success'] else "❌"
                print(f"{status} {exchange_name}: {result['message']}")

            except Exception as e:
                test_results[exchange_name] = {
                    'success': False,
                    'message': f'Test failed: {str(e)}'
                }
                print(f"❌ {exchange_name}: Test failed - {str(e)}")

        return test_results

    async def test_exchange_connection(self, exchange_name: str):
        """Test individual exchange connection"""

        try:
            # Initialize exchange adapter
            adapter = self.manager.get_adapter(exchange_name)

            # Test basic connectivity
            connection_result = await adapter.connect()

            if not connection_result['connected']:
                return {
                    'success': False,
                    'message': f'Connection failed: {connection_result.get("error", "Unknown error")}'
                }

            # Test basic API calls
            balance_result = await adapter.get_account_balance()
            market_data_result = await adapter.get_market_data('BTC/USD')

            # Test trading permissions (if enabled)
            if self.has_trading_permissions(exchange_name):
                # Place small test order and cancel immediately
                test_order_result = await self.test_trading_permissions(adapter)

            return {
                'success': True,
                'message': 'All tests passed',
                'balance_access': bool(balance_result),
                'market_data_access': bool(market_data_result),
                'trading_enabled': test_order_result.get('success', False)
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Exception during test: {str(e)}'
            }

    def has_trading_permissions(self, exchange_name: str) -> bool:
        """Check if trading is enabled for exchange"""
        # Check configuration
        return self.manager.config.get(f'{exchange_name}_trading_enabled', False)
```

---

## 🔄 Exchange-Specific Setup

### HyperLiquid Setup

**1. Account Creation:**
```bash
# Visit https://hyperliquid.xyz
# Create account and generate Ethereum wallet
# Fund wallet with ETH for gas fees
```

**2. API Configuration:**
```python
from neuroflux.src.exchanges.hyperliquid_adapter import HyperLiquidAdapter

# Initialize HyperLiquid adapter
hl_adapter = HyperLiquidAdapter()

# Configure connection
hl_config = {
    'wallet_address': '0x_your_wallet_address',
    'private_key': os.getenv('HYPER_LIQUID_ETH_PRIVATE_KEY'),
    'testnet': False,  # Use mainnet for production
    'max_leverage': 5.0,
    'gas_limit': 500000
}

# Connect to HyperLiquid
connection_result = await hl_adapter.connect(hl_config)
print(f"HyperLiquid connection: {connection_result}")
```

**3. Trading Configuration:**
```python
# Configure perpetual futures trading
hl_trading_config = {
    'default_leverage': 3.0,
    'max_position_size': 1000.0,  # USD
    'supported_symbols': ['BTC', 'ETH', 'SOL', 'ARB'],
    'order_types': ['market', 'limit', 'stop_loss', 'take_profit'],
    'slippage_tolerance': 0.001,  # 0.1%
    'gas_price_multiplier': 1.1   # 10% above network average
}
```

### Solana Setup

**1. Wallet Setup:**
```bash
# Install Solana CLI
sh -c "$(curl -sSfL https://release.solana.com/v1.18.4/install)"

# Create new wallet or import existing
solana-keygen new --outfile ~/.config/solana/my_wallet.json

# Get wallet address
solana address

# Fund wallet with SOL for fees
# Use faucet for devnet: solana airdrop 1
```

**2. API Configuration:**
```python
from neuroflux.src.exchanges.solana_adapter import SolanaAdapter

# Initialize Solana adapter
solana_adapter = SolanaAdapter()

# Configure connection
solana_config = {
    'private_key': os.getenv('SOLANA_PRIVATE_KEY'),
    'rpc_url': 'https://api.mainnet-beta.solana.com',  # or devnet/testnet
    'commitment': 'confirmed',
    'max_fee_per_tx': 0.000005,  # 0.000005 SOL
    'compute_unit_limit': 200000
}

# Connect to Solana
connection_result = await solana_adapter.connect(solana_config)
print(f"Solana connection: {connection_result}")
```

**3. DEX Configuration:**
```python
# Configure DEX integrations (Raydium, Orca, etc.)
dex_config = {
    'primary_dex': 'raydium',
    'slippage_tolerance': 0.005,  # 0.5%
    'max_impact_tolerance': 0.02,  # 2% price impact
    'supported_tokens': ['SOL', 'USDC', 'BTC', 'ETH'],
    'fee_structure': {
        'raydium': 0.0025,  # 0.25% fee
        'orca': 0.0030     # 0.30% fee
    }
}
```

### Extended Exchange Setup

**1. Account Setup:**
```python
# Extended Exchange supports multiple protocols
extended_config = {
    'protocols': ['uniswap_v3', 'sushiswap', 'pancakeswap'],
    'networks': ['ethereum', 'polygon', 'bsc', 'arbitrum'],
    'private_keys': {
        'ethereum': os.getenv('ETH_PRIVATE_KEY'),
        'polygon': os.getenv('POLYGON_PRIVATE_KEY'),
        'bsc': os.getenv('BSC_PRIVATE_KEY')
    }
}
```

**2. Multi-Network Configuration:**
```python
# Configure cross-chain trading
cross_chain_config = {
    'bridge_protocols': ['hop', 'across', 'celer'],
    'gas_optimization': {
        'max_gas_price_gwei': 50,
        'gas_multiplier': 1.2,
        'timeout_blocks': 100
    },
    'liquidity_thresholds': {
        'min_liquidity_usd': 100000,
        'max_slippage': 0.01
    }
}
```

---

## 🌐 Multi-Exchange Coordination

### Exchange Manager Setup

```python
from neuroflux.src.exchanges.exchange_manager import ExchangeManager

# Initialize exchange manager
exchange_manager = ExchangeManager()

# Configure multiple exchanges
exchange_configs = {
    'hyperliquid': {
        'enabled': True,
        'weight': 0.4,  # 40% of order flow
        'capabilities': ['perpetuals', 'high_leverage'],
        'fee_structure': {'maker': 0.02, 'taker': 0.06}
    },
    'solana': {
        'enabled': True,
        'weight': 0.4,  # 40% of order flow
        'capabilities': ['spot', 'low_fees'],
        'fee_structure': {'maker': 0.0, 'taker': 0.0025}
    },
    'extended': {
        'enabled': True,
        'weight': 0.2,  # 20% of order flow
        'capabilities': ['cross_chain', 'altcoins'],
        'fee_structure': {'maker': 0.003, 'taker': 0.003}
    }
}

# Initialize all exchanges
init_result = await exchange_manager.initialize_exchanges(exchange_configs)
print(f"Exchange initialization: {init_result}")
```

### Smart Order Routing

```python
class SmartOrderRouter:
    """Intelligent order routing across exchanges"""

    def __init__(self, exchange_manager: ExchangeManager):
        self.manager = exchange_manager
        self.routing_strategies = {
            'optimal': self.route_optimal,
            'fastest': self.route_fastest,
            'cheapest': self.route_cheapest,
            'redundant': self.route_redundant
        }

    async def route_order(self, order_request: Dict, strategy: str = 'optimal'):
        """Route order using specified strategy"""

        if strategy not in self.routing_strategies:
            raise ValueError(f"Unknown routing strategy: {strategy}")

        router = self.routing_strategies[strategy]
        return await router(order_request)

    async def route_optimal(self, order_request: Dict):
        """Route to exchange with best combination of price, liquidity, and speed"""

        # Get quotes from all exchanges
        quotes = await self.get_all_quotes(order_request)

        # Score each exchange
        scored_quotes = []
        for quote in quotes:
            score = self.calculate_optimal_score(quote, order_request)
            scored_quotes.append((quote, score))

        # Select best exchange
        best_quote, best_score = max(scored_quotes, key=lambda x: x[1])

        # Execute order
        return await self.execute_on_exchange(best_quote, order_request)

    async def route_redundant(self, order_request: Dict):
        """Place order on multiple exchanges for redundancy"""

        # Split order across exchanges
        split_orders = self.split_order_across_exchanges(order_request)

        # Execute on multiple exchanges simultaneously
        execution_results = await asyncio.gather(*[
            self.execute_on_exchange(split_order['exchange'], split_order)
            for split_order in split_orders
        ], return_exceptions=True)

        return {
            'primary_execution': execution_results[0],
            'backup_executions': execution_results[1:],
            'redundancy_level': len(execution_results)
        }

    def calculate_optimal_score(self, quote: Dict, order_request: Dict) -> float:
        """Calculate optimal routing score"""

        # Factors: price, liquidity, latency, fees
        price_score = 1.0 / (1.0 + abs(quote['price'] - order_request.get('target_price', quote['price'])))

        liquidity_score = min(1.0, quote['liquidity'] / order_request.get('min_liquidity', 10000))

        latency_score = 1.0 / (1.0 + quote['latency_ms'] / 100)  # Prefer < 100ms

        fee_score = 1.0 / (1.0 + quote['fee_percent'] / 0.1)  # Prefer < 0.1%

        # Weighted combination
        return (
            0.4 * price_score +
            0.3 * liquidity_score +
            0.2 * latency_score +
            0.1 * fee_score
        )

    async def get_all_quotes(self, order_request: Dict) -> List[Dict]:
        """Get quotes from all available exchanges"""

        quote_tasks = []
        for exchange_name in self.manager.active_exchanges:
            task = self.get_exchange_quote(exchange_name, order_request)
            quote_tasks.append(task)

        quotes = await asyncio.gather(*quote_tasks, return_exceptions=True)

        # Filter successful quotes
        valid_quotes = []
        for quote in quotes:
            if not isinstance(quote, Exception):
                valid_quotes.append(quote)

        return valid_quotes
```

### Unified Balance Management

```python
class UnifiedBalanceManager:
    """Manage balances across multiple exchanges"""

    def __init__(self, exchange_manager: ExchangeManager):
        self.manager = exchange_manager

    async def get_unified_balance(self, asset: str = None) -> Dict:
        """Get consolidated balance across all exchanges"""

        all_balances = {}

        # Get balances from all exchanges
        balance_tasks = []
        for exchange_name in self.manager.active_exchanges:
            task = self.get_exchange_balance(exchange_name, asset)
            balance_tasks.append(task)

        exchange_balances = await asyncio.gather(*balance_tasks, return_exceptions=True)

        # Aggregate balances
        total_balance = 0.0
        available_balance = 0.0
        locked_balance = 0.0

        for i, balance_result in enumerate(exchange_balances):
            exchange_name = list(self.manager.active_exchanges.keys())[i]

            if isinstance(balance_result, Exception):
                print(f"Failed to get balance from {exchange_name}: {balance_result}")
                continue

            # Convert to USD if needed
            usd_balance = await self.convert_to_usd(balance_result, asset)

            all_balances[exchange_name] = {
                'total': usd_balance['total'],
                'available': usd_balance['available'],
                'locked': usd_balance['locked']
            }

            total_balance += usd_balance['total']
            available_balance += usd_balance['available']
            locked_balance += usd_balance['locked']

        return {
            'total_balance_usd': total_balance,
            'available_balance_usd': available_balance,
            'locked_balance_usd': locked_balance,
            'exchange_breakdown': all_balances,
            'last_updated': datetime.now().isoformat()
        }

    async def rebalance_portfolio(self, target_allocations: Dict[str, float]):
        """Rebalance portfolio across exchanges"""

        current_balance = await self.get_unified_balance()

        # Calculate required transfers
        transfers = self.calculate_rebalancing_transfers(
            current_balance, target_allocations
        )

        # Execute transfers
        transfer_results = []
        for transfer in transfers:
            result = await self.execute_cross_exchange_transfer(transfer)
            transfer_results.append(result)

        return transfer_results
```

---

## ⚡ Rate Limiting & Error Handling

### Rate Limit Management

```python
class RateLimitManager:
    """Manage API rate limits across exchanges"""

    def __init__(self):
        self.limits = {
            'hyperliquid': {
                'requests_per_second': 10,
                'requests_per_minute': 300,
                'burst_limit': 50
            },
            'solana': {
                'requests_per_second': 100,
                'transactions_per_second': 5,
                'burst_limit': 200
            },
            'coingecko': {
                'requests_per_minute': 30,
                'burst_limit': 10
            }
        }

        self.usage_counters = {}  # Track current usage
        self.reset_times = {}     # Track rate limit reset times

    async def check_rate_limit(self, exchange: str, operation: str) -> bool:
        """Check if operation is within rate limits"""

        if exchange not in self.limits:
            return True  # No limits defined

        limit_config = self.limits[exchange]
        current_usage = self.usage_counters.get(exchange, {})

        # Check relevant limits
        if operation == 'request':
            if current_usage.get('per_second', 0) >= limit_config['requests_per_second']:
                return False
            if current_usage.get('per_minute', 0) >= limit_config['requests_per_minute']:
                return False

        elif operation == 'transaction':
            if current_usage.get('tx_per_second', 0) >= limit_config.get('transactions_per_second', 10):
                return False

        return True

    async def record_usage(self, exchange: str, operation: str):
        """Record API usage for rate limiting"""

        if exchange not in self.usage_counters:
            self.usage_counters[exchange] = {}

        counter = self.usage_counters[exchange]

        # Update counters
        if operation == 'request':
            counter['per_second'] = counter.get('per_second', 0) + 1
            counter['per_minute'] = counter.get('per_minute', 0) + 1

        elif operation == 'transaction':
            counter['tx_per_second'] = counter.get('tx_per_second', 0) + 1

        # Schedule counter resets
        await self.schedule_counter_resets(exchange)

    async def schedule_counter_resets(self, exchange: str):
        """Schedule automatic counter resets"""

        # Reset per-second counters every second
        if exchange not in self.reset_times:
            self.reset_times[exchange] = {}

        reset_times = self.reset_times[exchange]

        if 'per_second' not in reset_times:
            reset_times['per_second'] = asyncio.create_task(
                self.reset_counter_after_delay(exchange, 'per_second', 1)
            )

        if 'per_minute' not in reset_times:
            reset_times['per_minute'] = asyncio.create_task(
                self.reset_counter_after_delay(exchange, 'per_minute', 60)
            )

    async def reset_counter_after_delay(self, exchange: str, counter_type: str, delay: int):
        """Reset counter after specified delay"""

        await asyncio.sleep(delay)

        if exchange in self.usage_counters:
            self.usage_counters[exchange][counter_type] = 0

        # Schedule next reset
        self.reset_times[exchange][counter_type] = asyncio.create_task(
            self.reset_counter_after_delay(exchange, counter_type, delay)
        )
```

### Error Handling & Recovery

```python
class ExchangeErrorHandler:
    """Handle exchange errors and implement recovery strategies"""

    def __init__(self, exchange_manager: ExchangeManager):
        self.manager = exchange_manager
        self.error_counts = {}  # Track errors per exchange
        self.circuit_breakers = {}  # Circuit breaker state

    async def handle_exchange_error(self, exchange: str, error: Exception, operation: str):
        """Handle and potentially recover from exchange errors"""

        error_type = self.classify_error(error)

        # Update error tracking
        self.update_error_counts(exchange, error_type)

        # Check circuit breaker
        if self.should_trip_circuit_breaker(exchange):
            await self.trip_circuit_breaker(exchange)
            return {'action': 'circuit_breaker_tripped'}

        # Attempt recovery based on error type
        if error_type == 'rate_limit':
            await self.handle_rate_limit_error(exchange, error)
            return {'action': 'rate_limit_handled', 'retry_after': 60}

        elif error_type == 'network':
            await self.handle_network_error(exchange, error)
            return {'action': 'network_retry', 'retry_after': 5}

        elif error_type == 'authentication':
            await self.handle_auth_error(exchange, error)
            return {'action': 'auth_failed', 'requires_attention': True}

        elif error_type == 'insufficient_funds':
            await self.handle_insufficient_funds(exchange, operation)
            return {'action': 'funds_insufficient', 'requires_funding': True}

        else:
            # Generic error handling
            await self.handle_generic_error(exchange, error)
            return {'action': 'generic_retry', 'retry_after': 30}

    def classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate handling"""

        error_msg = str(error).lower()

        if 'rate limit' in error_msg or 'too many requests' in error_msg:
            return 'rate_limit'
        elif 'network' in error_msg or 'connection' in error_msg or 'timeout' in error_msg:
            return 'network'
        elif 'unauthorized' in error_msg or 'invalid key' in error_msg:
            return 'authentication'
        elif 'insufficient' in error_msg or 'balance' in error_msg:
            return 'insufficient_funds'
        else:
            return 'generic'

    async def handle_rate_limit_error(self, exchange: str, error: Exception):
        """Handle rate limit exceeded errors"""

        # Extract retry-after time from error if available
        retry_after = self.extract_retry_after(error)

        # Update rate limit tracking
        await self.rate_limiter.update_limits_from_error(exchange, error)

        # Exponential backoff for subsequent requests
        await self.implement_exponential_backoff(exchange, retry_after or 60)

    async def handle_network_error(self, exchange: str, error: Exception):
        """Handle network connectivity errors"""

        # Test connectivity
        is_reachable = await self.test_exchange_connectivity(exchange)

        if not is_reachable:
            # Mark exchange as temporarily unavailable
            await self.mark_exchange_unavailable(exchange, duration=300)  # 5 minutes
        else:
            # Transient network issue, retry immediately
            pass

    async def trip_circuit_breaker(self, exchange: str):
        """Trip circuit breaker for failing exchange"""

        self.circuit_breakers[exchange] = {
            'state': 'open',
            'tripped_at': datetime.now(),
            'failure_count': self.error_counts[exchange]['total'],
            'auto_reset_after': 600  # 10 minutes
        }

        # Stop routing orders to this exchange
        await self.manager.disable_exchange(exchange)

        # Schedule circuit breaker reset check
        asyncio.create_task(self.monitor_circuit_breaker_reset(exchange))

    async def monitor_circuit_breaker_reset(self, exchange: str):
        """Monitor and potentially reset circuit breaker"""

        breaker_state = self.circuit_breakers[exchange]

        # Wait for auto-reset period
        await asyncio.sleep(breaker_state['auto_reset_after'])

        # Test exchange health
        is_healthy = await self.test_exchange_health(exchange)

        if is_healthy:
            # Reset circuit breaker
            self.circuit_breakers[exchange]['state'] = 'closed'
            await self.manager.enable_exchange(exchange)
            print(f"✅ Circuit breaker reset for {exchange}")
        else:
            # Extend circuit breaker
            breaker_state['auto_reset_after'] *= 2  # Exponential backoff
            asyncio.create_task(self.monitor_circuit_breaker_reset(exchange))
```

---

## 🧪 Testing & Validation

### Exchange Integration Testing

```python
class ExchangeIntegrationTester:
    """Comprehensive testing suite for exchange integrations"""

    def __init__(self, exchange_manager: ExchangeManager):
        self.manager = exchange_manager

    async def run_full_test_suite(self):
        """Run complete integration test suite"""

        test_results = {
            'connectivity_tests': await self.test_connectivity(),
            'api_tests': await self.test_api_functionality(),
            'trading_tests': await self.test_trading_functionality(),
            'performance_tests': await self.test_performance(),
            'error_handling_tests': await self.test_error_handling()
        }

        # Generate test report
        report = self.generate_test_report(test_results)

        # Save report
        await self.save_test_report(report)

        return report

    async def test_connectivity(self):
        """Test basic connectivity to all exchanges"""

        connectivity_results = {}

        for exchange_name in self.manager.supported_exchanges:
            try:
                adapter = self.manager.get_adapter(exchange_name)
                result = await adapter.test_connectivity()

                connectivity_results[exchange_name] = {
                    'success': result['connected'],
                    'latency_ms': result.get('latency', 0),
                    'error': result.get('error')
                }

            except Exception as e:
                connectivity_results[exchange_name] = {
                    'success': False,
                    'error': str(e)
                }

        return connectivity_results

    async def test_api_functionality(self):
        """Test core API functionality"""

        api_results = {}

        test_operations = [
            'get_account_balance',
            'get_market_data',
            'get_order_book',
            'get_recent_trades'
        ]

        for exchange_name in self.manager.active_exchanges:
            exchange_results = {}

            for operation in test_operations:
                try:
                    result = await self.test_api_operation(exchange_name, operation)
                    exchange_results[operation] = {
                        'success': True,
                        'response_time': result.get('response_time', 0),
                        'data_valid': result.get('data_valid', False)
                    }

                except Exception as e:
                    exchange_results[operation] = {
                        'success': False,
                        'error': str(e)
                    }

            api_results[exchange_name] = exchange_results

        return api_results

    async def test_trading_functionality(self):
        """Test trading functionality (use testnet/paper trading)"""

        trading_results = {}

        # Only test if trading is enabled and we're on testnet
        for exchange_name in self.manager.active_exchanges:
            if not self.is_testnet_trading_safe(exchange_name):
                trading_results[exchange_name] = {'skipped': 'Not safe for trading tests'}
                continue

            try:
                # Test order placement and cancellation
                test_order = {
                    'symbol': 'BTC/USD',
                    'side': 'buy',
                    'type': 'limit',
                    'quantity': 0.001,  # Very small amount
                    'price': 30000.0    # Reasonable test price
                }

                # Place test order
                place_result = await self.manager.place_order(exchange_name, test_order)

                if place_result['success']:
                    order_id = place_result['order_id']

                    # Wait a moment
                    await asyncio.sleep(2)

                    # Cancel order
                    cancel_result = await self.manager.cancel_order(exchange_name, order_id)

                    trading_results[exchange_name] = {
                        'order_placement': True,
                        'order_cancellation': cancel_result['success'],
                        'order_id': order_id
                    }
                else:
                    trading_results[exchange_name] = {
                        'order_placement': False,
                        'error': place_result.get('error')
                    }

            except Exception as e:
                trading_results[exchange_name] = {
                    'success': False,
                    'error': str(e)
                }

        return trading_results

    def is_testnet_trading_safe(self, exchange_name: str) -> bool:
        """Check if trading tests are safe for this exchange"""

        # Only allow on known testnets or with explicit configuration
        testnet_exchanges = ['hyperliquid_testnet', 'solana_devnet']

        return (
            exchange_name in testnet_exchanges or
            self.manager.config.get(f'{exchange_name}_allow_trading_tests', False)
        )
```

### Validation Scripts

```python
# test_exchanges.py
import asyncio
from exchange_integration_tester import ExchangeIntegrationTester

async def main():
    """Run exchange integration tests"""

    # Initialize exchange manager
    from neuroflux.src.exchanges.exchange_manager import ExchangeManager
    manager = ExchangeManager()

    # Initialize exchanges
    await manager.initialize_exchanges()

    # Run tests
    tester = ExchangeIntegrationTester(manager)
    results = await tester.run_full_test_suite()

    # Print summary
    print("🧪 Exchange Integration Test Results")
    print("=" * 50)

    for test_category, category_results in results.items():
        print(f"\n📊 {test_category.replace('_', ' ').title()}:")
        for exchange, result in category_results.items():
            status = "✅" if result.get('success', False) else "❌"
            print(f"  {status} {exchange}: {result.get('message', 'Completed')}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🚀 Production Deployment

### Docker Configuration

```dockerfile
# Dockerfile for NeuroFlux with exchange integrations
FROM python:3.10.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash neuroflux

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set permissions
RUN chown -R neuroflux:neuroflux /app
USER neuroflux

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "
import asyncio
from src.exchanges.exchange_manager import ExchangeManager

async def check():
    manager = ExchangeManager()
    result = await manager.test_connectivity()
    return 'OK' if result['healthy'] else 'FAIL'

asyncio.run(check())
" || exit 1

# Start application
CMD ["python", "src/agents/trading_agent.py"]
```

### Production Environment Setup

```bash
# 1. Build production image
docker build -t neuroflux:prod .

# 2. Create secrets
echo "your_hyperliquid_key" | docker secret create hl_private_key -
echo "your_solana_key" | docker secret create solana_private_key -

# 3. Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# 4. Check logs
docker-compose logs -f trading_service
```

### docker-compose.prod.yml

```yaml
version: '3.8'

services:
  neuroflux-trading:
    image: neuroflux:prod
    container_name: neuroflux_trading
    environment:
      - ENV=production
      - LOG_LEVEL=INFO
    secrets:
      - hl_private_key
      - solana_private_key
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    networks:
      - neuroflux_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import asyncio; from src.exchanges.exchange_manager import ExchangeManager; asyncio.run(ExchangeManager().test_connectivity())"]
      interval: 30s
      timeout: 10s
      retries: 3

  neuroflux-monitor:
    image: neuroflux:prod
    container_name: neuroflux_monitor
    command: python src/monitoring/dashboard.py
    ports:
      - "8080:8080"
    environment:
      - ENV=production
    networks:
      - neuroflux_network
    restart: unless-stopped

secrets:
  hl_private_key:
    external: true
  solana_private_key:
    external: true

networks:
  neuroflux_network:
    driver: bridge
```

### Systemd Service (Alternative)

```ini
# /etc/systemd/system/neuroflux.service
[Unit]
Description=NeuroFlux Trading System
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=neuroflux
Group=neuroflux
WorkingDirectory=/opt/neuroflux
ExecStart=/usr/bin/docker-compose -f docker-compose.prod.yml up
ExecStop=/usr/bin/docker-compose -f docker-compose.prod.yml down
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

---

## 📊 Monitoring & Maintenance

### Exchange Health Monitoring

```python
class ExchangeHealthMonitor:
    """Monitor exchange health and performance"""

    def __init__(self, exchange_manager: ExchangeManager):
        self.manager = exchange_manager
        self.health_metrics = {}
        self.alert_thresholds = {
            'latency_threshold_ms': 1000,
            'error_rate_threshold': 0.05,  # 5%
            'downtime_threshold_minutes': 5
        }

    async def monitor_exchange_health(self):
        """Continuously monitor exchange health"""

        while True:
            for exchange_name in self.manager.active_exchanges:
                health_status = await self.check_exchange_health(exchange_name)

                # Update metrics
                self.update_health_metrics(exchange_name, health_status)

                # Check for alerts
                alerts = self.check_health_alerts(exchange_name, health_status)

                if alerts:
                    await self.send_health_alerts(exchange_name, alerts)

            await asyncio.sleep(60)  # Check every minute

    async def check_exchange_health(self, exchange_name: str):
        """Check comprehensive health of exchange"""

        health_checks = {
            'connectivity': await self.check_connectivity(exchange_name),
            'latency': await self.measure_latency(exchange_name),
            'error_rate': await self.calculate_error_rate(exchange_name),
            'balance_sync': await self.check_balance_sync(exchange_name),
            'order_book_depth': await self.check_order_book_depth(exchange_name)
        }

        # Calculate overall health score
        health_score = self.calculate_health_score(health_checks)

        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,  # 0.0 to 1.0
            'checks': health_checks,
            'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.5 else 'unhealthy'
        }

    async def measure_latency(self, exchange_name: str) -> float:
        """Measure API response latency"""

        start_time = time.time()

        try:
            adapter = self.manager.get_adapter(exchange_name)
            # Quick API call (market data)
            await adapter.get_market_data('BTC/USD')
            latency = (time.time() - start_time) * 1000  # Convert to ms
            return latency

        except Exception:
            return float('inf')

    async def calculate_error_rate(self, exchange_name: str) -> float:
        """Calculate recent error rate"""

        # Get error count from last hour
        recent_errors = await self.get_recent_errors(exchange_name, hours=1)
        recent_requests = await self.get_recent_requests(exchange_name, hours=1)

        if recent_requests == 0:
            return 0.0

        return recent_errors / recent_requests

    def check_health_alerts(self, exchange_name: str, health_status: Dict):
        """Check if health status triggers alerts"""

        alerts = []

        # Latency alert
        if health_status['checks']['latency'] > self.alert_thresholds['latency_threshold_ms']:
            alerts.append({
                'type': 'latency',
                'severity': 'warning',
                'message': f'High latency: {health_status["checks"]["latency"]:.1f}ms'
            })

        # Error rate alert
        if health_status['checks']['error_rate'] > self.alert_thresholds['error_rate_threshold']:
            alerts.append({
                'type': 'error_rate',
                'severity': 'error',
                'message': f'High error rate: {health_status["checks"]["error_rate"]:.1%}'
            })

        # Connectivity alert
        if not health_status['checks']['connectivity']:
            alerts.append({
                'type': 'connectivity',
                'severity': 'critical',
                'message': 'Exchange connectivity lost'
            })

        return alerts

    async def send_health_alerts(self, exchange_name: str, alerts: List[Dict]):
        """Send health alerts through configured channels"""

        for alert in alerts:
            alert_message = f"🚨 {exchange_name.upper()} {alert['type'].upper()}: {alert['message']}"

            # Send to configured channels
            await self.send_email_alert(alert)
            await self.send_slack_alert(alert)
            await self.log_alert_to_file(exchange_name, alert)

            print(alert_message)
```

### Maintenance Procedures

```python
class ExchangeMaintenanceManager:
    """Handle routine exchange maintenance tasks"""

    def __init__(self, exchange_manager: ExchangeManager):
        self.manager = exchange_manager

    async def perform_maintenance_checks(self):
        """Perform routine maintenance checks"""

        maintenance_tasks = [
            self.check_api_key_expiration(),
            self.validate_exchange_connectivity(),
            self.update_exchange_fees(),
            self.check_for_exchange_updates(),
            self.perform_balance_reconciliation()
        ]

        results = await asyncio.gather(*maintenance_tasks, return_exceptions=True)

        # Log maintenance results
        await self.log_maintenance_results(results)

        return results

    async def check_api_key_expiration(self):
        """Check for expiring API keys"""

        expiring_keys = []

        for exchange_name in self.manager.active_exchanges:
            key_info = await self.get_api_key_info(exchange_name)

            if key_info['expires_in_days'] < 30:  # Expires within 30 days
                expiring_keys.append({
                    'exchange': exchange_name,
                    'expires_in': key_info['expires_in_days'],
                    'action_required': 'renew_key'
                })

        if expiring_keys:
            await self.send_key_expiration_alerts(expiring_keys)

        return expiring_keys

    async def perform_balance_reconciliation(self):
        """Reconcile balances across exchanges"""

        reconciliation_results = {}

        for exchange_name in self.manager.active_exchanges:
            try:
                # Get exchange balance
                exchange_balance = await self.manager.get_exchange_balance(exchange_name)

                # Get internal balance tracking
                internal_balance = await self.get_internal_balance_tracking(exchange_name)

                # Compare balances
                discrepancies = self.compare_balances(exchange_balance, internal_balance)

                reconciliation_results[exchange_name] = {
                    'status': 'reconciled' if not discrepancies else 'discrepancies_found',
                    'discrepancies': discrepancies,
                    'exchange_balance': exchange_balance,
                    'internal_balance': internal_balance
                }

            except Exception as e:
                reconciliation_results[exchange_name] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Report significant discrepancies
        significant_discrepancies = [
            (exchange, result) for exchange, result in reconciliation_results.items()
            if result['status'] == 'discrepancies_found' and
            abs(result['discrepancies']['total']) > 10  # $10 threshold
        ]

        if significant_discrepancies:
            await self.report_balance_discrepancies(significant_discrepancies)

        return reconciliation_results
```

---

## 📚 Additional Resources

- **[Base Exchange API](../api/exchanges/base_exchange.md)** - Core exchange interface
- **[HyperLiquid Adapter](../api/exchanges/hyperliquid.md)** - Perpetual futures integration
- **[Exchange Manager](../api/exchanges/manager.md)** - Multi-exchange coordination
- **[Trading Workflow Guide](trading_workflow.md)** - Complete trading setup
- **[Multi-Agent Coordination](multi_agent_coordination.md)** - Advanced agent orchestration

---

## ⚠️ Important Security Notes

- **Key Management**: Rotate API keys every 90 days
- **IP Restrictions**: Whitelist server IPs on exchange accounts
- **Withdrawal Limits**: Never enable automatic withdrawals
- **Test First**: Always test on paper trading/testnet before live trading
- **Backup Keys**: Maintain backup API keys for redundancy
- **Audit Logs**: Log all exchange interactions for compliance

---

*Built with ❤️ by Nyros Veil | [GitHub](https://github.com/nyrosveil/neuroflux) | [Issues](https://github.com/nyrosveil/neuroflux/issues)*