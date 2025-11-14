# 🛠️ NeuroFlux Custom Agent Development Guide

## Overview

This comprehensive guide covers the development of custom agents for the NeuroFlux multi-agent trading system. Learn how to extend the BaseAgent framework, implement specialized functionality, and integrate with the broader NeuroFlux ecosystem.

**Target Audience:** Python developers building custom trading agents

**Prerequisites:**
- Python 3.10.9+ proficiency
- Understanding of asynchronous programming
- Basic knowledge of NeuroFlux architecture
- Experience with API integrations

---

## 📋 Table of Contents

1. [Agent Architecture Overview](#agent-architecture-overview)
2. [Extending BaseAgent](#extending-baseagent)
3. [Lifecycle Management](#lifecycle-management)
4. [Neuro-Flux Integration](#neuro-flux-integration)
5. [Communication Patterns](#communication-patterns)
6. [Performance Optimization](#performance-optimization)
7. [Testing Strategies](#testing-strategies)
8. [Deployment & Monitoring](#deployment--monitoring)
9. [Best Practices](#best-practices)
10. [Advanced Patterns](#advanced-patterns)

---

## 🏗️ Agent Architecture Overview

### BaseAgent Framework

NeuroFlux agents follow a hierarchical architecture with the `BaseAgent` class providing the foundation:

```
BaseAgent (Abstract)
├── Core Components
│   ├── Lifecycle Management
│   ├── Neuro-Flux Integration
│   ├── Performance Monitoring
│   ├── Communication Interface
│   └── Configuration Management
├── Specialized Agents
│   ├── TradingAgent
│   ├── RiskAgent
│   ├── AnalysisAgent
│   └── Custom Agents (Your implementations)
```

### Key Design Principles

- **Asynchronous First**: All operations are async by default
- **Flux-Aware**: Agents adapt behavior based on market volatility
- **Modular Design**: Clear separation of concerns
- **Observable**: Comprehensive logging and metrics
- **Resilient**: Built-in error handling and recovery

---

## 🔧 Extending BaseAgent

### Basic Agent Structure

```python
from typing import Dict, Any, Optional, List
from neuroflux.src.agents.base_agent import BaseAgent
from neuroflux.src.orchestration.communication_bus import Message, MessageType

class CustomTradingAgent(BaseAgent):
    """
    Custom trading agent extending BaseAgent framework.

    This agent implements specialized trading logic while leveraging
    the robust foundation provided by BaseAgent.
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Set default configuration
        if config is None:
            config = {
                'trading_pairs': ['BTC/USD', 'ETH/USD'],
                'max_position_size': 1000.0,
                'risk_per_trade': 0.02,  # 2% risk per trade
                'min_confidence': 0.7,
                'update_interval': 30.0
            }

        # Initialize parent class
        super().__init__(agent_id, config, **kwargs)

        # Agent-specific attributes
        self.trading_pairs = config.get('trading_pairs', [])
        self.active_positions = {}
        self.pending_orders = {}

    # Required abstract method implementations
    def _initialize_agent(self) -> bool:
        """Initialize agent-specific components."""
        # Implementation goes here
        pass

    def _execute_agent_cycle(self):
        """Execute main agent logic."""
        # Implementation goes here
        pass
```

### Configuration Management

```python
class ConfigurableAgent(BaseAgent):
    """Agent with advanced configuration management."""

    REQUIRED_CONFIG_KEYS = ['api_key', 'secret_key', 'base_url']
    OPTIONAL_CONFIG_KEYS = {
        'timeout': 30.0,
        'retry_attempts': 3,
        'rate_limit': 10
    }

    def __init__(self, agent_id: str, config: Dict[str, Any], **kwargs):
        # Validate required configuration
        self._validate_config(config)

        # Merge with defaults
        full_config = self._merge_config_defaults(config)

        super().__init__(agent_id, full_config, **kwargs)

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration parameters."""
        missing_keys = []
        for key in self.REQUIRED_CONFIG_KEYS:
            if key not in config or config[key] is None:
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        # Validate value types and ranges
        if 'timeout' in config and config['timeout'] <= 0:
            raise ValueError("Timeout must be positive")

        if 'retry_attempts' in config and config['retry_attempts'] < 0:
            raise ValueError("Retry attempts must be non-negative")

    def _merge_config_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge provided config with defaults."""
        merged = dict(self.OPTIONAL_CONFIG_KEYS)
        merged.update(config)
        return merged
```

---

## 🔄 Lifecycle Management

### Initialization Phase

```python
def _initialize_agent(self) -> bool:
    """
    Initialize agent-specific components.

    This method is called during agent startup and should:
    1. Set up connections (database, APIs, etc.)
    2. Initialize state variables
    3. Validate configuration
    4. Prepare for operation

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        self.logger.info(f"🚀 Initializing {self.agent_id}")

        # 1. Initialize connections
        self.api_client = self._setup_api_client()
        self.database = self._setup_database_connection()

        # 2. Load or initialize state
        self.state = self._load_agent_state()

        # 3. Set up subscriptions
        self._setup_event_subscriptions()

        # 4. Initialize performance trackers
        self._setup_performance_tracking()

        self.logger.info(f"✅ {self.agent_id} initialization complete")
        return True

    except Exception as e:
        self.logger.error(f"❌ {self.agent_id} initialization failed: {e}")
        return False

def _setup_api_client(self):
    """Set up external API client."""
    # Example: Initialize exchange API client
    from neuroflux.src.exchanges.base_exchange import BaseExchange

    return BaseExchange(
        api_key=self.config['api_key'],
        secret_key=self.config['secret_key'],
        base_url=self.config['base_url']
    )

def _setup_database_connection(self):
    """Set up database connection for state persistence."""
    # Example: Initialize database connection
    import sqlite3

    self.db_path = f"data/{self.agent_id}.db"
    return sqlite3.connect(self.db_path)
```

### Execution Cycle

```python
async def _execute_agent_cycle(self):
    """
    Execute one complete cycle of agent logic.

    This method is called repeatedly during agent operation and should:
    1. Gather required data
    2. Perform analysis/processing
    3. Take actions based on results
    4. Update internal state
    5. Record metrics
    """
    try:
        # 1. Gather market data
        market_data = await self._gather_market_data()

        # 2. Perform analysis
        analysis_result = await self._analyze_market_conditions(market_data)

        # 3. Determine actions
        actions = await self._determine_actions(analysis_result)

        # 4. Execute actions
        execution_results = await self._execute_actions(actions)

        # 5. Update state and metrics
        await self._update_agent_state(execution_results)
        self._record_cycle_metrics(execution_results)

    except Exception as e:
        self.logger.error(f"❌ Error in agent cycle: {e}")
        self.metrics.record_request(False)

        # Implement error recovery
        await self._handle_cycle_error(e)

async def _gather_market_data(self) -> Dict[str, Any]:
    """Gather market data from various sources."""
    data = {}

    for symbol in self.trading_pairs:
        try:
            # Get data from exchange
            ticker_data = await self.api_client.get_ticker(symbol)
            orderbook_data = await self.api_client.get_orderbook(symbol, depth=20)

            data[symbol] = {
                'ticker': ticker_data,
                'orderbook': orderbook_data,
                'timestamp': time.time()
            }

        except Exception as e:
            self.logger.warning(f"Failed to get data for {symbol}: {e}")
            continue

    return data

async def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze market conditions using AI and technical indicators."""
    analysis = {}

    for symbol, data in market_data.items():
        try:
            # Technical analysis
            technical_signals = self._calculate_technical_indicators(data)

            # AI-powered analysis
            ai_insights = await self._generate_ai_analysis(symbol, data, technical_signals)

            analysis[symbol] = {
                'technical': technical_signals,
                'ai_insights': ai_insights,
                'confidence': ai_insights.get('confidence', 0.0),
                'recommendation': ai_insights.get('recommendation', 'HOLD')
            }

        except Exception as e:
            self.logger.error(f"Analysis failed for {symbol}: {e}")
            analysis[symbol] = {'error': str(e)}

    return analysis
```

### Cleanup Phase

```python
def _cleanup_agent(self):
    """
    Clean up agent-specific resources.

    This method is called during agent shutdown and should:
    1. Close connections
    2. Save final state
    3. Release resources
    4. Log final status
    """
    try:
        self.logger.info(f"🧹 Cleaning up {self.agent_id}")

        # 1. Cancel pending orders
        await self._cancel_all_pending_orders()

        # 2. Close positions if necessary
        await self._close_all_positions()

        # 3. Save final state
        self._save_final_state()

        # 4. Close connections
        if hasattr(self, 'api_client'):
            await self.api_client.close()

        if hasattr(self, 'database'):
            self.database.close()

        self.logger.info(f"✅ {self.agent_id} cleanup complete")

    except Exception as e:
        self.logger.error(f"❌ Error during cleanup: {e}")

def _save_final_state(self):
    """Save agent state for recovery."""
    state = {
        'agent_id': self.agent_id,
        'active_positions': self.active_positions,
        'pending_orders': self.pending_orders,
        'last_update': time.time(),
        'performance_metrics': self.metrics.get_summary()
    }

    # Save to file
    state_file = f"data/{self.agent_id}_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
```

---

## 🌊 Neuro-Flux Integration

### Flux-Aware Decision Making

```python
def _update_flux_level(self):
    """
    Update agent's flux sensitivity based on market conditions and performance.

    Flux levels affect:
    - Risk tolerance
    - Decision confidence thresholds
    - Action frequency
    - Analysis depth
    """
    # Calculate current market flux
    market_flux = self._calculate_market_flux()

    # Adjust agent flux based on performance
    performance_score = self.metrics.get_success_rate()

    # Flux adjustment logic
    if market_flux > 0.8:  # High volatility
        target_flux = min(0.9, market_flux)
    elif market_flux < 0.3:  # Low volatility
        target_flux = max(0.4, market_flux + 0.2)
    else:  # Normal volatility
        target_flux = market_flux

    # Performance-based adjustment
    if performance_score > 0.8:
        target_flux = min(1.0, target_flux + 0.1)  # Increase confidence
    elif performance_score < 0.5:
        target_flux = max(0.2, target_flux - 0.1)  # Increase caution

    # Smooth flux changes to prevent oscillations
    flux_change = target_flux - self.flux_level
    if abs(flux_change) > 0.05:  # Minimum change threshold
        new_flux = self.flux_level + (flux_change * 0.3)  # Gradual adjustment
        self.update_flux_level(new_flux)

        self.logger.info(f"🌊 Flux level updated: {self.flux_level:.2f} → {new_flux:.2f}")

def _calculate_market_flux(self) -> float:
    """Calculate current market flux level."""
    # Implementation depends on your market data
    # Example: Based on price volatility, volume spikes, etc.

    flux_indicators = []

    for symbol in self.trading_pairs:
        try:
            # Price volatility (24h change %)
            price_change = abs(self.market_data[symbol]['price_change_24h'])
            flux_indicators.append(min(1.0, price_change / 10))  # Normalize

            # Volume spike detection
            volume_ratio = self.market_data[symbol]['volume_ratio']
            flux_indicators.append(min(1.0, volume_ratio / 5))

        except KeyError:
            continue

    # Average flux across all indicators
    return sum(flux_indicators) / len(flux_indicators) if flux_indicators else 0.5
```

### Flux-Adaptive Behavior

```python
async def _determine_actions(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Determine actions based on analysis and current flux level."""

    actions = []

    for symbol, analysis in analysis_result.items():
        if 'error' in analysis:
            continue

        recommendation = analysis['recommendation']
        confidence = analysis['confidence']

        # Adjust confidence threshold based on flux
        base_threshold = self.config['min_confidence']
        flux_adjustment = self.flux_level * 0.2  # Higher flux = lower threshold
        adjusted_threshold = base_threshold - flux_adjustment

        if confidence >= adjusted_threshold:
            if recommendation == 'BUY':
                actions.append({
                    'type': 'BUY',
                    'symbol': symbol,
                    'confidence': confidence,
                    'flux_adjusted': True
                })
            elif recommendation == 'SELL':
                actions.append({
                    'type': 'SELL',
                    'symbol': symbol,
                    'confidence': confidence,
                    'flux_adjusted': True
                })

    return actions

async def _execute_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute determined actions with flux-aware sizing."""

    results = []

    for action in actions:
        try:
            # Adjust position size based on flux
            base_size = self.config['max_position_size']
            flux_multiplier = 1.0 - (self.flux_level * 0.5)  # Reduce size in high flux
            adjusted_size = base_size * flux_multiplier

            if action['type'] == 'BUY':
                result = await self._execute_buy_order(
                    action['symbol'],
                    adjusted_size,
                    action['confidence']
                )
            elif action['type'] == 'SELL':
                result = await self._execute_sell_order(
                    action['symbol'],
                    adjusted_size,
                    action['confidence']
                )

            results.append(result)

        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            results.append({
                'action': action,
                'success': False,
                'error': str(e)
            })

    return results
```

---

## 📡 Communication Patterns

### Message Handling

```python
async def handle_message(self, message: Message) -> Optional[Message]:
    """
    Handle incoming messages from other agents.

    This method is called when the agent receives a message and should:
    1. Validate message authenticity
    2. Process message content
    3. Generate appropriate response
    4. Update agent state if needed
    """
    try:
        # Log message receipt
        self.logger.info(f"📨 Received message: {message.message_type.value} from {message.sender_id}")

        # Route based on message type
        if message.message_type == MessageType.REQUEST:
            return await self._handle_request(message)
        elif message.message_type == MessageType.EVENT:
            return await self._handle_event(message)
        elif message.message_type == MessageType.COMMAND:
            return await self._handle_command(message)
        else:
            self.logger.warning(f"Unhandled message type: {message.message_type}")
            return None

    except Exception as e:
        self.logger.error(f"Message handling failed: {e}")
        return self._create_error_response(message, str(e))

async def _handle_request(self, message: Message) -> Message:
    """Handle request messages."""
    topic = message.topic

    if topic == 'market_data':
        # Provide market data
        data = await self._gather_market_data()
        return self._create_response(message, {'market_data': data})

    elif topic == 'analysis':
        # Provide analysis
        symbol = message.payload.get('symbol')
        analysis = await self._analyze_symbol(symbol)
        return self._create_response(message, {'analysis': analysis})

    elif topic == 'status':
        # Provide agent status
        status = self.get_agent_info()
        return self._create_response(message, {'status': status})

    else:
        return self._create_error_response(message, f"Unknown request topic: {topic}")

async def _handle_event(self, message: Message) -> Optional[Message]:
    """Handle event messages (typically no response needed)."""
    topic = message.topic

    if topic == 'market_update':
        # Update internal market data
        await self._update_market_data(message.payload)

    elif topic == 'risk_alert':
        # Handle risk alerts
        await self._handle_risk_alert(message.payload)

    elif topic == 'system_shutdown':
        # Prepare for shutdown
        await self._prepare_shutdown()

    # Events typically don't require responses
    return None

async def _handle_command(self, message: Message) -> Message:
    """Handle command messages."""
    command = message.payload.get('command')

    if command == 'pause_trading':
        self.trading_paused = True
        return self._create_response(message, {'status': 'paused'})

    elif command == 'resume_trading':
        self.trading_paused = False
        return self._create_response(message, {'status': 'resumed'})

    elif command == 'force_close_positions':
        result = await self._close_all_positions()
        return self._create_response(message, {'closed_positions': result})

    else:
        return self._create_error_response(message, f"Unknown command: {command}")

def _create_response(self, original_message: Message, payload: Dict[str, Any]) -> Message:
    """Create a response message."""
    return Message(
        message_id=f"resp_{original_message.message_id}",
        sender_id=self.agent_id,
        recipient_id=original_message.sender_id,
        message_type=MessageType.RESPONSE,
        topic=original_message.topic,
        payload=payload,
        correlation_id=original_message.correlation_id
    )

def _create_error_response(self, original_message: Message, error: str) -> Message:
    """Create an error response message."""
    return Message(
        message_id=f"err_{original_message.message_id}",
        sender_id=self.agent_id,
        recipient_id=original_message.sender_id,
        message_type=MessageType.RESPONSE,
        topic=original_message.topic,
        payload={'error': error, 'success': False},
        correlation_id=original_message.correlation_id
    )
```

### Publishing Events

```python
async def publish_trading_signal(self, symbol: str, signal: str, confidence: float):
    """Publish trading signal to other agents."""

    signal_event = Message(
        message_id=f"signal_{self.agent_id}_{int(time.time())}",
        sender_id=self.agent_id,
        recipient_id=None,  # Broadcast
        message_type=MessageType.EVENT,
        topic='trading_signal',
        payload={
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'timestamp': time.time(),
            'agent_id': self.agent_id,
            'flux_level': self.flux_level
        }
    )

    await self.communication_bus.publish_event(signal_event)

async def publish_performance_update(self):
    """Publish performance metrics to monitoring agents."""

    metrics = self.metrics.get_summary()

    performance_event = Message(
        message_id=f"perf_{self.agent_id}_{int(time.time())}",
        sender_id=self.agent_id,
        recipient_id=None,  # Broadcast
        message_type=MessageType.EVENT,
        topic='agent_performance',
        payload={
            'agent_id': self.agent_id,
            'metrics': metrics,
            'timestamp': time.time()
        }
    )

    await self.communication_bus.publish_event(performance_event)
```

---

## ⚡ Performance Optimization

### Efficient Data Processing

```python
class OptimizedAgent(BaseAgent):
    """Agent with performance optimizations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Caching
        self.data_cache = {}
        self.cache_ttl = 30  # seconds

        # Batch processing
        self.pending_updates = []
        self.batch_size = 10

        # Async processing pools
        self.processing_semaphore = asyncio.Semaphore(5)  # Limit concurrent operations

    async def _execute_agent_cycle(self):
        """Optimized execution with batching and caching."""

        # Batch market data updates
        await self._process_pending_updates()

        # Use cached data where possible
        market_data = await self._get_cached_market_data()

        # Parallel analysis
        analysis_tasks = []
        for symbol in self.trading_pairs:
            task = self._analyze_symbol_parallel(symbol, market_data.get(symbol))
            analysis_tasks.append(task)

        # Execute analysis in parallel with concurrency control
        analysis_results = await asyncio.gather(*[
            self._execute_with_semaphore(task)
            for task in analysis_tasks
        ], return_exceptions=True)

        # Process results
        await self._process_analysis_results(analysis_results)

    async def _get_cached_market_data(self) -> Dict[str, Any]:
        """Get market data with intelligent caching."""

        current_time = time.time()
        cached_data = {}

        for symbol in self.trading_pairs:
            cache_key = f"market_data_{symbol}"
            cached_entry = self.data_cache.get(cache_key)

            # Check if cache is valid
            if cached_entry and (current_time - cached_entry['timestamp']) < self.cache_ttl:
                cached_data[symbol] = cached_entry['data']
            else:
                # Fetch fresh data
                fresh_data = await self._fetch_market_data(symbol)
                self.data_cache[cache_key] = {
                    'data': fresh_data,
                    'timestamp': current_time
                }
                cached_data[symbol] = fresh_data

        return cached_data

    async def _execute_with_semaphore(self, coro):
        """Execute coroutine with concurrency control."""
        async with self.processing_semaphore:
            return await coro

    async def _process_pending_updates(self):
        """Process accumulated updates in batches."""

        if len(self.pending_updates) >= self.batch_size:
            # Process batch
            await self._execute_batch_update(self.pending_updates)
            self.pending_updates.clear()

    async def _execute_batch_update(self, updates: List[Dict[str, Any]]):
        """Execute batch database update."""

        # Use batch insert/update for efficiency
        try:
            async with self.database.transaction():
                for update in updates:
                    await self.database.execute(update['query'], update['params'])

        except Exception as e:
            self.logger.error(f"Batch update failed: {e}")
            # Fallback to individual updates
            for update in updates:
                try:
                    await self.database.execute(update['query'], update['params'])
                except Exception as e2:
                    self.logger.error(f"Individual update failed: {e2}")
```

### Memory Management

```python
class MemoryEfficientAgent(BaseAgent):
    """Agent with optimized memory usage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Memory limits
        self.max_cache_size = 1000
        self.cleanup_interval = 300  # 5 minutes

        # Weak references for large objects
        self.large_objects = weakref.WeakValueDictionary()

        # Periodic cleanup task
        self.cleanup_task = None

    async def _initialize_agent(self) -> bool:
        success = await super()._initialize_agent()

        if success:
            # Start periodic cleanup
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())

        return success

    async def _periodic_cleanup(self):
        """Periodic cleanup of memory and resources."""

        while not self.should_stop:
            await asyncio.sleep(self.cleanup_interval)

            try:
                # Clean up expired cache entries
                await self._cleanup_cache()

                # Force garbage collection if needed
                await self._check_memory_usage()

                # Clean up old references
                await self._cleanup_references()

            except Exception as e:
                self.logger.error(f"Cleanup failed: {e}")

    async def _cleanup_cache(self):
        """Clean up expired cache entries."""

        current_time = time.time()
        expired_keys = []

        for key, entry in self.data_cache.items():
            if current_time - entry['timestamp'] > self.cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.data_cache[key]

        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _check_memory_usage(self):
        """Check and manage memory usage."""

        import psutil
        process = psutil.Process()

        memory_mb = process.memory_info().rss / 1024 / 1024

        if memory_mb > 500:  # 500MB threshold
            self.logger.warning(f"High memory usage: {memory_mb:.1f}MB")

            # Force garbage collection
            import gc
            collected = gc.collect()

            self.logger.info(f"Garbage collected {collected} objects")

            # If still high, clear caches
            if memory_mb > 600:
                await self._emergency_memory_cleanup()

    async def _emergency_memory_cleanup(self):
        """Emergency memory cleanup."""

        # Clear all caches
        self.data_cache.clear()
        self.large_objects.clear()

        # Cancel non-essential tasks
        await self._cancel_non_essential_tasks()

        self.logger.warning("Emergency memory cleanup performed")
```

---

## 🧪 Testing Strategies

### Unit Testing

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from your_custom_agent import CustomTradingAgent

class TestCustomTradingAgent:
    """Comprehensive test suite for custom trading agent."""

    @pytest.fixture
    async def agent(self):
        """Create test agent instance."""
        config = {
            'trading_pairs': ['BTC/USD'],
            'max_position_size': 100.0,
            'min_confidence': 0.6
        }

        agent = CustomTradingAgent('test_agent', config)

        # Mock dependencies
        agent.api_client = AsyncMock()
        agent.database = AsyncMock()

        # Initialize agent
        success = await agent.initialize()
        assert success

        yield agent

        # Cleanup
        await agent.stop()

    @pytest.mark.asyncio
    async def test_initialization_success(self, agent):
        """Test successful agent initialization."""
        assert agent.agent_id == 'test_agent'
        assert agent.trading_pairs == ['BTC/USD']
        assert agent.flux_level == 0.5  # Default

    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test agent initialization failure."""
        config = {}  # Missing required config

        with pytest.raises(ValueError):
            CustomTradingAgent('test_agent', config)

    @pytest.mark.asyncio
    async def test_market_data_gathering(self, agent):
        """Test market data gathering functionality."""

        # Mock API response
        mock_data = {
            'BTC/USD': {
                'price': 45000.0,
                'volume': 1000000,
                'timestamp': time.time()
            }
        }
        agent.api_client.get_market_data.return_value = mock_data

        # Test data gathering
        data = await agent._gather_market_data()

        assert 'BTC/USD' in data
        assert data['BTC/USD']['price'] == 45000.0
        agent.api_client.get_market_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_analysis_execution(self, agent):
        """Test market analysis execution."""

        market_data = {
            'BTC/USD': {
                'price': 45000.0,
                'volume': 1000000,
                'rsi': 65,
                'macd': 0.002
            }
        }

        # Mock AI response
        mock_response = {
            'success': True,
            'response': '{"recommendation": "BUY", "confidence": 0.8}'
        }
        agent.generate_response = AsyncMock(return_value=mock_response)

        analysis = await agent._analyze_market_conditions(market_data)

        assert 'BTC/USD' in analysis
        assert analysis['BTC/USD']['recommendation'] == 'BUY'
        assert analysis['BTC/USD']['confidence'] == 0.8

    @pytest.mark.asyncio
    async def test_flux_adaptive_behavior(self, agent):
        """Test flux-adaptive behavior."""

        # Set high flux level
        agent.update_flux_level(0.9)

        # Mock analysis with moderate confidence
        analysis = {
            'BTC/USD': {
                'recommendation': 'BUY',
                'confidence': 0.65  # Below normal threshold
            }
        }

        actions = await agent._determine_actions(analysis)

        # Should still trigger action due to high flux lowering threshold
        assert len(actions) > 0
        assert actions[0]['flux_adjusted'] == True

    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling and recovery."""

        # Mock API failure
        agent.api_client.get_market_data.side_effect = Exception("API Error")

        # Should not crash, should log error and continue
        await agent._execute_agent_cycle()

        # Check that error was logged (would need log capture in real test)
        # assert error logged appropriately

    @pytest.mark.asyncio
    async def test_message_handling(self, agent):
        """Test inter-agent message handling."""

        from neuroflux.src.orchestration.communication_bus import Message, MessageType

        # Create test message
        message = Message(
            message_id="test_msg_123",
            sender_id="other_agent",
            recipient_id=agent.agent_id,
            message_type=MessageType.REQUEST,
            topic="status",
            payload={}
        )

        # Handle message
        response = await agent.handle_message(message)

        assert response is not None
        assert response.message_type == MessageType.RESPONSE
        assert response.correlation_id == message.correlation_id
        assert 'status' in response.payload

    @pytest.mark.asyncio
    async def test_performance_metrics(self, agent):
        """Test performance metrics collection."""

        # Execute several cycles
        for _ in range(5):
            await agent._execute_agent_cycle()

        # Check metrics
        metrics = agent.metrics.get_summary()

        assert 'total_requests' in metrics
        assert 'success_rate' in metrics
        assert 'average_response_time' in metrics
        assert metrics['total_requests'] >= 5
```

### Integration Testing

```python
class TestAgentIntegration:
    """Integration tests for agent ecosystem."""

    @pytest.fixture(scope="class")
    async def agent_system(self):
        """Set up multi-agent test system."""

        # Create communication bus
        from neuroflux.src.orchestration.communication_bus import CommunicationBus
        bus = CommunicationBus()

        # Create multiple agents
        trading_agent = CustomTradingAgent('trading_agent', {})
        risk_agent = RiskAgent('risk_agent', {})
        analysis_agent = AnalysisAgent('analysis_agent', {})

        agents = [trading_agent, risk_agent, analysis_agent]

        # Initialize all agents
        for agent in agents:
            success = await agent.initialize()
            assert success

        # Register with bus
        for agent in agents:
            await bus.register_agent(agent.agent_id)

        yield {
            'bus': bus,
            'agents': agents,
            'trading_agent': trading_agent,
            'risk_agent': risk_agent,
            'analysis_agent': analysis_agent
        }

        # Cleanup
        for agent in agents:
            await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_communication(self, agent_system):
        """Test communication between agents."""

        bus = agent_system['bus']
        trading_agent = agent_system['trading_agent']
        analysis_agent = agent_system['analysis_agent']

        # Send analysis request
        request = Message(
            message_id="test_comm_123",
            sender_id=trading_agent.agent_id,
            recipient_id=analysis_agent.agent_id,
            message_type=MessageType.REQUEST,
            topic="analysis",
            payload={'symbol': 'BTC/USD'}
        )

        await bus.send_message(request)

        # Wait for processing
        await asyncio.sleep(1)

        # Check that analysis agent processed request
        # (Would need to check agent state or logs)

    @pytest.mark.asyncio
    async def test_full_trading_workflow(self, agent_system):
        """Test complete trading workflow."""

        trading_agent = agent_system['trading_agent']
        risk_agent = agent_system['risk_agent']

        # Simulate market conditions
        await self._setup_test_market_conditions(trading_agent)

        # Execute trading cycle
        await trading_agent._execute_agent_cycle()

        # Verify risk checks were performed
        # Verify appropriate actions were taken
        # Check that communications occurred

    async def _setup_test_market_conditions(self, agent):
        """Set up test market conditions."""

        # Mock favorable market conditions
        agent.market_data = {
            'BTC/USD': {
                'price': 45000.0,
                'rsi': 35,  # Oversold
                'macd': -0.002,
                'sentiment_score': 0.8
            }
        }
```

---

## 🚀 Deployment & Monitoring

### Production Configuration

```python
# production_agent_config.py
PRODUCTION_AGENT_CONFIG = {
    'agent_id': 'prod_trading_agent_01',
    'flux_level': 0.7,
    'config': {
        'trading_pairs': ['BTC/USD', 'ETH/USD', 'SOL/USD'],
        'max_position_size': 5000.0,
        'risk_per_trade': 0.01,  # Conservative 1% risk
        'min_confidence': 0.8,
        'update_interval': 15.0,  # More frequent updates
        'max_concurrent_trades': 3,
        'emergency_stop_loss': 0.05,  # 5% emergency stop
        'performance_log_interval': 60,  # Log every minute
        'health_check_interval': 30
    },
    'monitoring': {
        'enabled': True,
        'metrics_port': 9090,
        'alert_webhook': 'https://hooks.slack.com/your-webhook',
        'log_level': 'INFO'
    },
    'scaling': {
        'auto_scale': True,
        'min_instances': 1,
        'max_instances': 5,
        'scale_up_threshold': 0.8,  # CPU usage
        'scale_down_threshold': 0.3
    }
}
```

### Health Monitoring

```python
class AgentHealthMonitor:
    """Monitor agent health and performance."""

    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.health_checks = []
        self.last_health_check = 0
        self.health_check_interval = 30

    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""

        health_status = {
            'timestamp': time.time(),
            'agent_id': self.agent.agent_id,
            'status': 'healthy',
            'checks': {},
            'issues': []
        }

        # Check agent responsiveness
        response_time = await self._check_agent_responsiveness()
        health_status['checks']['responsiveness'] = {
            'status': 'pass' if response_time < 5.0 else 'fail',
            'response_time': response_time
        }

        # Check memory usage
        memory_usage = await self._check_memory_usage()
        health_status['checks']['memory'] = {
            'status': 'pass' if memory_usage < 80 else 'warning' if memory_usage < 90 else 'fail',
            'usage_percent': memory_usage
        }

        # Check error rate
        error_rate = self._check_error_rate()
        health_status['checks']['error_rate'] = {
            'status': 'pass' if error_rate < 0.05 else 'warning' if error_rate < 0.1 else 'fail',
            'rate': error_rate
        }

        # Check connectivity
        connectivity = await self._check_connectivity()
        health_status['checks']['connectivity'] = {
            'status': 'pass' if connectivity else 'fail'
        }

        # Determine overall status
        failed_checks = [check for check in health_status['checks'].values()
                        if check['status'] == 'fail']

        if failed_checks:
            health_status['status'] = 'unhealthy'
            health_status['issues'].extend([
                f"{check_name} check failed" for check_name, check in health_status['checks'].items()
                if check['status'] == 'fail'
            ])
        elif any(check['status'] == 'warning' for check in health_status['checks'].values()):
            health_status['status'] = 'degraded'

        self.health_checks.append(health_status)
        return health_status

    async def _check_agent_responsiveness(self) -> float:
        """Check agent response time."""

        start_time = time.time()

        try:
            # Send status request
            status = self.agent.get_agent_info()
            response_time = time.time() - start_time
            return response_time

        except Exception:
            return float('inf')

    async def _check_memory_usage(self) -> float:
        """Check agent memory usage."""

        import psutil
        process = psutil.Process()
        memory_percent = process.memory_percent()
        return memory_percent

    def _check_error_rate(self) -> float:
        """Check recent error rate."""

        recent_metrics = self.agent.metrics.get_recent_metrics(hours=1)

        if not recent_metrics:
            return 0.0

        total_requests = sum(m.get('total_requests', 0) for m in recent_metrics)
        failed_requests = sum(m.get('failed_requests', 0) for m in recent_metrics)

        return failed_requests / total_requests if total_requests > 0 else 0.0

    async def _check_connectivity(self) -> bool:
        """Check external connectivity."""

        # Test API connectivity
        try:
            # Quick API test
            test_result = await self.agent.api_client.test_connectivity()
            return test_result.get('connected', False)

        except Exception:
            return False
```

### Logging and Alerting

```python
class AgentLogger:
    """Advanced logging for agent operations."""

    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.setup_structured_logging()

    def setup_structured_logging(self):
        """Set up structured JSON logging."""

        import logging
        import json

        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': record.created,
                    'level': record.levelname,
                    'agent_id': self.agent.agent_id,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }

                # Add extra fields
                if hasattr(record, 'extra_data'):
                    log_entry.update(record.extra_data)

                return json.dumps(log_entry)

        # Configure logger
        self.logger = logging.getLogger(f"agent.{self.agent.agent_id}")
        self.logger.setLevel(logging.INFO)

        # File handler with structured format
        file_handler = logging.FileHandler(f"logs/{self.agent.agent_id}.log")
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)

        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)

    def log_trading_action(self, action: str, symbol: str, details: Dict[str, Any]):
        """Log trading actions with structured data."""

        self.logger.info(
            f"Trading action: {action} on {symbol}",
            extra={
                'extra_data': {
                    'event_type': 'trading_action',
                    'action': action,
                    'symbol': symbol,
                    'details': details,
                    'flux_level': self.agent.flux_level,
                    'timestamp': time.time()
                }
            }
        )

    def log_performance_metric(self, metric_name: str, value: float, metadata: Dict = None):
        """Log performance metrics."""

        self.logger.info(
            f"Performance metric: {metric_name} = {value}",
            extra={
                'extra_data': {
                    'event_type': 'performance_metric',
                    'metric_name': metric_name,
                    'value': value,
                    'metadata': metadata or {},
                    'timestamp': time.time()
                }
            }
        )

    def log_error(self, error: Exception, context: str = ""):
        """Log errors with full context."""

        self.logger.error(
            f"Error in {context}: {str(error)}",
            extra={
                'extra_data': {
                    'event_type': 'error',
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'context': context,
                    'traceback': traceback.format_exc(),
                    'timestamp': time.time()
                }
            }
        )
```

---

## 📋 Best Practices

### Code Organization

```python
# Recommended agent file structure
custom_agent/
├── __init__.py
├── agent.py              # Main agent class
├── config.py             # Configuration management
├── strategies/           # Trading strategies
│   ├── __init__.py
│   ├── momentum_strategy.py
│   └── mean_reversion_strategy.py
├── indicators/           # Technical indicators
│   ├── __init__.py
│   ├── rsi.py
│   └── macd.py
├── analysis/             # Analysis modules
│   ├── __init__.py
│   ├── market_analyzer.py
│   └── sentiment_analyzer.py
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── data_fetcher.py
│   └── risk_calculator.py
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_strategies.py
│   └── test_integration.py
├── data/                 # Data storage
├── logs/                 # Log files
└── docs/                 # Documentation
    ├── README.md
    └── API.md
```

### Error Handling Patterns

```python
class RobustAgent(BaseAgent):
    """Agent with comprehensive error handling."""

    async def _execute_agent_cycle(self):
        """Execution with multiple fallback strategies."""

        # Primary execution path
        try:
            return await self._primary_execution_path()
        except TemporaryError as e:
            # Retry with backoff
            return await self._retry_with_backoff(e)
        except DataError as e:
            # Fallback to cached data
            return await self._fallback_to_cache(e)
        except CriticalError as e:
            # Emergency procedures
            return await self._emergency_procedure(e)
        except Exception as e:
            # Unexpected error - log and escalate
            await self._handle_unexpected_error(e)
            raise

    async def _retry_with_backoff(self, error: TemporaryError):
        """Retry operation with exponential backoff."""

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                return await self._primary_execution_path()

            except TemporaryError:
                continue

        # All retries failed
        raise PermanentError(f"Operation failed after {max_retries} retries")

    async def _fallback_to_cache(self, error: DataError):
        """Fallback to cached/stale data."""

        self.logger.warning(f"Using cached data due to: {error}")

        # Use cached data with reduced confidence
        cached_data = await self._get_cached_data()
        analysis = await self._analyze_with_reduced_confidence(cached_data)

        # Mark results as using fallback data
        analysis['using_fallback'] = True
        analysis['fallback_reason'] = str(error)

        return analysis

    async def _emergency_procedure(self, error: CriticalError):
        """Execute emergency procedures."""

        self.logger.critical(f"Critical error encountered: {error}")

        # Immediate actions
        await self._cancel_all_orders()
        await self._notify_risk_manager()
        await self._enter_safe_mode()

        # Don't attempt recovery - require manual intervention
        raise error

    async def _handle_unexpected_error(self, error: Exception):
        """Handle unexpected errors."""

        # Log with full context
        self.logger.error(
            f"Unexpected error: {error}",
            extra={
                'traceback': traceback.format_exc(),
                'agent_state': self.get_agent_info(),
                'system_info': await self._get_system_info()
            }
        )

        # Increment error metrics
        self.metrics.record_error(error)

        # Check if error threshold exceeded
        if self._error_threshold_exceeded():
            await self._escalate_to_monitoring()
```

### Performance Patterns

```python
class HighPerformanceAgent(BaseAgent):
    """Agent optimized for high-frequency operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pre-allocate data structures
        self.data_buffer = deque(maxlen=1000)
        self.signal_cache = LRUCache(maxsize=500)

        # Use connection pooling
        self.connection_pool = ConnectionPool(max_size=20)

        # Optimize for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def _execute_agent_cycle(self):
        """High-performance execution loop."""

        # Parallel data fetching
        data_tasks = [
            self._fetch_data_parallel(symbol)
            for symbol in self.trading_pairs
        ]

        market_data = await asyncio.gather(*data_tasks)

        # CPU-intensive analysis in thread pool
        loop = asyncio.get_event_loop()
        analysis_results = await asyncio.gather(*[
            loop.run_in_executor(self.executor, self._analyze_symbol_sync, data)
            for data in market_data
        ])

        # Fast decision making
        decisions = await self._make_fast_decisions(analysis_results)

        # Batch order execution
        await self._execute_orders_batch(decisions)

    def _analyze_symbol_sync(self, data):
        """Synchronous analysis for CPU-bound operations."""

        # Complex calculations here
        # This runs in thread pool to avoid blocking event loop

        result = {
            'symbol': data['symbol'],
            'signals': self._calculate_signals(data),
            'indicators': self._calculate_indicators(data)
        }

        return result

    async def _execute_orders_batch(self, decisions):
        """Execute orders in optimized batches."""

        # Group orders by exchange
        exchange_orders = defaultdict(list)

        for decision in decisions:
            if decision['action'] != 'HOLD':
                exchange_orders[decision['exchange']].append(decision)

        # Execute batches in parallel
        batch_tasks = [
            self._execute_exchange_batch(exchange, orders)
            for exchange, orders in exchange_orders.items()
        ]

        await asyncio.gather(*batch_tasks)

    async def _execute_exchange_batch(self, exchange: str, orders: List[Dict]):
        """Execute batch of orders for single exchange."""

        # Use batch API if available
        if hasattr(self.exchanges[exchange], 'batch_orders'):
            await self.exchanges[exchange].batch_orders(orders)
        else:
            # Execute individually with concurrency control
            semaphore = asyncio.Semaphore(5)  # Max 5 concurrent per exchange

            async def execute_single(order):
                async with semaphore:
                    return await self.exchanges[exchange].place_order(order)

            await asyncio.gather(*[
                execute_single(order) for order in orders
            ])
```

---

## 🔄 Advanced Patterns

### Agent Composition

```python
class CompositeAgent(BaseAgent):
    """Agent composed of multiple specialized sub-agents."""

    def __init__(self, agent_id: str, config: Dict[str, Any], **kwargs):
        super().__init__(agent_id, config, **kwargs)

        # Initialize sub-agents
        self.data_agent = DataCollectionAgent(f"{agent_id}_data", config.get('data_config', {}))
        self.analysis_agent = AnalysisAgent(f"{agent_id}_analysis", config.get('analysis_config', {}))
        self.execution_agent = ExecutionAgent(f"{agent_id}_execution", config.get('execution_config', {}))

        self.sub_agents = [self.data_agent, self.analysis_agent, self.execution_agent]

    async def _initialize_agent(self) -> bool:
        """Initialize all sub-agents."""

        # Initialize sub-agents in dependency order
        for agent in self.sub_agents:
            if not await agent.initialize():
                self.logger.error(f"Failed to initialize sub-agent: {agent.agent_id}")
                return False

        # Set up inter-agent communication
        await self._setup_agent_communication()

        return True

    async def _execute_agent_cycle(self):
        """Orchestrate sub-agent execution."""

        # Phase 1: Data collection
        market_data = await self.data_agent.gather_data()

        # Phase 2: Analysis
        analysis = await self.analysis_agent.analyze_data(market_data)

        # Phase 3: Decision making (this agent)
        decisions = await self._make_decisions(analysis)

        # Phase 4: Execution
        results = await self.execution_agent.execute_decisions(decisions)

        # Aggregate results
        await self._aggregate_results(results)

    async def _setup_agent_communication(self):
        """Set up communication channels between sub-agents."""

        # Create communication channels
        self.channels = {
            'data_to_analysis': asyncio.Queue(),
            'analysis_to_decision': asyncio.Queue(),
            'decision_to_execution': asyncio.Queue()
        }

        # Connect sub-agents to channels
        self.data_agent.output_channel = self.channels['data_to_analysis']
        self.analysis_agent.input_channel = self.channels['data_to_analysis']
        self.analysis_agent.output_channel = self.channels['analysis_to_decision']
        # ... etc
```

### Event-Driven Architecture

```python
class EventDrivenAgent(BaseAgent):
    """Agent based on event-driven architecture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Event system
        self.event_handlers = {}
        self.event_queue = asyncio.Queue()

        # Register event handlers
        self._register_event_handlers()

    def _register_event_handlers(self):
        """Register event handlers."""

        self.event_handlers = {
            'market_data_update': self._handle_market_data_update,
            'trading_signal': self._handle_trading_signal,
            'risk_alert': self._handle_risk_alert,
            'system_status_change': self._handle_system_status_change
        }

    async def _execute_agent_cycle(self):
        """Process events in event-driven manner."""

        # Process all pending events
        while not self.event_queue.empty():
            event = await self.event_queue.get()

            try:
                await self._process_event(event)
            except Exception as e:
                self.logger.error(f"Event processing failed: {e}")
            finally:
                self.event_queue.task_done()

    async def _process_event(self, event: Dict[str, Any]):
        """Process individual event."""

        event_type = event.get('type')

        if event_type in self.event_handlers:
            handler = self.event_handlers[event_type]
            await handler(event)
        else:
            self.logger.warning(f"Unhandled event type: {event_type}")

    async def publish_event(self, event: Dict[str, Any]):
        """Publish event to agent's event queue."""

        await self.event_queue.put(event)

        # Trigger immediate processing if needed
        if event.get('priority') == 'high':
            await self._execute_agent_cycle()

    async def _handle_market_data_update(self, event: Dict[str, Any]):
        """Handle market data updates."""

        symbol = event['symbol']
        data = event['data']

        # Update internal state
        self.market_data[symbol] = data

        # Trigger analysis if conditions met
        if self._should_analyze(symbol):
            await self._trigger_analysis(symbol)

    async def _handle_trading_signal(self, event: Dict[str, Any]):
        """Handle trading signals from other agents."""

        signal = event['signal']
        confidence = event['confidence']

        # Validate signal
        if await self._validate_signal(signal, confidence):
            # Execute trade
            await self._execute_signal(signal)
        else:
            self.logger.info(f"Signal validation failed: {signal}")

    async def _handle_risk_alert(self, event: Dict[str, Any]):
        """Handle risk management alerts."""

        alert_type = event['alert_type']
        severity = event['severity']

        if severity == 'critical':
            # Immediate action required
            await self._handle_critical_risk_alert(alert_type)
        elif severity == 'warning':
            # Monitor closely
            await self._handle_warning_risk_alert(alert_type)
```

### Machine Learning Integration

```python
class MLAugmentedAgent(BaseAgent):
    """Agent augmented with machine learning capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ML components
        self.feature_engineer = FeatureEngineer()
        self.model_manager = ModelManager()
        self.learning_system = OnlineLearningSystem()

        # Training data
        self.training_buffer = deque(maxlen=10000)

    async def _execute_agent_cycle(self):
        """ML-augmented execution."""

        # Gather data
        market_data = await self._gather_market_data()

        # Extract features
        features = await self.feature_engineer.extract_features(market_data)

        # Get ML predictions
        predictions = await self.model_manager.predict(features)

        # Combine with traditional analysis
        traditional_signals = await self._traditional_analysis(market_data)

        # Ensemble decision
        final_decision = await self._ensemble_decision(
            predictions, traditional_signals
        )

        # Execute decision
        result = await self._execute_decision(final_decision)

        # Store for learning
        await self._store_experience(features, final_decision, result)

        # Online learning
        if len(self.training_buffer) >= self.config['batch_size']:
            await self._perform_online_learning()

    async def _ensemble_decision(self, ml_predictions: Dict, traditional_signals: Dict):
        """Combine ML and traditional signals."""

        ensemble_decision = {}

        for symbol in self.trading_pairs:
            ml_signal = ml_predictions.get(symbol, {}).get('signal', 'HOLD')
            ml_confidence = ml_predictions.get(symbol, {}).get('confidence', 0.0)

            trad_signal = traditional_signals.get(symbol, {}).get('signal', 'HOLD')
            trad_confidence = traditional_signals.get(symbol, {}).get('confidence', 0.0)

            # Weighted ensemble
            ml_weight = self.config.get('ml_weight', 0.7)
            trad_weight = 1.0 - ml_weight

            # Signal agreement boost
            agreement_bonus = 0.1 if ml_signal == trad_signal else 0.0

            # Final confidence
            final_confidence = (
                ml_weight * ml_confidence +
                trad_weight * trad_confidence +
                agreement_bonus
            )

            # Choose signal (prefer ML when confident)
            if final_confidence > self.config['decision_threshold']:
                final_signal = ml_signal if ml_confidence > trad_confidence else trad_signal
            else:
                final_signal = 'HOLD'

            ensemble_decision[symbol] = {
                'signal': final_signal,
                'confidence': final_confidence,
                'ml_contribution': ml_weight * ml_confidence,
                'traditional_contribution': trad_weight * trad_confidence
            }

        return ensemble_decision

    async def _store_experience(self, features: Dict, decision: Dict, result: Dict):
        """Store experience for online learning."""

        for symbol, symbol_decision in decision.items():
            experience = {
                'features': features.get(symbol, {}),
                'decision': symbol_decision,
                'result': result.get(symbol, {}),
                'timestamp': time.time(),
                'flux_level': self.flux_level
            }

            self.training_buffer.append(experience)

    async def _perform_online_learning(self):
        """Perform online model updates."""

        # Prepare training data
        training_data = list(self.training_buffer)
        self.training_buffer.clear()

        # Update models
        await self.learning_system.update_models(training_data)

        # Evaluate performance
        performance = await self.learning_system.evaluate_performance(training_data)

        self.logger.info(f"Online learning completed. Performance: {performance}")

        # Adapt learning rate based on performance
        if performance['accuracy'] > 0.8:
            self.learning_system.increase_learning_rate()
        elif performance['accuracy'] < 0.6:
            self.learning_system.decrease_learning_rate()
```

---

*Built with ❤️ by Nyros Veil | [GitHub](https://github.com/nyrosveil/neuroflux) | [Issues](https://github.com/nyrosveil/neuroflux/issues)*