# Trading Agent API Reference

The Trading Agent provides NeuroFlux's core trading functionality with neuro-enhanced decision making and flux-adaptive strategies.

## Overview

The Trading Agent makes buy/sell/hold decisions based on market analysis with neural network enhancements. It supports multi-exchange trading with real-time flux monitoring and integrates with the Risk Agent for safety checks.

## Core Functions

### `get_market_data(token_address=None)`

Get market data for analysis from exchange APIs.

**Parameters:**
- `token_address` (str, optional): Token address or symbol to analyze

**Returns:**
```python
{
    'price': float,           # Current price
    'volume_24h': float,      # 24h trading volume
    'price_change_24h': float, # 24h price change percentage
    'rsi': float,             # Relative Strength Index
    'macd': float,            # MACD indicator value
    'support': float,         # Support level
    'resistance': float,      # Resistance level
    'sentiment_score': float, # Market sentiment (0-1)
    'flux_level': float       # Current market flux level (0-1)
}
```

**Example:**
```python
# Get market data for BTC
btc_data = get_market_data("BTC/USD")
print(f"BTC Price: ${btc_data['price']}")
print(f"RSI: {btc_data['rsi']}")
print(f"Flux Level: {btc_data['flux_level']}")
```

### `analyze_market_neuro(data)`

Neuro-enhanced market analysis using flux-adaptive AI models.

**Parameters:**
- `data` (dict): Market data from `get_market_data()`

**Returns:**
```python
{
    'signal': str,           # 'BUY', 'SELL', or 'HOLD'
    'confidence': float,     # Confidence score (0-1)
    'reasoning': str,        # Explanation for the signal
    'flux_adjustment': float # Flux-based confidence adjustment
}
```

**Signal Logic:**
- **BUY**: RSI < 30 and flux level < sensitivity threshold
- **SELL**: RSI > 70 and flux level < sensitivity threshold
- **HOLD**: Neutral conditions or high flux levels

**Example:**
```python
market_data = get_market_data("ETH/USD")
analysis = analyze_market_neuro(market_data)

print(f"Signal: {analysis['signal']}")
print(f"Confidence: {analysis['confidence']:.2f}")
print(f"Reasoning: {analysis['reasoning']}")

if analysis['signal'] != 'HOLD' and analysis['confidence'] > 0.7:
    print("Strong trading signal detected!")
```

### `execute_trade(signal, token, amount, analysis)`

Execute trade with neuro-flux confirmation and risk validation.

**Parameters:**
- `signal` (str): Trading signal ('BUY', 'SELL', 'HOLD')
- `token` (str): Token symbol to trade
- `amount` (float): Trade amount in USD
- `analysis` (dict): Analysis results from `analyze_market_neuro()`

**Returns:**
```python
{
    'status': str,           # 'executed', 'skipped'
    'signal': str,           # Original signal
    'token': str,            # Token traded
    'amount': float,         # Amount traded
    'confidence': float,     # Analysis confidence
    'timestamp': str,        # Execution timestamp
    'flux_adjusted': bool    # Whether flux adjustment was applied
}
```

**Execution Conditions:**
- Skips if signal is 'HOLD'
- Skips if confidence < 0.6
- Requires risk agent approval

**Example:**
```python
# Execute a trade based on analysis
result = execute_trade(
    signal=analysis['signal'],
    token="BTC/USD",
    amount=100.0,
    analysis=analysis
)

if result['status'] == 'executed':
    print(f"✅ Trade executed: {result['signal']} {result['amount']} USD of {result['token']}")
else:
    print(f"⏭️ Trade skipped: {result['reason']}")
```

### `save_decision(data, analysis, result)`

Save trading decision and execution result to persistent storage.

**Parameters:**
- `data` (dict): Market data used for decision
- `analysis` (dict): Analysis results
- `result` (dict): Execution result

**Saves to:**
- `src/data/trading_agent/latest_decision.json` - Latest decision
- `src/data/trading_agent/trading_history.jsonl` - Historical decisions

**Example:**
```python
# Save complete trading decision
save_decision(market_data, analysis, execution_result)
print("Decision saved to trading history")
```

### `check_risk_status()`

Check risk status from Risk Agent before allowing trades.

**Returns:**
- `bool`: True if trading is allowed, False if blocked by risk limits

**Checks:**
- Reads latest risk report from `src/data/risk_agent/latest_report.json`
- Returns risk check status
- Falls back to True if risk report not found

**Example:**
```python
if check_risk_status():
    print("✅ Risk checks passed - trading allowed")
    # Proceed with trading logic
else:
    print("🚫 Risk checks failed - trading blocked")
    # Skip trading cycle
```

### `main()`

Main trading loop with neuro-flux decision making.

**Features:**
- Continuous monitoring loop
- Risk status checking before each cycle
- Multi-token analysis and trading
- Decision logging and history tracking
- Error handling and recovery

**Configuration:**
- Uses config variables: `SLEEP_BETWEEN_RUNS_MINUTES`, `usd_size`, `max_usd_order_size`
- Monitors tokens from `get_active_tokens()` function

## Integration with Risk Agent

The Trading Agent integrates tightly with the Risk Agent:

```python
# Risk check before trading
if not check_risk_status():
    print("Risk limits exceeded - skipping trading")
    return

# Proceed with analysis and execution
```

## Flux-Adaptive Trading

The agent adapts trading behavior based on market flux levels:

- **Low Flux**: Normal confidence thresholds and position sizing
- **High Flux**: Reduced confidence, smaller position sizes, more conservative signals
- **Critical Flux**: May skip trading entirely

## Usage Examples

### Basic Trading Cycle

```python
from neuroflux.agents.trading_agent import (
    get_market_data,
    analyze_market_neuro,
    execute_trade,
    check_risk_status
)

# Check if trading is allowed
if not check_risk_status():
    print("Risk limits exceeded")
    exit(1)

# Analyze multiple tokens
tokens = ["BTC/USD", "ETH/USD", "SOL/USD"]

for token in tokens:
    # Get market data
    data = get_market_data(token)

    # Neuro analysis
    analysis = analyze_market_neuro(data)

    # Execute if signal is strong
    if analysis['confidence'] > 0.75:
        result = execute_trade(analysis['signal'], token, 50.0, analysis)
        print(f"Executed {result['signal']} for {token}")
    else:
        print(f"Signal too weak for {token}: {analysis['confidence']:.2f}")
```

### Integration with Full NeuroFlux System

```python
from neuroflux.orchestration import TaskOrchestrator
from neuroflux.agents.trading_agent import main as trading_main

# Initialize orchestrator
orchestrator = TaskOrchestrator()

# Register trading agent
await orchestrator.register_agent({
    'agent_type': 'trading_agent',
    'capabilities': ['trading', 'market_analysis'],
    'flux_level': 0.3
})

# Start trading operations
trading_main()
```

## Configuration

The Trading Agent uses configuration from `config.py`:

- `FLUX_SENSITIVITY`: Flux threshold for signal filtering
- `SLEEP_BETWEEN_RUNS_MINUTES`: Cycle interval
- `usd_size`: Base trade size
- `max_usd_order_size`: Maximum trade size

## Error Handling

- **Network Errors**: Retries with exponential backoff
- **Exchange API Errors**: Logs and continues with other tokens
- **Risk Check Failures**: Blocks trading cycle
- **Analysis Errors**: Falls back to HOLD signal

## Performance Considerations

- **Analysis Speed**: Neuro analysis optimized for real-time execution
- **Memory Usage**: Minimal footprint with efficient data structures
- **API Rate Limits**: Respects exchange API limits
- **Concurrent Processing**: Can analyze multiple tokens simultaneously

## Cross-References

- See [Risk Agent API](risk.md) for risk management integration
- See [Exchange Manager API](../exchanges/manager.md) for trading execution
- See [Model Factory API](../model_factory.md) for AI analysis integration
- See [Base Agent Framework](../base_agent.md) for agent lifecycle

## File Location
- **Source**: `src/agents/trading_agent.py`
- **Output Directory**: `src/data/trading_agent/`
- **Dependencies**: Risk Agent, Exchange APIs, Config</content>
<parameter name="filePath">neuroflux/docs/api/agents/trading.md