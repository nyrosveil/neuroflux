# Risk Agent API Reference

The Risk Agent provides NeuroFlux's circuit breaker functionality with neuro-flux awareness, monitoring portfolio risk, P&L, and exposure with adaptive flux sensitivity.

## Overview

The Risk Agent enforces position limits, loss/gain thresholds, and emergency closures. It includes AI confirmation for critical decisions and adapts risk parameters based on market flux levels.

## Core Functions

### `calculate_flux_level()`

Calculate current market flux level for adaptive risk management.

**Returns:**
- `float`: Flux level (0-1), higher means more volatility

**Description:**
- Analyzes recent price movements, volume spikes, and market indicators
- Uses config default `FLUX_SENSITIVITY` in current implementation
- Higher flux levels trigger more conservative risk parameters

**Example:**
```python
flux = calculate_flux_level()
if flux > 0.7:
    print("High market volatility detected")
    # Implement conservative risk measures
```

### `get_portfolio_balance()`

Get current portfolio balance across all exchanges.

**Returns:**
```python
{
    'equity': float,     # Total equity in USD
    'available': float,  # Available for trading
    'positions_value': float  # Current positions value
}
```

**Example:**
```python
balance = get_portfolio_balance()
print(f"Total Equity: ${balance['equity']:.2f}")
print(f"Available: ${balance['available']:.2f}")
```

### `get_positions()`

Get all current positions across exchanges.

**Returns:**
- `list`: List of position dictionaries

**Position Structure:**
```python
[
    {
        'symbol': str,        # Trading symbol
        'side': str,          # 'long' or 'short'
        'size': float,        # Position size
        'entry_price': float, # Entry price
        'current_price': float, # Current price
        'pnl_percentage': float, # P&L percentage
        'pnl_usd': float      # P&L in USD
    }
]
```

### `check_risk_limits(balance, positions, flux_level)`

Check all risk limits with flux-adaptive thresholds.

**Parameters:**
- `balance` (dict): Portfolio balance from `get_portfolio_balance()`
- `positions` (list): Current positions from `get_positions()`
- `flux_level` (float): Current market flux level

**Returns:**
```python
{
    'ok': bool,              # Overall risk status
    'violations': list,      # Critical violations
    'warnings': list,        # Non-critical warnings
    'recommendations': list  # Risk management suggestions
}
```

**Risk Checks Performed:**

1. **Balance Check**: Minimum balance requirement (`MINIMUM_BALANCE_USD`)
2. **Loss Limit Check**: Maximum loss thresholds (USD or percentage-based)
3. **Gain Target Check**: Maximum gain limits
4. **Position Size Check**: Flux-adjusted position limits
5. **Cash Buffer Check**: Minimum cash percentage requirement

**Violation Structure:**
```python
{
    'type': str,        # 'balance', 'loss_limit', 'gain_limit', etc.
    'message': str,     # Human-readable description
    'severity': str     # 'critical' or 'warning'
}
```

**Example:**
```python
balance = get_portfolio_balance()
positions = get_positions()
flux = calculate_flux_level()

risk_check = check_risk_limits(balance, positions, flux)

if not risk_check['ok']:
    print("🚫 Risk violations detected:")
    for violation in risk_check['violations']:
        print(f"  {violation['message']}")

for warning in risk_check['warnings']:
    print(f"⚠️  {warning['message']}")

for rec in risk_check['recommendations']:
    print(f"💡 {rec['message']}")
```

### `save_risk_report(results, balance, positions, flux_level)`

Save risk assessment report to file.

**Parameters:**
- `results` (dict): Risk check results from `check_risk_limits()`
- `balance` (dict): Portfolio balance
- `positions` (list): Current positions
- `flux_level` (float): Market flux level

**Saves to:**
- `src/data/risk_agent/latest_report.json` - Latest risk report
- `src/data/risk_agent/risk_history.jsonl` - Historical risk data

**Report Structure:**
```python
{
    'timestamp': str,        # ISO format timestamp
    'flux_level': float,     # Market flux level
    'balance': dict,         # Portfolio balance
    'positions': list,       # Current positions
    'risk_check': dict,      # Risk assessment results
    'total_positions': int,  # Number of positions
    'total_pnl': float       # Total P&L across positions
}
```

### `emergency_close_positions(reason)`

Emergency close all positions.

**Parameters:**
- `reason` (str): Reason for emergency closure

**Actions:**
- Logs emergency closure event
- Placeholder for exchange API integration
- Should close all positions immediately

**Example:**
```python
# Emergency closure due to critical risk violation
emergency_close_positions("Portfolio loss exceeded maximum threshold")
```

### `main()`

Main risk monitoring loop with neuro-flux awareness.

**Features:**
- Continuous risk monitoring
- Flux level calculation and adaptation
- Portfolio balance and position tracking
- Risk limit checking with configurable intervals
- Emergency response handling
- AI confirmation for critical decisions

**Configuration:**
- Uses config variables: `MINIMUM_BALANCE_USD`, `MAX_LOSS_USD/PERCENT`, etc.
- `SLEEP_BETWEEN_RUNS_MINUTES`: Monitoring interval
- `USE_AI_CONFIRMATION`: Whether to use AI for critical decisions

## Risk Management Features

### Flux-Adaptive Risk Controls

The Risk Agent adapts risk parameters based on market volatility:

```python
# Flux-adjusted position limits
flux_adjusted_limit = max_position_limit * (1 - flux_level * 0.5)

if total_position_value > flux_adjusted_limit:
    # Trigger risk warning or reduction
    pass
```

### Multi-Level Risk Thresholds

- **Critical Violations**: Immediate action required (emergency closure)
- **Warnings**: Monitoring required, potential action needed
- **Recommendations**: Suggestions for risk optimization

### Percentage vs USD-Based Limits

```python
if USE_PERCENTAGE:
    # Percentage-based limits
    loss_threshold = MAX_LOSS_PERCENT / 100 * balance['equity']
else:
    # USD-based limits
    loss_threshold = MAX_LOSS_USD
```

## Integration with Trading Agent

The Risk Agent integrates tightly with the Trading Agent:

```python
# Trading Agent checks risk status before trading
from neuroflux.agents.risk_agent import check_risk_limits, get_portfolio_balance, get_positions, calculate_flux_level

def check_risk_status():
    """Check if trading is allowed based on risk limits."""
    try:
        balance = get_portfolio_balance()
        positions = get_positions()
        flux = calculate_flux_level()

        results = check_risk_limits(balance, positions, flux)
        return results['ok']
    except FileNotFoundError:
        return True  # Fallback if risk data unavailable
```

## Configuration Parameters

The Risk Agent uses configuration from `config.py`:

- **Balance Limits**: `MINIMUM_BALANCE_USD`
- **Loss Limits**: `MAX_LOSS_USD`, `MAX_LOSS_PERCENT`, `USE_PERCENTAGE`
- **Gain Limits**: `MAX_GAIN_USD`, `MAX_GAIN_PERCENT`
- **Position Limits**: `MAX_POSITION_PERCENTAGE`
- **Cash Requirements**: `CASH_PERCENTAGE`
- **Flux Sensitivity**: `FLUX_SENSITIVITY`
- **AI Confirmation**: `USE_AI_CONFIRMATION`
- **Monitoring Interval**: `SLEEP_BETWEEN_RUNS_MINUTES`

## Usage Examples

### Basic Risk Monitoring

```python
from neuroflux.agents.risk_agent import (
    calculate_flux_level,
    get_portfolio_balance,
    get_positions,
    check_risk_limits,
    save_risk_report
)

# Perform risk assessment
flux_level = calculate_flux_level()
balance = get_portfolio_balance()
positions = get_positions()

risk_results = check_risk_limits(balance, positions, flux_level)

# Save assessment
save_risk_report(risk_results, balance, positions, flux_level)

# Check overall status
if risk_results['ok']:
    print("✅ Risk checks passed")
else:
    print("🚫 Risk violations detected")
    for violation in risk_results['violations']:
        if violation['severity'] == 'critical':
            print(f"🚨 CRITICAL: {violation['message']}")
```

### Integration with NeuroFlux Orchestrator

```python
from neuroflux.orchestration import TaskOrchestrator
from neuroflux.agents.risk_agent import main as risk_main

# Initialize orchestrator
orchestrator = TaskOrchestrator()

# Register risk agent
await orchestrator.register_agent({
    'agent_type': 'risk_agent',
    'capabilities': ['risk_management', 'portfolio_monitoring'],
    'flux_level': 0.2  # Conservative flux level
})

# Start risk monitoring
risk_main()
```

### Emergency Risk Response

```python
from neuroflux.agents.risk_agent import check_risk_limits, emergency_close_positions

# Continuous risk monitoring
while True:
    balance = get_portfolio_balance()
    positions = get_positions()
    flux = calculate_flux_level()

    risk_check = check_risk_limits(balance, positions, flux)

    if not risk_check['ok']:
        critical_violations = [
            v for v in risk_check['violations']
            if v['severity'] == 'critical'
        ]

        if critical_violations:
            # Emergency response
            reasons = [v['message'] for v in critical_violations]
            emergency_close_positions("; ".join(reasons))
            break

    time.sleep(60)  # Check every minute
```

## Error Handling

- **Exchange API Errors**: Graceful fallback to cached data
- **Configuration Errors**: Uses safe defaults
- **File I/O Errors**: Continues operation with warnings
- **Network Timeouts**: Implements retry logic

## Performance Considerations

- **Lightweight Monitoring**: Minimal computational overhead
- **Efficient Data Structures**: Fast risk calculations
- **Configurable Intervals**: Adjustable monitoring frequency
- **Memory Management**: Automatic cleanup of old reports

## Cross-References

- See [Trading Agent API](trading.md) for integration details
- See [Exchange Manager API](../exchanges/manager.md) for portfolio data
- See [Base Agent Framework](../base_agent.md) for agent lifecycle
- See [Communication Bus API](../communication_bus.md) for inter-agent coordination

## File Location
- **Source**: `src/agents/risk_agent.py`
- **Output Directory**: `src/data/risk_agent/`
- **Dependencies**: Exchange APIs, Config, Trading Agent</content>
<parameter name="filePath">neuroflux/docs/api/agents/risk.md