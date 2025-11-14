# Specialized Agent API Reference

The Specialized Agent provides NeuroFlux's advanced market monitoring capabilities with specialized agents for funding rates, liquidation tracking, and whale activity monitoring.

## Overview

Specialized Agents focus on niche market dynamics that can provide critical trading signals. They monitor institutional activity, liquidation cascades, and funding rate arbitrage opportunities.

## Core Agent Types

### Funding Agent (`funding_agent.py`)

Monitors funding rates across perpetual futures markets to identify arbitrage opportunities and directional bias signals.

### Liquidation Agent (`liquidation_agent.py`)

Tracks liquidation events and cascading sell-offs that can create market opportunities or indicate trend changes.

### Whale Agent (`whale_agent.py`)

Monitors large wallet activity and institutional movements to predict market direction and identify accumulation/distribution patterns.

### Chat Agent (`chat_agent.py`)

Conversational AI agent that provides trading guidance, answers market questions, and offers educational content with neuro-enhanced responses.

### CoinGecko Agent (`coingecko_agent.py`)

Fetches comprehensive token metadata and market data from CoinGecko API, enriched with neuro-flux analysis for market insights.

### CopyBot Agent (`copybot_agent.py`)

Monitors successful traders and copies their trades with AI-enhanced validation and timing analysis.

### Strategy Agent (`strategy_agent.py`)

Executes user-defined trading strategies with flux-adaptive optimization and performance monitoring.

### Web Search Agent (`websearch_agent.py`)

Performs intelligent web searches and filters results using AI to find relevant market information, news, and trading insights.

### Backtest Runner Agent (`backtest_runner.py`)

Executes programmatic backtests with neuro-optimization, integrating with RBI agent for automated strategy testing and performance analytics.

## Funding Agent Functions

### `get_funding_rates(exchange=None)`

Retrieve current funding rates from perpetual futures exchanges.

**Parameters:**
- `exchange` (str, optional): Specific exchange to query ('hyperliquid', 'binance', etc.)

**Returns:**
```python
{
    'timestamp': str,           # Data timestamp
    'exchange': str,            # Exchange name
    'rates': dict,              # Funding rates by symbol
    'averages': dict,           # Average rates by timeframe
    'premiums': dict           # Premium/discount indicators
}
```

**Rate Structure:**
```python
{
    'BTC/USD': {
        'rate': float,          # Current funding rate (annualized %)
        'mark_price': float,    # Mark price
        'index_price': float,   # Index price
        'premium': float       # Premium/discount
    }
}
```

### `analyze_funding_arbitrage()`

Identify funding rate arbitrage opportunities across exchanges.

**Returns:**
```python
{
    'opportunities': list,      # Arbitrage opportunities
    'best_long': dict,         # Best exchange for long positions
    'best_short': dict,        # Best exchange for short positions
    'spread_analysis': dict,   # Cross-exchange spread analysis
    'confidence': float       # Analysis confidence (0-1)
}
```

**Opportunity Structure:**
```python
{
    'symbol': str,             # Trading pair
    'long_exchange': str,      # Best for long
    'short_exchange': str,     # Best for short
    'spread': float,          # Funding rate differential
    'annual_return': float,    # Expected annual return
    'risk_score': float       # Risk assessment (0-1)
}
```

### `funding_rate_signals()`

Generate trading signals based on funding rate analysis.

**Returns:**
```python
{
    'directional_signals': list, # Market direction signals
    'arbitrage_signals': list,   # Arbitrage opportunities
    'risk_warnings': list,      # High-risk funding environments
    'market_bias': str         # Overall market bias ('bullish', 'bearish', 'neutral')
}
```

## Liquidation Agent Functions

### `monitor_liquidations(exchange=None, symbol=None)`

Monitor real-time liquidation events across exchanges.

**Parameters:**
- `exchange` (str, optional): Specific exchange to monitor
- `symbol` (str, optional): Specific symbol to monitor

**Returns:**
```python
{
    'recent_liquidations': list, # Recent liquidation events
    'total_volume': float,       # Total liquidated volume (24h)
    'largest_liquidation': dict, # Largest single liquidation
    'liquidation_ratio': float,  # Long/short liquidation ratio
    'cascade_risk': float       # Risk of liquidation cascade (0-1)
}
```

**Liquidation Event Structure:**
```python
{
    'timestamp': str,           # Liquidation timestamp
    'exchange': str,            # Exchange where liquidation occurred
    'symbol': str,              # Trading pair
    'side': str,                # 'long' or 'short'
    'price': float,             # Liquidation price
    'quantity': float,          # Liquidated quantity
    'value': float             # USD value liquidated
}
```

### `analyze_liquidation_cascades()`

Analyze patterns that may lead to liquidation cascades.

**Returns:**
```python
{
    'cascade_probability': float, # Probability of cascade (0-1)
    'trigger_levels': dict,      # Price levels that could trigger cascades
    'vulnerable_positions': list, # Position sizes at risk
    'recommended_actions': list, # Recommended trading actions
    'risk_zones': dict          # High-risk price zones
}
```

### `liquidation_impact_signals()`

Generate signals based on liquidation activity and market impact.

**Returns:**
```python
{
    'market_signals': list,      # Signals for market direction
    'entry_signals': list,       # Potential entry opportunities
    'exit_warnings': list,       # Positions to exit
    'volatility_forecast': float, # Expected volatility increase
    'momentum_shift': str       # Momentum shift direction
}
```

## Whale Agent Functions

### `track_large_transactions(symbol=None, min_value=100000)`

Monitor large wallet transactions and whale activity.

**Parameters:**
- `symbol` (str, optional): Token symbol to monitor
- `min_value` (float): Minimum transaction value in USD (default: $100k)

**Returns:**
```python
{
    'large_transactions': list,  # Recent large transactions
    'whale_movements': list,     # Significant whale activity
    'accumulation_signals': list, # Potential accumulation patterns
    'distribution_signals': list, # Potential distribution patterns
    'whale_alerts': list        # Critical whale activity alerts
}
```

**Transaction Structure:**
```python
{
    'timestamp': str,           # Transaction timestamp
    'from_address': str,        # Sender address
    'to_address': str,          # Receiver address
    'value': float,             # Transaction value in USD
    'token': str,               # Token symbol
    'transaction_type': str,    # 'transfer', 'swap', 'bridge'
    'whale_category': str      # 'institutional', 'large_holder', 'exchange'
}
```

### `analyze_whale_behavior()`

Analyze whale trading patterns and market influence.

**Returns:**
```python
{
    'behavior_patterns': dict,   # Identified behavior patterns
    'market_influence': float,   # Whale influence score (0-1)
    'accumulation_phases': list, # Periods of accumulation
    'distribution_phases': list, # Periods of distribution
    'predictive_signals': list   # Predictive market signals
}
```

### `whale_portfolio_tracking(wallet_addresses)`

Track specific whale wallet portfolios and changes.

**Parameters:**
- `wallet_addresses` (list): List of wallet addresses to track

**Returns:**
```python
{
    'portfolio_changes': dict,   # Changes in tracked portfolios
    'significant_moves': list,   # Significant position changes
    'exposure_analysis': dict,   # Portfolio exposure analysis
    'risk_assessment': dict,     # Risk assessment for each wallet
    'correlation_signals': list  # Correlation with market movements
}
```

## Cross-Agent Integration

### `specialized_market_analysis(symbol)`

Combine funding, liquidation, and whale analysis for comprehensive market view.

**Parameters:**
- `symbol` (str): Token symbol to analyze

**Returns:**
```python
{
    'funding_analysis': dict,    # Funding rate analysis
    'liquidation_analysis': dict, # Liquidation risk analysis
    'whale_analysis': dict,      # Whale activity analysis
    'combined_signals': list,    # Integrated trading signals
    'market_sentiment': str,     # Overall market sentiment
    'risk_level': str,          # Risk level ('low', 'medium', 'high')
    'opportunity_score': float   # Opportunity score (0-1)
}
```

### `real_time_monitoring_dashboard()`

Provide real-time monitoring data for all specialized agents.

**Returns:**
```python
{
    'funding_rates': dict,       # Current funding rates
    'liquidation_feed': list,    # Recent liquidations
    'whale_activity': list,      # Recent whale transactions
    'alerts': list,             # Active alerts
    'market_health': dict,      # Overall market health metrics
    'last_update': str          # Last update timestamp
}
```

## Usage Examples

### Funding Rate Arbitrage

```python
from neuroflux.agents.specialized_agents import funding_agent

# Check for arbitrage opportunities
opportunities = funding_agent.analyze_funding_arbitrage()

if opportunities['opportunities']:
    for opp in opportunities['opportunities']:
        if opp['annual_return'] > 0.05:  # 5% annual return
            print(f"🚀 Arbitrage opportunity: {opp['symbol']}")
            print(f"Long: {opp['long_exchange']}, Short: {opp['short_exchange']}")
            print(f"Expected return: {opp['annual_return']:.1%}")
```

### Liquidation Cascade Monitoring

```python
from neuroflux.agents.specialized_agents import liquidation_agent

# Monitor for cascade risks
cascade_analysis = liquidation_agent.analyze_liquidation_cascades()

if cascade_analysis['cascade_probability'] > 0.7:
    print("⚠️ High liquidation cascade risk!")
    print(f"Trigger levels: {cascade_analysis['trigger_levels']}")

    # Get impact signals
    signals = liquidation_agent.liquidation_impact_signals()
    for signal in signals['entry_signals']:
        print(f"📈 Entry opportunity: {signal}")
```

### Whale Activity Tracking

```python
from neuroflux.agents.specialized_agents import whale_agent

# Track large BTC transactions
btc_activity = whale_agent.track_large_transactions("BTC", min_value=500000)

for tx in btc_activity['large_transactions']:
    if tx['value'] > 1000000:  # $1M+ transactions
        print(f"🐋 Large BTC transaction: ${tx['value']:,.0f}")
        print(f"Type: {tx['transaction_type']}")

# Analyze whale behavior patterns
behavior = whale_agent.analyze_whale_behavior()
print(f"Whale influence score: {behavior['market_influence']:.2f}")
```

### Integrated Specialized Analysis

```python
from neuroflux.agents.specialized_agents import specialized_market_analysis

# Get comprehensive analysis for SOL
sol_analysis = specialized_market_analysis("SOL/USD")

print(f"Market sentiment: {sol_analysis['market_sentiment']}")
print(f"Risk level: {sol_analysis['risk_level']}")
print(f"Opportunity score: {sol_analysis['opportunity_score']:.2f}")

# Check for strong signals
for signal in sol_analysis['combined_signals']:
    if signal['strength'] > 0.8:
        print(f"🔥 Strong signal: {signal['type']} - {signal['description']}")
```

## Configuration

Specialized Agents use configuration from `config.py`:

- `FUNDING_RATE_THRESHOLDS`: Funding rate signal thresholds
- `LIQUIDATION_MONITORING`: Liquidation tracking parameters
- `WHALE_TRACKING`: Whale monitoring settings and wallet lists
- `ALERT_THRESHOLDS`: Alert trigger thresholds for all agents

## Error Handling

- **API Rate Limits**: Implements backoff strategies and fallback data sources
- **Network Issues**: Graceful degradation with cached data
- **Exchange Outages**: Automatic failover to alternative data sources
- **Data Validation**: Comprehensive validation of transaction and market data

## Performance Considerations

- **Real-time Processing**: Optimized for low-latency monitoring
- **Memory Management**: Efficient data structures for large transaction volumes
- **Concurrent Monitoring**: Multiple exchanges and tokens monitored simultaneously
- **Caching Strategy**: Smart caching to reduce API calls while maintaining freshness

## Cross-References

- See [Trading Agent API](trading.md) for signal consumption
- See [Risk Agent API](risk.md) for risk integration
- See [Exchange Manager API](../exchanges/manager.md) for market data
- See [Analysis Agent API](analysis.md) for complementary analysis
- See [Base Agent Framework](../base_agent.md) for agent lifecycle

## File Locations

- **Funding Agent**: `src/agents/funding_agent.py`
- **Liquidation Agent**: `src/agents/liquidation_agent.py`
- **Whale Agent**: `src/agents/whale_agent.py`
- **Output Directory**: `src/data/specialized_agents/`
- **Dependencies**: Exchange APIs, Blockchain APIs, Social APIs</content>
<parameter name="filePath">neuroflux/docs/api/agents/specialized.md