# Analysis Agent API Reference

The Analysis Agent provides NeuroFlux's market analysis capabilities with specialized agents for research, sentiment analysis, and chart pattern recognition.

## Overview

Analysis Agents process market data through multiple specialized lenses to provide comprehensive insights. They work together to generate trading signals and support decision-making across the NeuroFlux system.

## Core Agent Types

### Research Agent (`research_agent.py`)

Conducts fundamental analysis and market research with AI-enhanced data gathering.

### Sentiment Agent (`sentiment_agent.py`)

Analyzes market sentiment from social media, news, and trading community signals.

### Chart Analysis Agent (`chartanalysis_agent.py`)

Performs technical analysis with pattern recognition and neural network enhancements.

## Research Agent Functions

### `gather_fundamentals(token_address)`

Gather fundamental data for comprehensive token analysis.

**Parameters:**
- `token_address` (str): Token contract address or symbol

**Returns:**
```python
{
    'market_cap': float,        # Market capitalization
    'volume_24h': float,        # 24h trading volume
    'circulating_supply': float, # Circulating supply
    'total_supply': float,      # Total supply
    'fdv': float,              # Fully diluted valuation
    'holders': int,            # Number of holders
    'contract_verified': bool,  # Smart contract verification status
    'liquidity_score': float,   # Liquidity assessment (0-1)
    'risk_score': float        # Fundamental risk assessment (0-1)
}
```

**Example:**
```python
# Analyze token fundamentals
fundamentals = gather_fundamentals("0x1234...abcd")
print(f"Market Cap: ${fundamentals['market_cap']:,.0f}")
print(f"Risk Score: {fundamentals['risk_score']:.2f}")
```

### `analyze_tokenomics(token_address)`

Deep tokenomics analysis including supply mechanics and distribution.

**Parameters:**
- `token_address` (str): Token contract address

**Returns:**
```python
{
    'supply_mechanics': str,     # 'fixed', 'inflationary', 'deflationary'
    'burn_mechanism': bool,      # Whether token has burn mechanics
    'vesting_schedule': dict,    # Token vesting information
    'distribution': dict,        # Holder distribution analysis
    'liquidity_analysis': dict,  # DEX liquidity breakdown
    'score': float              # Overall tokenomics score (0-1)
}
```

### `research_report(token_address)`

Generate comprehensive research report combining all analysis.

**Parameters:**
- `token_address` (str): Token to research

**Returns:**
```python
{
    'token': str,               # Token symbol/address
    'timestamp': str,           # Report generation time
    'fundamentals': dict,       # Fundamental data
    'tokenomics': dict,         # Tokenomics analysis
    'recommendation': str,      # 'BUY', 'HOLD', 'SELL', 'AVOID'
    'confidence': float,        # Confidence in recommendation (0-1)
    'key_findings': list,       # List of key findings
    'risks': list              # Identified risks
}
```

## Sentiment Agent Functions

### `analyze_social_sentiment(token_symbol)`

Analyze sentiment from social media platforms and crypto communities.

**Parameters:**
- `token_symbol` (str): Token symbol to analyze

**Returns:**
```python
{
    'overall_sentiment': float,  # Overall sentiment score (-1 to 1)
    'twitter_sentiment': float,  # Twitter sentiment score
    'reddit_sentiment': float,   # Reddit sentiment score
    'telegram_sentiment': float, # Telegram sentiment score
    'news_sentiment': float,     # News article sentiment
    'volume_score': float,       # Social volume indicator
    'momentum': float,          # Sentiment momentum (-1 to 1)
    'confidence': float         # Analysis confidence (0-1)
}
```

**Sentiment Scale:**
- **-1.0**: Extremely negative
- **0.0**: Neutral
- **1.0**: Extremely positive

### `track_sentiment_trends(token_symbol, hours=24)`

Track sentiment changes over time for trend analysis.

**Parameters:**
- `token_symbol` (str): Token symbol
- `hours` (int): Hours to look back (default: 24)

**Returns:**
```python
{
    'current': float,           # Current sentiment
    'previous': float,          # Previous period sentiment
    'change': float,            # Sentiment change
    'trend': str,              # 'improving', 'declining', 'stable'
    'volatility': float,       # Sentiment volatility
    'data_points': list        # Historical sentiment data
}
```

### `sentiment_alerts(token_symbol)`

Monitor for significant sentiment shifts that may indicate market moves.

**Parameters:**
- `token_symbol` (str): Token to monitor

**Returns:**
```python
{
    'alerts': list,             # List of active alerts
    'critical_events': list,    # Major sentiment events
    'threshold_breached': bool, # Whether alert thresholds exceeded
    'recommendation': str      # Trading recommendation based on sentiment
}
```

## Chart Analysis Agent Functions

### `analyze_chart_patterns(token_symbol, timeframe='1h')`

Identify technical chart patterns using neural pattern recognition.

**Parameters:**
- `token_symbol` (str): Token symbol
- `timeframe` (str): Chart timeframe ('1m', '5m', '1h', '1d')

**Returns:**
```python
{
    'patterns': list,           # Detected patterns
    'strength': dict,           # Pattern strength scores
    'confidence': dict,         # Neural confidence scores
    'key_levels': dict,         # Support/resistance levels
    'trend': str,              # 'bullish', 'bearish', 'sideways'
    'momentum': float          # Momentum indicator (-1 to 1)
}
```

**Supported Patterns:**
- Head and Shoulders
- Double Top/Bottom
- Triangle patterns
- Flag patterns
- Cup and Handle

### `calculate_technical_indicators(token_symbol, indicators=None)`

Calculate comprehensive technical indicators.

**Parameters:**
- `token_symbol` (str): Token symbol
- `indicators` (list, optional): Specific indicators to calculate

**Returns:**
```python
{
    'rsi': float,              # Relative Strength Index
    'macd': dict,              # MACD with signal/histogram
    'bollinger': dict,         # Bollinger Bands
    'stoch': dict,             # Stochastic Oscillator
    'williams_r': float,       # Williams %R
    'cci': float,              # Commodity Channel Index
    'adx': float,              # Average Directional Index
    'volume_profile': dict     # Volume analysis
}
```

### `generate_chart_signals(token_symbol)`

Generate trading signals based on technical analysis.

**Parameters:**
- `token_symbol` (str): Token symbol

**Returns:**
```python
{
    'primary_signal': str,     # Main trading signal
    'secondary_signals': list, # Supporting signals
    'strength': float,         # Signal strength (0-1)
    'timeframe': str,          # Recommended timeframe
    'stop_loss': float,        # Suggested stop loss level
    'take_profit': float,      # Suggested take profit level
    'risk_reward': float      # Risk-reward ratio
}
```

## Cross-Agent Integration

### `combined_analysis(token_symbol)`

Combine research, sentiment, and chart analysis for comprehensive insights.

**Parameters:**
- `token_symbol` (str): Token to analyze

**Returns:**
```python
{
    'research_score': float,    # Research analysis score
    'sentiment_score': float,   # Sentiment analysis score
    'technical_score': float,   # Technical analysis score
    'combined_signal': str,     # Overall recommendation
    'confidence': float,        # Combined confidence score
    'key_insights': list,       # Key findings from all analyses
    'risk_assessment': dict     # Overall risk assessment
}
```

### `analysis_pipeline(tokens)`

Run complete analysis pipeline for multiple tokens.

**Parameters:**
- `tokens` (list): List of token symbols to analyze

**Returns:**
```python
{
    'results': dict,            # Analysis results by token
    'summary': dict,            # Portfolio-level summary
    'opportunities': list,      # Identified opportunities
    'risks': list,             # Identified risks
    'timestamp': str           # Analysis timestamp
}
```

## Usage Examples

### Complete Analysis Workflow

```python
from neuroflux.agents.analysis_agents import (
    research_agent,
    sentiment_agent,
    chartanalysis_agent
)

# Analyze a token comprehensively
token = "SOL/USD"

# Research analysis
research = research_agent.research_report(token)
print(f"Research: {research['recommendation']} ({research['confidence']:.2f})")

# Sentiment analysis
sentiment = sentiment_agent.analyze_social_sentiment(token)
print(f"Sentiment: {sentiment['overall_sentiment']:.2f}")

# Technical analysis
technical = chartanalysis_agent.generate_chart_signals(token)
print(f"Technical: {technical['primary_signal']} ({technical['strength']:.2f})")

# Combined analysis
combined = combined_analysis(token)
print(f"Combined: {combined['combined_signal']} ({combined['confidence']:.2f})")
```

### Real-time Monitoring

```python
from neuroflux.agents.analysis_agents import analysis_pipeline

# Monitor multiple tokens
tokens = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]

# Run analysis pipeline
results = analysis_pipeline(tokens)

# Check for opportunities
for token, analysis in results['results'].items():
    if analysis['combined_signal'] == 'BUY' and analysis['confidence'] > 0.8:
        print(f"🚀 Strong buy signal for {token}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        print(f"Key insights: {', '.join(analysis['key_insights'][:3])}")
```

## Configuration

Analysis Agents use configuration from `config.py`:

- `SENTIMENT_SOURCES`: Social media sources to monitor
- `TECHNICAL_TIMEFRAMES`: Chart timeframes for analysis
- `RESEARCH_DEPTH`: Level of fundamental analysis detail
- `ALERT_THRESHOLDS`: Sentiment and technical alert thresholds

## Error Handling

- **API Rate Limits**: Implements backoff and fallback strategies
- **Data Unavailability**: Graceful degradation with cached data
- **Network Issues**: Retry logic with exponential backoff
- **Invalid Tokens**: Validation and error reporting

## Performance Considerations

- **Concurrent Processing**: Multiple tokens analyzed simultaneously
- **Caching**: Results cached to reduce API calls
- **Optimized APIs**: Efficient data structures and algorithms
- **Resource Management**: Memory-efficient processing for large datasets

## Cross-References

- See [Trading Agent API](trading.md) for signal consumption
- See [Risk Agent API](risk.md) for risk integration
- See [Exchange Manager API](../exchanges/manager.md) for market data
- See [Model Factory API](../model_factory.md) for AI analysis
- See [Base Agent Framework](../base_agent.md) for agent lifecycle

## File Locations

- **Research Agent**: `src/agents/research_agent.py`
- **Sentiment Agent**: `src/agents/sentiment_agent.py`
- **Chart Analysis Agent**: `src/agents/chartanalysis_agent.py`
- **Output Directory**: `src/data/analysis_agents/`
- **Dependencies**: Exchange APIs, Social APIs, Technical Libraries</content>
<parameter name="filePath">neuroflux/docs/api/agents/analysis.md