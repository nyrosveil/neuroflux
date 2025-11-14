# 🏛️ NeuroFlux Trading Workflow Guide

## Overview

This comprehensive guide walks you through setting up and running NeuroFlux for automated trading. Learn how to configure agents, establish trading workflows, and monitor performance in production environments.

**Target Audience:** Traders and developers implementing NeuroFlux trading systems

**Prerequisites:**
- Python 3.10.9 environment
- Basic understanding of cryptocurrency trading
- API keys for exchanges and AI providers

---

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Agent Configuration](#agent-configuration)
3. [Trading Workflow](#trading-workflow)
4. [Risk Management Integration](#risk-management-integration)
5. [Performance Monitoring](#performance-monitoring)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

---

## 🛠️ Environment Setup

### 1. Clone and Install NeuroFlux

```bash
# Clone the repository
git clone https://github.com/nyrosveil/neuroflux.git
cd neuroflux

# Create conda environment (recommended)
conda create -n neuroflux python=3.10.9
conda activate neuroflux

# Or use venv
python -m venv neuroflux_env
source neuroflux_env/bin/activate  # Windows: neuroflux_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy template
cp .env_example .env

# Edit .env with your credentials
# Required: AI provider keys
ANTHROPIC_KEY=sk-ant-api03-...
OPENAI_KEY=sk-...

# Optional: Exchange API keys for live trading
SOLANA_PRIVATE_KEY=your_solana_private_key
HYPER_LIQUID_ETH_PRIVATE_KEY=your_hyperliquid_key

# Market data APIs (recommended)
BIRDEYE_API_KEY=your_birdeye_key
COINGECKO_API_KEY=your_coingecko_key
```

### 3. Verify Installation

```bash
# Test basic functionality
python -c "import neuroflux; print('NeuroFlux installed successfully')"

# Test agent import
python -c "from neuroflux.src.agents.trading_agent import TradingAgent; print('Trading agent ready')"
```

---

## ⚙️ Agent Configuration

### Core Agent Setup

```python
from neuroflux.src.agents.trading_agent import TradingAgent
from neuroflux.src.agents.risk_agent import RiskAgent
from neuroflux.src.agents.sentiment_agent import SentimentAgent
from neuroflux.src.config import Config

# Initialize configuration
config = Config()

# Create trading agent with flux sensitivity
trading_agent = TradingAgent(
    agent_id="main_trader",
    flux_level=0.7,  # Adaptive sensitivity (0.0-1.0)
    config=config
)

# Create risk management agent
risk_agent = RiskAgent(
    agent_id="risk_manager",
    flux_level=0.8,  # Conservative risk settings
    config=config
)

# Optional: Add sentiment analysis
sentiment_agent = SentimentAgent(
    agent_id="sentiment_analyzer",
    flux_level=0.6,
    config=config
)
```

### Flux Level Tuning

| Flux Level | Description | Use Case |
|------------|-------------|----------|
| 0.0-0.3 | Conservative | Low volatility, stable markets |
| 0.4-0.6 | Balanced | Normal market conditions |
| 0.7-0.8 | Aggressive | High volatility, trending markets |
| 0.9-1.0 | Experimental | Extreme conditions (use with caution) |

```python
# Dynamic flux adjustment based on market conditions
def adjust_flux_based_on_volatility(agent, volatility_score):
    if volatility_score > 0.8:
        agent.update_flux_level(0.9)  # More aggressive in high vol
    elif volatility_score < 0.3:
        agent.update_flux_level(0.4)  # More conservative in low vol
    else:
        agent.update_flux_level(0.7)  # Balanced default
```

---

## 🔄 Trading Workflow

### End-to-End Trading Process

```python
import time
from datetime import datetime

def run_trading_cycle():
    """Complete trading workflow execution"""

    # 1. Initialize agents
    agents = [trading_agent, risk_agent, sentiment_agent]
    for agent in agents:
        if not agent.initialize():
            print(f"Failed to initialize {agent.agent_id}")
            return False

    # 2. Start agents in background threads
    for agent in agents:
        agent.start()

    print("All agents initialized and running")

    # 3. Main trading loop
    while True:
        try:
            # Get market data
            market_data = trading_agent.get_market_data("BTC/USD")

            # Perform neuro-enhanced analysis
            analysis = trading_agent.analyze_market_neuro(market_data)

            # Check risk limits
            balance = risk_agent.get_portfolio_balance()
            positions = risk_agent.get_positions()
            flux_level = risk_agent.calculate_flux_level()

            risk_check = risk_agent.check_risk_limits(
                balance, positions, flux_level
            )

            if not risk_check['safe_to_trade']:
                print(f"Risk limits exceeded: {risk_check['reason']}")
                time.sleep(300)  # Wait 5 minutes
                continue

            # Execute trade if signal is strong
            if (analysis['signal'] != 'HOLD' and
                analysis['confidence'] > 0.75 and
                risk_check['safe_to_trade']):

                result = trading_agent.execute_trade(
                    signal=analysis['signal'],
                    token="BTC/USD",
                    amount=100.0,  # $100 trade
                    analysis=analysis
                )

                print(f"Trade executed: {result}")

            # Log performance metrics
            log_trading_metrics(analysis, risk_check, market_data)

            # Wait before next cycle (respect rate limits)
            time.sleep(60)  # 1-minute intervals

        except Exception as e:
            print(f"Trading cycle error: {e}")
            time.sleep(300)  # Wait 5 minutes on error

def log_trading_metrics(analysis, risk_check, market_data):
    """Log key trading metrics for monitoring"""
    timestamp = datetime.now().isoformat()

    metrics = {
        'timestamp': timestamp,
        'signal': analysis['signal'],
        'confidence': analysis['confidence'],
        'flux_level': market_data['flux_level'],
        'risk_safe': risk_check['safe_to_trade'],
        'portfolio_value': risk_check.get('portfolio_value', 0)
    }

    # Save to file or database
    with open('trading_metrics.jsonl', 'a') as f:
        f.write(json.dumps(metrics) + '\n')
```

### Signal Processing Pipeline

```
Market Data → Sentiment Analysis → Technical Analysis → Risk Check → Trade Execution

1. Market Data Collection
   ├── Price feeds (exchange APIs)
   ├── Volume data (24h metrics)
   ├── Technical indicators (RSI, MACD)
   └── Flux level calculation

2. Multi-Agent Analysis
   ├── Sentiment scoring (0-1 scale)
   ├── Technical signal generation
   └── Neural network enhancement

3. Risk Validation
   ├── Portfolio balance checks
   ├── Position limit validation
   └── Flux-adjusted thresholds

4. Trade Execution
   ├── Signal confirmation
   ├── Amount calculation
   └── Exchange order placement
```

---

## 🛡️ Risk Management Integration

### Circuit Breaker Configuration

```python
# Configure risk parameters in config.py or .env
RISK_CONFIG = {
    'max_loss_usd': 25.0,          # Maximum loss per trade
    'max_daily_loss': 100.0,       # Daily loss limit
    'max_position_size': 500.0,    # Maximum position size
    'min_balance': 50.0,           # Minimum account balance
    'max_open_positions': 5,       # Maximum concurrent positions
    'flux_sensitivity': 0.8,       # Risk adjustment factor
    'emergency_stop_loss': 0.95    # Emergency stop (5% loss)
}
```

### Risk Monitoring Dashboard

```python
def monitor_portfolio_health():
    """Real-time risk monitoring"""

    while True:
        # Get current portfolio status
        balance = risk_agent.get_portfolio_balance()
        positions = risk_agent.get_positions()
        flux_level = risk_agent.calculate_flux_level()

        # Calculate risk metrics
        total_pnl = sum(pos['pnl_usd'] for pos in positions)
        total_exposure = sum(abs(pos['size'] * pos['current_price'])
                           for pos in positions)

        # Check risk thresholds
        risk_status = {
            'total_pnl': total_pnl,
            'total_exposure': total_exposure,
            'flux_level': flux_level,
            'positions_count': len(positions),
            'available_balance': balance['available']
        }

        # Alert on risk violations
        if total_pnl < -RISK_CONFIG['max_daily_loss']:
            alert_critical("Daily loss limit exceeded")
            emergency_stop()

        if flux_level > 0.9:
            alert_warning("Extreme market volatility detected")

        if len(positions) > RISK_CONFIG['max_open_positions']:
            alert_warning("Too many open positions")

        time.sleep(30)  # Check every 30 seconds

def emergency_stop():
    """Emergency position closure"""
    print("EMERGENCY STOP ACTIVATED")

    positions = risk_agent.get_positions()
    for position in positions:
        # Close all positions immediately
        risk_agent.emergency_close_position(position['symbol'])

    # Stop all trading agents
    trading_agent.stop()
    print("All positions closed, trading halted")
```

### Flux-Adaptive Risk

```python
def adjust_risk_parameters(flux_level):
    """Dynamically adjust risk based on market flux"""

    if flux_level < 0.3:
        # Low volatility - relaxed risk
        return {
            'max_position_size': 1000.0,
            'min_confidence': 0.6,
            'trade_frequency': 'high'
        }

    elif flux_level < 0.7:
        # Normal volatility - standard risk
        return {
            'max_position_size': 500.0,
            'min_confidence': 0.75,
            'trade_frequency': 'medium'
        }

    else:
        # High volatility - conservative risk
        return {
            'max_position_size': 100.0,
            'min_confidence': 0.85,
            'trade_frequency': 'low'
        }
```

---

## 📊 Performance Monitoring

### Key Metrics to Track

```python
def collect_performance_metrics():
    """Comprehensive performance tracking"""

    metrics = {
        'timestamp': datetime.now().isoformat(),
        'portfolio': {
            'total_value': 0,
            'daily_pnl': 0,
            'win_rate': 0,
            'sharpe_ratio': 0
        },
        'trading': {
            'total_trades': 0,
            'successful_trades': 0,
            'average_win': 0,
            'average_loss': 0
        },
        'risk': {
            'max_drawdown': 0,
            'volatility': 0,
            'flux_level': 0
        },
        'system': {
            'cpu_usage': 0,
            'memory_usage': 0,
            'agent_status': 'healthy'
        }
    }

    # Calculate portfolio metrics
    balance = risk_agent.get_portfolio_balance()
    positions = risk_agent.get_positions()

    metrics['portfolio']['total_value'] = balance['equity']
    metrics['risk']['flux_level'] = risk_agent.calculate_flux_level()

    # Calculate trading performance
    # (Implementation depends on your trade history storage)

    return metrics
```

### Alert System Setup

```python
def setup_alerts():
    """Configure monitoring alerts"""

    alerts = {
        'critical': {
            'portfolio_loss_10pct': 'Portfolio down 10%+ today',
            'api_key_expired': 'Exchange API key expired',
            'agent_crashed': 'Critical agent stopped responding'
        },
        'warning': {
            'high_volatility': 'Flux level > 0.8',
            'low_balance': 'Available balance < $100',
            'stale_data': 'Market data > 5 minutes old'
        },
        'info': {
            'trade_executed': 'New trade completed',
            'agent_restarted': 'Agent recovered from error'
        }
    }

    # Configure notification channels
    # Email, SMS, Discord, Telegram, etc.

    return alerts

def send_alert(level, message, details=None):
    """Send alert through configured channels"""
    timestamp = datetime.now().isoformat()

    alert_data = {
        'level': level,
        'message': message,
        'timestamp': timestamp,
        'details': details or {}
    }

    # Send to monitoring service
    print(f"[{level.upper()}] {message}")

    # In production: send to email/SMS/webhook
    # send_email_alert(alert_data)
    # send_sms_alert(alert_data)
    # send_discord_webhook(alert_data)
```

### Dashboard Integration

```python
# Example: Simple web dashboard with Flask
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/metrics')
def get_metrics():
    """REST API endpoint for metrics"""
    metrics = collect_performance_metrics()
    return jsonify(metrics)

@app.route('/api/health')
def health_check():
    """System health check endpoint"""
    health = {
        'status': 'healthy',
        'agents': {
            'trading': trading_agent.is_running(),
            'risk': risk_agent.is_running(),
            'sentiment': sentiment_agent.is_running()
        },
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(health)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## 🚀 Production Deployment

### Docker Container Setup

```dockerfile
# Dockerfile for NeuroFlux trading system
FROM python:3.10.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash neuroflux
USER neuroflux

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "from neuroflux.src.agents.base_agent import BaseAgent; print('OK')"

# Start command
CMD ["python", "src/agents/trading_agent.py"]
```

### Production Configuration

```python
# production_config.py
PRODUCTION_CONFIG = {
    'environment': 'production',
    'log_level': 'INFO',
    'database_url': 'postgresql://user:pass@host:5432/neuroflux',
    'redis_url': 'redis://host:6379',
    'monitoring': {
        'enabled': True,
        'prometheus_port': 9090,
        'grafana_dashboard': True
    },
    'alerts': {
        'email_enabled': True,
        'sms_enabled': True,
        'webhook_url': 'https://hooks.slack.com/...',
        'alert_thresholds': {
            'critical_loss': 0.05,  # 5% loss
            'warning_loss': 0.02    # 2% loss
        }
    },
    'trading': {
        'max_concurrent_trades': 3,
        'trade_cooldown_seconds': 300,
        'flux_adaptive_enabled': True,
        'backtesting_enabled': False
    }
}
```

### Systemd Service (Linux)

```ini
# /etc/systemd/system/neuroflux.service
[Unit]
Description=NeuroFlux Trading System
After=network.target
Wants=network.target

[Service]
Type=simple
User=neuroflux
Group=neuroflux
WorkingDirectory=/opt/neuroflux
ExecStart=/opt/neuroflux/venv/bin/python src/agents/trading_agent.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PATH=/opt/neuroflux/venv/bin
Environment=PYTHONPATH=/opt/neuroflux

[Install]
WantedBy=multi-user.target
```

### Deployment Commands

```bash
# Build and deploy
sudo systemctl stop neuroflux  # Stop existing service
git pull origin main           # Update code
pip install -r requirements.txt  # Update dependencies
sudo systemctl start neuroflux   # Start service
sudo systemctl enable neuroflux  # Enable auto-start

# Monitor logs
journalctl -u neuroflux -f

# Check status
sudo systemctl status neuroflux
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Agent Initialization Failures

**Problem:** Agent fails to initialize
```
Error: Failed to initialize trading_agent
```

**Solutions:**
```python
# Check configuration
from neuroflux.src.config import Config
config = Config()
print("Config loaded:", config.is_valid())

# Verify API keys
import os
print("Anthropic key exists:", bool(os.getenv('ANTHROPIC_KEY')))
print("Exchange keys configured:", bool(os.getenv('SOLANA_PRIVATE_KEY')))

# Test agent creation
try:
    from neuroflux.src.agents.trading_agent import TradingAgent
    agent = TradingAgent(agent_id="test")
    print("Agent created successfully")
except Exception as e:
    print(f"Agent creation failed: {e}")
```

#### 2. Trading Signal Issues

**Problem:** No trades executing despite market movement

**Debug Steps:**
```python
# Check market data
market_data = trading_agent.get_market_data("BTC/USD")
print("Market data:", market_data)

# Test analysis
analysis = trading_agent.analyze_market_neuro(market_data)
print("Analysis result:", analysis)

# Check risk status
balance = risk_agent.get_portfolio_balance()
positions = risk_agent.get_positions()
flux_level = risk_agent.calculate_flux_level()

risk_check = risk_agent.check_risk_limits(balance, positions, flux_level)
print("Risk check:", risk_check)

# Verify confidence threshold
min_confidence = 0.75
if analysis['confidence'] < min_confidence:
    print(f"Signal confidence {analysis['confidence']} below threshold {min_confidence}")
```

#### 3. High Latency or Performance Issues

**Problem:** System responding slowly

**Performance Tuning:**
```python
# Check system resources
import psutil
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Optimize agent configuration
trading_agent.update_flux_level(0.5)  # Reduce computational load

# Check thread status
print(f"Trading agent running: {trading_agent.is_running()}")
print(f"Active threads: {threading.active_count()}")

# Database connection pooling
# Redis caching for market data
# Async processing for non-critical tasks
```

#### 4. Exchange API Errors

**Problem:** Exchange connectivity issues

**API Troubleshooting:**
```python
# Test exchange connection
from neuroflux.src.exchanges.base_exchange import BaseExchange

try:
    exchange = BaseExchange()
    balance = exchange.get_balance()
    print("Exchange connection successful")
    print("Balance:", balance)
except Exception as e:
    print(f"Exchange error: {e}")

# Check API key permissions
# Verify rate limits not exceeded
# Test with different exchange endpoints
```

#### 5. Memory Leaks or Crashes

**Problem:** System becoming unstable over time

**Memory Management:**
```python
# Monitor memory usage
import gc
print(f"Objects in memory: {len(gc.get_objects())}")

# Force garbage collection
gc.collect()

# Check for circular references
# Implement proper cleanup in agent stop() methods
# Use memory profiling tools
```

### Debug Logging Setup

```python
import logging

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neuroflux_debug.log'),
        logging.StreamHandler()
    ]
)

# Add agent-specific loggers
trading_logger = logging.getLogger('trading_agent')
risk_logger = logging.getLogger('risk_agent')

# Log with context
def log_with_context(logger, message, context=None):
    if context:
        logger.info(f"{message} | Context: {context}")
    else:
        logger.info(message)
```

### Emergency Procedures

```python
def emergency_shutdown():
    """Complete system shutdown for critical issues"""

    print("INITIATING EMERGENCY SHUTDOWN")

    # Stop all agents gracefully
    agents = [trading_agent, risk_agent, sentiment_agent]
    for agent in agents:
        try:
            agent.stop(timeout=30.0)
            print(f"Stopped {agent.agent_id}")
        except Exception as e:
            print(f"Error stopping {agent.agent_id}: {e}")

    # Close all positions
    positions = risk_agent.get_positions()
    for position in positions:
        try:
            risk_agent.emergency_close_position(position['symbol'])
            print(f"Closed position: {position['symbol']}")
        except Exception as e:
            print(f"Error closing {position['symbol']}: {e}")

    # Save final state
    save_system_state()

    print("EMERGENCY SHUTDOWN COMPLETE")

def save_system_state():
    """Save critical system state for recovery"""
    state = {
        'timestamp': datetime.now().isoformat(),
        'agents_status': {
            'trading': trading_agent.status.value,
            'risk': risk_agent.status.value,
            'sentiment': sentiment_agent.status.value
        },
        'portfolio': risk_agent.get_portfolio_balance(),
        'positions': risk_agent.get_positions()
    }

    with open('emergency_state.json', 'w') as f:
        json.dump(state, f, indent=2)
```

---

## 📚 Additional Resources

- **[API Documentation](../api/README.md)** - Complete technical reference
- **[Multi-Agent Coordination Guide](multi_agent_coordination.md)** - Advanced agent orchestration
- **[Exchange Setup Guide](exchange_setup.md)** - Exchange integration details
- **[Custom Agent Development](custom_agent_development.md)** - Building custom agents

---

## ⚠️ Important Notes

- **Risk Warning**: Trading cryptocurrencies involves substantial risk of loss
- **Backtesting**: Always backtest strategies before live deployment
- **Monitoring**: Implement comprehensive monitoring and alerting
- **Security**: Never commit API keys to version control
- **Updates**: Regularly update NeuroFlux for latest features and security patches

---

*Built with ❤️ by Nyros Veil | [GitHub](https://github.com/nyrosveil/neuroflux) | [Issues](https://github.com/nyrosveil/neuroflux/issues)*