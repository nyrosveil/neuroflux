# 🧠 NeuroFlux - AI Trading with Neuro-Inspired Algorithms

<p align="center">
  <img src="https://via.placeholder.com/300x100?text=NeuroFlux" width="300" alt="NeuroFlux">
</p>

## 🎯 Vision

NeuroFlux is an advanced AI trading system that combines neuro-inspired algorithms, flux-based adaptive learning, and enhanced swarm intelligence for dynamic market conditions. Our mission is to create trading systems that adapt in real-time to market flux, using neural networks and collective intelligence for superior performance.

**Key Innovations:**
- 🧠 **Neuro-Inspired Algorithms**: Neural network-based decision making
- 🌊 **Flux-Based Adaptation**: Real-time market flux monitoring and response
- 🐝 **Enhanced Swarm Intelligence**: Multi-agent consensus with adaptive learning
- 🔄 **Dynamic Strategy Evolution**: Strategies that evolve based on market conditions

## 📋 Features

- **Multi-Exchange Support**: Solana, HyperLiquid, Extended Exchange
- **48+ Specialized Agents**: Trading, analysis, research, content creation
- **LLM Abstraction Layer**: Support for 6+ AI providers
- **Risk-First Architecture**: Circuit breakers and position management
- **Backtesting Framework**: RBI agent for automated strategy testing
- **Real-Time Flux Monitoring**: Adaptive responses to market changes

## 🚀 Quick Start

### Prerequisites
- Python 3.10.9
- Conda (recommended) or venv

### Installation

1. **Clone and Setup Environment**
```bash
git clone https://github.com/yourusername/neuroflux.git
cd neuroflux

# Create conda environment
conda create -n neuroflux python=3.10.9
conda activate neuroflux

# Or use venv
python -m venv neuroflux_env
source neuroflux_env/bin/activate  # On Windows: neuroflux_env\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment**
```bash
# Copy environment template
cp .env_example .env

# Edit .env with your API keys
# Required: At least one AI provider (Anthropic, OpenAI, etc.)
# Optional: Exchange APIs for live trading
```

4. **Run Your First Agent**
```bash
# Activate environment first
conda activate neuroflux  # or source neuroflux_env/bin/activate

# Run a simple agent
python src/agents/chat_agent.py
```

## 🏗️ Architecture

```
neuroflux/
├── src/
│   ├── agents/          # 48+ AI agents
│   ├── models/          # LLM provider abstraction
│   ├── strategies/      # Trading strategies
│   ├── data/           # Agent outputs and memory
│   └── config.py       # Global configuration
├── docs/               # Documentation
├── .claude/           # Claude skill for expert guidance
├── .env_example       # API key template
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🤖 Agent Categories

| Category | Count | Examples |
|----------|-------|----------|
| **Trading** | 4 | trading_agent, strategy_agent, risk_agent, copybot_agent |
| **Market Analysis** | 6 | sentiment_agent, whale_agent, funding_agent, liquidation_agent |
| **Research** | 7 | rbi_agent, research_agent, websearch_agent |
| **Content Creation** | 8 | chat_agent, tweet_agent, video_agent |
| **Specialized** | 8 | sniper_agent, solana_agent, polymarket_agent |
| **Arbitrage** | 2 | fundingarb_agent, listingarb_agent |
| **Coordination** | 2 | swarm_agent, base_agent |
| **Infrastructure** | 7 | api, backtest_runner, code_runner_agent |

## ⚙️ Configuration

### Core Settings (config.py)
- **Exchange Selection**: `EXCHANGE = 'solana'` (options: solana, hyperliquid, extended)
- **Risk Management**: `MAX_LOSS_USD = 25`, `MINIMUM_BALANCE_USD = 50`
- **AI Models**: `AI_MODEL = "claude-3-sonnet-20240229"`
- **NeuroFlux Settings**: `FLUX_SENSITIVITY = 0.8`, `NEURAL_NETWORK_LAYERS = [64, 32, 16]`

### Environment Variables (.env)
```bash
# AI Providers (at least one)
ANTHROPIC_KEY=sk-ant-...
OPENAI_KEY=sk-...

# Market Data
BIRDEYE_API_KEY=...
COINGECKO_API_KEY=...

# Blockchain/Exchanges
SOLANA_PRIVATE_KEY=...
HYPER_LIQUID_ETH_PRIVATE_KEY=...
```

## 🧪 Backtesting

Use the RBI (Research-Based Inference) agent to automatically backtest strategies:

```bash
python src/agents/rbi_agent.py
```

Provide YouTube URLs, PDFs, or text descriptions of trading strategies. The agent will:
1. Extract strategy logic using AI
2. Generate backtesting code
3. Test across multiple datasets
4. Return performance metrics

## 🛡️ Risk Management

NeuroFlux implements a risk-first approach:
- **Circuit Breakers**: Automatic position closure on loss/gain limits
- **Flux-Aware Risk**: Adaptive risk based on market volatility
- **AI Confirmation**: Optional AI review before emergency closures
- **Position Limits**: Maximum allocation per position and total exposure

## 📊 Data Flow

```
User Input / Scheduler
    ↓
Main Orchestrator (main.py)
    ↓
Risk Agent (circuit breaker check)
    ↓
Active Agents (parallel execution)
    ↓
ModelFactory → AI Provider
    ↓
Exchange API (Solana/HyperLiquid/Extended)
    ↓
Market Data Processing
    ↓
Neuro-Flux Analysis (neural networks, adaptation)
    ↓
Decision Output
    ↓
Result Storage (src/data/)
    ↓
Optional: Trade Execution
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚠️ Disclaimer

**This is an experimental project for educational purposes only.**

- No guarantees of profitability
- Trading involves substantial risk of loss
- Past performance does not indicate future results
- Always backtest strategies before live trading
- Use at your own risk

## 📞 Support

- **Discord**: [Join our community](https://discord.gg/neuroflux)
- **Documentation**: Check the `docs/` folder
- **Issues**: Report bugs on GitHub

## 📜 License

This project is open source and available under the MIT License.

---

**Built with 🧠 by Nyros Veil**

*Advancing AI trading through neuro-inspired intelligence and adaptive flux.*