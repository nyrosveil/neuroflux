---
name: neuroflux-trading-agents
description: Master NeuroFlux's AI trading system with 48+ neuro-enhanced agents, flux-based adaptation, multi-exchange support, LLM abstraction, and autonomous trading capabilities across crypto markets
---

# NeuroFlux AI Trading System

Expert knowledge for working with NeuroFlux's experimental AI trading system that orchestrates 48+ specialized neuro-enhanced AI agents for cryptocurrency trading across Hyperliquid, Solana (BirdEye), and Extended Exchange with flux-based adaptive learning.

## When to Use This Skill

Use this skill when:
- Working with NeuroFlux trading agents repository
- Need to understand neuro-enhanced agent architecture and capabilities
- Running, modifying, or creating flux-adaptive trading agents
- Configuring trading system, exchanges, or LLM providers
- Debugging neuro-flux operations or agent interactions
- Understanding backtesting with RBI agent
- Setting up new exchanges or neuro-inspired strategies

## Environment Setup Note

**For New Users**: This repo uses Python 3.10.9. If using conda, create environment named `neuroflux`, but you can name it whatever you want. If you don't use conda, standard pip/venv works fine too.

## Quick Start Commands

```bash
# Activate your Python environment (conda, venv, or whatever you use)
# Example with conda: conda activate neuroflux
# Example with venv: source neuroflux_env/bin/activate
# Use whatever environment manager you prefer

# Run main orchestrator (controls multiple neuro-flux agents)
python src/main.py

# Run individual neuro-enhanced agent
python src/agents/trading_agent.py
python src/agents/risk_agent.py
python src/agents/rbi_agent.py

# Update requirements after adding packages
pip freeze > requirements.txt
```

## Core Architecture

### Directory Structure

```
src/
├── agents/              # 48+ neuro-enhanced AI agents (<800 lines each)
├── models/              # LLM provider abstraction (ModelFactory)
├── strategies/          # User-defined neuro-flux trading strategies
├── scripts/             # Standalone utility scripts
├── data/                # Agent outputs, memory, analysis results
├── config.py            # Global configuration with neuro-flux settings
├── main.py              # Main orchestrator loop with flux monitoring
├── nice_funcs.py        # Core trading utilities with flux adaptation (~1,200 lines)
├── nice_funcs_hl.py     # Hyperliquid-specific functions
├── nice_funcs_extended.py # Extended Exchange functions
└── ezbot.py             # Legacy trading controller
```

### Key Components

**Neuro-Enhanced Agents** (src/agents/)
- Each agent is standalone executable with flux-awareness
- Uses ModelFactory for LLM access with neuro-inspired prompts
- Stores outputs in src/data/[agent_name]/
- Under 800 lines (split if longer)

**LLM Integration** (src/models/)
- ModelFactory provides unified interface with flux-adaptive parameters
- Supports: Claude, GPT-4, DeepSeek, Groq, Gemini, Ollama
- Pattern: `ModelFactory.create_model('anthropic')`

**Neuro-Flux Trading Utilities**
- `nice_funcs.py`: Core functions with flux monitoring (Solana/BirdEye)
- `nice_funcs_hl.py`: Hyperliquid exchange with adaptive leverage
- `nice_funcs_extended.py`: Extended Exchange with neuro-positioning

**Configuration**
- `config.py`: Trading settings with neuro-flux parameters
- `.env`: API keys and secrets (never expose these)

## Agent Categories

**Trading**: trading_agent, strategy_agent, risk_agent, copybot_agent

**Market Analysis**: sentiment_agent, whale_agent, funding_agent, liquidation_agent, chartanalysis_agent, coingecko_agent

**Content**: chat_agent, clips_agent, tweet_agent, video_agent, phone_agent

**Research**: rbi_agent (codes backtests from videos/PDFs), research_agent, websearch_agent

**Specialized**: sniper_agent, solana_agent, tx_agent, million_agent, polymarket_agent, compliance_agent, focus_agent

**Arbitrage**: fundingarb_agent, listingarb_agent

**Coordination**: swarm_agent (enhanced with neuro-consensus), base_agent

**Infrastructure**: api, backtest_runner, code_runner_agent, clean_ideas, new_or_top_agent, stream_agent, demo_countdown

See complete agent list below.

## Common Workflows

### 1. Run Single Neuro-Flux Agent

```bash
# Activate environment
python src/agents/[agent_name].py
```

Each agent is standalone and can run independently with flux monitoring.

### 2. Run Main NeuroFlux Orchestrator

```bash
python src/main.py
```

Runs multiple agents in loop based on `ACTIVE_AGENTS` dict in main.py, with real-time flux adaptation.

Configure which agents run:
```python
ACTIVE_AGENTS = {
    "risk_agent": True,        # Always first
    "trading_agent": True,
    "sentiment_agent": False,  # Disabled
    "whale_agent": True,
}
```

### 3. Change Exchange with Neuro-Flux Adaptation

Edit `src/agents/trading_agent.py`:

```python
# Exchange Configuration - Line ~20
EXCHANGE = "hyperliquid"  # Options: "hyperliquid", "birdeye", "extended"

# Import corresponding functions
if EXCHANGE == "hyperliquid":
    from src import nice_funcs_hl as nf
elif EXCHANGE == "extended":
    from src import nice_funcs_extended as nf
elif EXCHANGE == "birdeye":
    from src import nice_funcs as nf
```

### 4. Switch AI Model with Flux Sensitivity

Edit `src/config.py`:

```python
# AI Configuration with neuro-flux parameters
AI_MODEL = "claude-3-haiku-20240307"    # Fast, cheap
# AI_MODEL = "claude-3-sonnet-20240229"  # Balanced (recommended)
# AI_MODEL = "claude-3-opus-20240229"    # Most powerful, expensive

AI_MAX_TOKENS = 4000
AI_TEMPERATURE = 0.3  # Lower for flux stability
FLUX_SENSITIVITY = 0.8  # Neuro-flux adaptation threshold
```

### 5. Backtest Strategy with Neuro-Optimization

```bash
python src/agents/rbi_agent.py
```

Workflow:
1. Agent prompts: "Provide YouTube URL, PDF path, or describe strategy"
2. You provide input (e.g., YouTube link to trading tutorial)
3. DeepSeek-R1 extracts strategy logic with neuro-analysis
4. Generates backtesting.py compatible code with flux adaptation
5. Executes backtest with neuro-optimized parameters
6. Returns performance metrics with flux sensitivity analysis

## Development Rules

### CRITICAL Rules

1. **Keep files under 800 lines** - split into new files if longer
2. **NEVER move files** - can create new, but no moving without asking
3. **Use existing environment** - don't create new virtual environments, use the one from initial setup
4. **Update requirements.txt** after any pip install: `pip freeze > requirements.txt`
5. **Use real data only** - never synthetic/fake data
6. **Minimal error handling** - user wants to see errors, not over-engineered try/except
7. **Never expose API keys** - don't show .env contents

### Neuro-Flux Agent Development Pattern

Creating new neuro-enhanced agents:
```python
# 1. Use ModelFactory for LLM with flux parameters
from src.models.model_factory import ModelFactory
model = ModelFactory.create_model('anthropic')

# 2. Store outputs in src/data/
output_dir = "src/data/my_neuro_agent/"

# 3. Make independently executable
if __name__ == "__main__":
    # Standalone logic here

# 4. Follow naming: [purpose]_agent.py

# 5. Add neuro-flux features
flux_sensitivity = config.FLUX_SENSITIVITY
neural_layers = config.NEURAL_NETWORK_LAYERS
```

### Backtesting with Neuro-Optimization

- Use `backtesting.py` library (NOT built-in indicators)
- Use `pandas_ta` or `talib` for indicators
- Sample data: `src/data/rbi/BTC-USD-15m.csv`
- Add neuro-flux optimization layers

## Configuration Files

**config.py**: Trading settings with neuro-flux parameters
```python
# Monitored assets
MONITORED_TOKENS = ["BTC", "ETH", "SOL"]
EXCLUDED_TOKENS = ["SCAM", "RUG"]

# Position sizing
usd_size = 100
max_usd_order_size = 500

# Risk limits with flux adaptation
CASH_PERCENTAGE = 10
MAX_POSITION_PERCENTAGE = 50
MAX_LOSS_USD = 500
MAX_GAIN_USD = 2000
MINIMUM_BALANCE_USD = 100

# Agent behavior
SLEEP_BETWEEN_RUNS_MINUTES = 15

# AI settings
AI_MODEL = "claude-3-sonnet-20240229"
AI_MAX_TOKENS = 4000
AI_TEMPERATURE = 0.3

# NeuroFlux specific settings
FLUX_SENSITIVITY = 0.8
ADAPTIVE_LEARNING_RATE = 0.1
NEURAL_NETWORK_LAYERS = [64, 32, 16]
SWARM_SIZE = 6
```

**.env**: Secrets (NEVER expose)
```bash
# AI Providers
ANTHROPIC_KEY=sk-ant-...
OPENAI_KEY=sk-...
DEEPSEEK_KEY=...
GROQ_API_KEY=...
GEMINI_KEY=...
XAI_API_KEY=...

# Trading APIs
BIRDEYE_API_KEY=...
MOONDEV_API_KEY=...
COINGECKO_API_KEY=...

# Exchanges
HYPER_LIQUID_ETH_PRIVATE_KEY=0x...
X10_API_KEY=...
X10_PRIVATE_KEY=0x...
X10_PUBLIC_KEY=...
X10_VAULT_ID=...

# Blockchain
SOLANA_PRIVATE_KEY=...
RPC_ENDPOINT=https://api.mainnet-beta.solana.com/
```

## Exchange Support with Neuro-Flux

**Hyperliquid** (`nice_funcs_hl.py`)
- EVM-compatible perpetuals DEX with adaptive leverage
- Functions: `market_buy()`, `market_sell()`, `get_position()`, `close_position()`
- Leverage up to 50x with flux-based adjustment

**BirdEye/Solana** (`nice_funcs.py`)
- Solana token data + spot trading with neuro-analysis
- Functions: `token_overview()`, `token_price()`, `get_ohlcv_data()`
- Real-time market data for 15,000+ tokens with flux monitoring

**Extended Exchange** (`nice_funcs_extended.py`)
- StarkNet perpetuals DEX with neuro-positioning
- Auto symbol conversion (BTC → BTC-USD)
- Leverage up to 20x with AI confirmation

## Data Flow Pattern with Neuro-Flux

```
Config/Input → Agent Init → Flux Monitoring → API Data Fetch →
Data Parsing → Neuro Analysis (neural networks) → Flux Adaptation →
LLM Analysis (via ModelFactory) → Decision Output →
Result Storage (CSV/JSON in src/data/) → Optional Trade Execution
```

## Common Tasks

**Add new package:**
```bash
pip install package-name
pip freeze > requirements.txt
```

**Read market data with flux analysis:**
```python
from src.nice_funcs import token_overview, get_ohlcv_data, token_price

overview = token_overview(token_address)
ohlcv = get_ohlcv_data(token_address, timeframe='1H', days_back=3)
price = token_price(token_address)
```

**Execute trade with neuro-confirmation:**
```python
from src import nice_funcs_hl as nf

# Neuro-flux enhanced trading
flux_level = calculate_market_flux()
if flux_level > config.FLUX_SENSITIVITY:
    nf.market_buy("BTC", usd_amount=100, leverage=10)
position = nf.get_position("BTC")
nf.close_position("BTC")
```

## Risk Management with Flux Awareness

- Risk Agent runs FIRST before any trading decisions
- Circuit breakers: MAX_LOSS_USD, MAX_GAIN_USD, MINIMUM_BALANCE_USD
- Neuro-flux adaptive risk: Adjust limits based on market volatility
- AI confirmation for position-closing (configurable)
- Default loop: every 15 minutes with flux monitoring

## Philosophy

This is an **experimental, educational project** with neuro-flux innovation:
- No guarantees of profitability
- Open source and free
- Community-supported
- No official token (avoid scams)

Goal: Democratize neuro-inspired AI agent development.

## Complete Agent List

### Trading Agents
- **trading_agent.py**: Primary trading execution with neuro-decision making
- **strategy_agent.py**: Executes user-defined strategies with flux adaptation
- **risk_agent.py**: Monitors portfolio risk with neuro-flux awareness
- **copybot_agent.py**: Copies trades with AI-enhanced analysis

### Market Analysis Agents
- **sentiment_agent.py**: Analyzes market sentiment with neural processing
- **whale_agent.py**: Tracks large wallet movements with flux detection
- **funding_agent.py**: Monitors funding rates with neuro-prediction
- **liquidation_agent.py**: Tracks liquidation data with AI analysis
- **chartanalysis_agent.py**: Technical analysis using AI vision with neural networks
- **coingecko_agent.py**: Fetches token metadata with flux enrichment

### Research & Strategy Development
- **rbi_agent.py**: Codes backtests with neuro-optimization
- **rbi_agent_v2.py**: Alternative RBI with enhanced parsing
- **rbi_agent_v3.py**: Latest RBI with flux adaptation
- **rbi_agent_pp.py**: PowerPoint integration
- **rbi_agent_pp_multi.py**: Multi-slide analysis with parallel processing
- **rbi_batch_backtester.py**: Batch backtest multiple strategies
- **research_agent.py**: General market research with AI
- **websearch_agent.py**: Web search with neuro-filtering

### Content Creation Agents
- **chat_agent.py**: Conversational AI for trading questions
- **chat_agent_og.py**: Original chat implementation
- **chat_agent_ad.py**: Ad-supported variant
- **chat_question_generator.py**: Generates questions for engagement
- **clips_agent.py**: Creates short video clips
- **realtime_clips_agent.py**: Real-time clip generation
- **tweet_agent.py**: Generates tweets with AI
- **video_agent.py**: Full-length video content
- **shortvid_agent.py**: Short-form video (TikTok, Reels)
- **tiktok_agent.py**: TikTok-specific content
- **phone_agent.py**: Voice/phone interaction

### Specialized Trading Agents
- **sniper_agent.py**: Fast execution for new launches with neuro-timing
- **solana_agent.py**: Solana-specific operations with flux monitoring
- **tx_agent.py**: Transaction monitoring with AI analysis
- **million_agent.py**: High-volume strategies with risk adaptation
- **polymarket_agent.py**: Polymarket prediction markets
- **housecoin_agent.py**: Custom token specific agent
- **compliance_agent.py**: Regulatory compliance monitoring
- **focus_agent.py**: Concentrated analysis on selected tokens

### Arbitrage Agents
- **fundingarb_agent.py**: Funding rate arbitrage with flux timing
- **listingarb_agent.py**: New listing arbitrage opportunities

### Multi-Agent Coordination
- **swarm_agent.py**: Coordinates multiple agents with neuro-consensus
- **base_agent.py**: Base class for all agents

### Infrastructure & Utilities
- **api.py**: MoonDevAPI class for custom endpoints
- **backtest_runner.py**: Executes backtests programmatically
- **code_runner_agent.py**: Executes generated code safely
- **clean_ideas.py**: Cleans and organizes trading ideas
- **new_or_top_agent.py**: Identifies new or trending tokens
- **stream_agent.py**: Real-time data streaming
- **demo_countdown.py**: Demo countdown timer

## Agent Selection Guide

**Want to trade?** → `trading_agent.py`

**Need risk management?** → `risk_agent.py` (runs first!)

**Test a strategy?** → `rbi_agent.py` (backtest from video/PDF)

**Custom strategy?** → `strategy_agent.py` (load from src/strategies/)

**Market sentiment?** → `sentiment_agent.py`, `whale_agent.py`

**Find opportunities?** → `funding_agent.py`, `liquidation_agent.py`

**Prediction markets?** → `polymarket_agent.py`

**Copy traders?** → `copybot_agent.py`

**Create content?** → `chat_agent.py`, `tweet_agent.py`, `video_agent.py`

**On-chain analysis?** → `tx_agent.py`, `solana_agent.py`

**Coordinate multiple agents?** → `swarm_agent.py`

## Creating New Neuro-Flux Agents

Follow the agent development pattern:

```python
"""
🌙 NeuroFlux's [Agent Name]
[Brief description with neuro-flux features]
"""

from src.models.model_factory import ModelFactory
from termcolor import cprint
import os

# Configuration with neuro-flux parameters
OUTPUT_DIR = "src/data/my_neuro_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """Main agent logic with flux adaptation"""
    cprint("🧠 NeuroFlux's [Agent Name] starting...", "cyan")

    # Initialize AI model with flux parameters
    model = ModelFactory.create_model('anthropic')

    # Flux monitoring
    flux_level = calculate_market_flux()
    if flux_level > config.FLUX_SENSITIVITY:
        # Neuro-adaptive behavior
        adaptive_temperature = config.AI_TEMPERATURE * (1 + flux_level)
        model.set_temperature(adaptive_temperature)

    # Agent logic here with neuro-enhancements
    # ...

    # Store results
    # Save to OUTPUT_DIR

    cprint("✅ Neuro agent complete!", "green")

if __name__ == "__main__":
    main()
```

**Requirements:**
- Under 800 lines (split if longer)
- Use ModelFactory for LLM access with flux parameters
- Store outputs in src/data/[agent_name]/
- Make independently executable
- Follow naming: [purpose]_agent.py
- Include neuro-flux features (neural networks, adaptation, flux monitoring)

---

**Built with 🧠 by Nyros Veil**

*"Neuro-inspired intelligence for adaptive trading in dynamic markets."*