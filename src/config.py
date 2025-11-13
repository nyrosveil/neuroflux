"""
🌙 NeuroFlux Configuration File
Built with love by Nyros Veil 🚀
"""

# 🔄 Exchange Selection
EXCHANGE = 'solana'  # Options: 'solana', 'hyperliquid', 'extended'

# 💰 Trading Configuration
USDC_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # Never trade or close
SOL_ADDRESS = "So11111111111111111111111111111111111111111"   # Never trade or close

# Create a list of addresses to exclude from trading/closing
EXCLUDED_TOKENS = [USDC_ADDRESS, SOL_ADDRESS]

# Token List for Trading 📋
MONITORED_TOKENS = [
    # Add your tokens here
]

# NeuroFlux Token Trading List 🚀
tokens_to_trade = MONITORED_TOKENS  # Using the same list for trading

# ⚡ HyperLiquid Configuration
HYPERLIQUID_SYMBOLS = ['BTC', 'ETH', 'SOL']  # Symbols to trade on HyperLiquid perps
HYPERLIQUID_LEVERAGE = 5  # Default leverage for HyperLiquid trades (1-50)

# 🔄 Exchange-Specific Token Lists
def get_active_tokens():
    """Returns the appropriate token/symbol list based on active exchange"""
    if EXCHANGE == 'hyperliquid':
        return HYPERLIQUID_SYMBOLS
    else:
        return MONITORED_TOKENS

# Token to Exchange Mapping
TOKEN_EXCHANGE_MAP = {
    'BTC': 'hyperliquid',
    'ETH': 'hyperliquid',
    'SOL': 'hyperliquid',
    # All other tokens default to Solana
}

# Token and wallet settings
symbol = 'SOL'  # Default symbol
address = 'YOUR_WALLET_ADDRESS_HERE'  # YOUR WALLET ADDRESS HERE

# Position sizing 🎯
usd_size = 25  # Size of position to hold
max_usd_order_size = 3  # Max order size
tx_sleep = 30  # Sleep between transactions
slippage = 199  # Slippage settings

# Risk Management Settings 🛡️
CASH_PERCENTAGE = 20  # Minimum % to keep in USDC as safety buffer (0-100)
MAX_POSITION_PERCENTAGE = 30  # Maximum % allocation per position (0-100)
STOPLOSS_PRICE = 1  # NOT USED YET
BREAKOUT_PRICE = .0001  # NOT USED YET
SLEEP_AFTER_CLOSE = 600  # Prevent overtrading

MAX_LOSS_GAIN_CHECK_HOURS = 12  # How far back to check for max loss/gain limits (in hours)
SLEEP_BETWEEN_RUNS_MINUTES = 15  # How long to sleep between agent runs 🕒

# Max Loss/Gain Settings FOR RISK AGENT
USE_PERCENTAGE = False  # If True, use percentage-based limits. If False, use USD-based limits

# USD-based limits (used if USE_PERCENTAGE is False)
MAX_LOSS_USD = 25  # Maximum loss in USD before stopping trading
MAX_GAIN_USD = 25  # Maximum gain in USD before stopping trading

# USD MINIMUM BALANCE RISK CONTROL
MINIMUM_BALANCE_USD = 50  # If balance falls below this, risk agent will consider closing all positions
USE_AI_CONFIRMATION = True  # If True, consult AI before closing positions. If False, close immediately on breach

# Percentage-based limits (used if USE_PERCENTAGE is True)
MAX_LOSS_PERCENT = 5  # Maximum loss as percentage (e.g., 20 = 20% loss)
MAX_GAIN_PERCENT = 5  # Maximum gain as percentage (e.g., 50 = 50% gain)

# Transaction settings ⚡
slippage = 199  # 500 = 5% and 50 = .5% slippage
PRIORITY_FEE = 100000  # ~0.02 USD at current SOL prices
orders_per_open = 3  # Multiple orders for better fill rates

# Market maker settings 📊
buy_under = .0946
sell_over = 1

# Data collection settings 📈
DAYSBACK_4_DATA = 3
DATA_TIMEFRAME = '1H'  # 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 8H, 12H, 1D, 3D, 1W, 1M
SAVE_OHLCV_DATA = False  # Set to True to save data permanently, False will only use temp data during run

# AI Model Settings 🤖
AI_MODEL = "claude-3-haiku-20240307"  # Model Options:
                                      # - claude-3-haiku-20240307 (Fast, efficient Claude model)
                                      # - claude-3-sonnet-20240229 (Balanced Claude model)
                                      # - claude-3-opus-20240229 (Most powerful Claude model)
AI_MAX_TOKENS = 1024  # Max tokens for response
AI_TEMPERATURE = 0.7  # Creativity vs precision (0-1)

# Trading Strategy Agent Settings
ENABLE_STRATEGIES = True  # Set this to True to use strategies
STRATEGY_MIN_CONFIDENCE = 0.7  # Minimum confidence to act on strategy signals

# Sleep time between main agent runs
SLEEP_BETWEEN_RUNS_MINUTES = 15  # How long to sleep between agent runs 🕒

# Minimum trades last hour for token validation
MIN_TRADES_LAST_HOUR = 2

# NeuroFlux Specific Settings 🧠
# Flux monitoring thresholds
FLUX_SENSITIVITY = 0.8  # Sensitivity for flux detection (0-1)
ADAPTIVE_LEARNING_RATE = 0.1  # Learning rate for neuro-adaptation
NEURAL_NETWORK_LAYERS = [64, 32, 16]  # Neural network architecture for predictions
SWARM_SIZE = 6  # Number of agents in swarm consensus

# ============================================================================
# ACTIVE AGENTS CONFIGURATION
# ============================================================================

# Master switch for all agents
AGENTS_ENABLED = True

# Individual agent enable/disable switches
ACTIVE_AGENTS = {
    # Core Trading Agents
    'trading_agent': True,        # Main trading execution
    'risk_agent': True,          # Risk management and position monitoring
    'strategy_agent': True,      # Strategy-based trading signals

    # Market Analysis Agents
    'sentiment_agent': True,     # Market sentiment analysis
    'research_agent': True,      # Fundamental research and news
    'chartanalysis_agent': True, # Technical analysis
    'coingecko_agent': True,     # CoinGecko data and metrics

    # Specialized Agents
    'funding_agent': True,       # Funding rate analysis
    'liquidation_agent': True,   # Liquidation monitoring
    'whale_agent': True,         # Large wallet tracking
    'websearch_agent': True,     # Web search and data gathering

    # Advanced Agents
    'copybot_agent': False,      # Copy trading (disabled by default)
    'sniper_agent': True,        # New launch sniping
    'swarm_agent': True,         # Swarm intelligence consensus
    'rbi_agent': True,           # Research-Backed Intelligence

    # Backtesting
    'backtest_runner': False,    # Automated backtesting (run manually)
}

# Agent execution priorities (higher = runs first)
AGENT_PRIORITIES = {
    'risk_agent': 10,           # Always run risk checks first
    'sentiment_agent': 8,       # Market sentiment affects all decisions
    'research_agent': 7,        # Fundamental analysis
    'chartanalysis_agent': 6,   # Technical analysis
    'coingecko_agent': 5,       # Market data
    'funding_agent': 4,         # Funding rates
    'liquidation_agent': 3,     # Liquidation risks
    'whale_agent': 2,           # Whale movements
    'strategy_agent': 1,        # Trading strategies
    'swarm_agent': 1,           # Swarm consensus
    'rbi_agent': 1,             # RBI analysis
    'sniper_agent': 0,          # New launches (lower priority)
    'websearch_agent': 0,       # Background research
    'copybot_agent': 0,         # Copy trading
    'trading_agent': -1,        # Execute trades last
    'backtest_runner': -10,     # Manual only
}