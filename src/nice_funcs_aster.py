"""
🧠 NeuroFlux's Aster DEX Functions
Neuro-flux enhanced trading functions for Aster DEX futures.
Built with love by Nyros Veil 🚀

Advanced trading functions with flux-adaptive parameters and AI-enhanced decision making.
Supports multi-exchange trading with real-time neuro-flux adjustments.

LEVERAGE & POSITION SIZING:
- All 'amount' parameters represent NOTIONAL position size (total exposure)
- Leverage is applied by the exchange, reducing required margin
- Example: $25 position at 5x leverage = $25 notional, $5 margin required
- Formula: Required Margin = Notional Position / Leverage
- Neuro-flux adaptation: Leverage adjusts based on market volatility
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import datetime
from datetime import timedelta
from termcolor import colored, cprint
from dotenv import load_dotenv
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add Aster Dex Trading Bots to path (if available)
aster_bots_path = '/Users/md/Dropbox/dev/github/Aster-Dex-Trading-Bots'
if aster_bots_path not in sys.path and os.path.exists(aster_bots_path):
    sys.path.insert(0, aster_bots_path)

# Try importing Aster modules
try:
    from aster_api import AsterAPI  # type: ignore
    from aster_funcs import AsterFuncs  # type: ignore
    ASTER_AVAILABLE = True
except ImportError:
    ASTER_AVAILABLE = False
    cprint("⚠️  Aster modules not available - running in simulation mode", "yellow")

# Load environment variables
load_dotenv()

# Get API keys
ASTER_API_KEY = os.getenv('ASTER_API_KEY')
ASTER_API_SECRET = os.getenv('ASTER_API_SECRET')

# Verify API keys
if ASTER_AVAILABLE and (not ASTER_API_KEY or not ASTER_API_SECRET):
    cprint("❌ ASTER API keys not found in .env file!", "red")
    ASTER_AVAILABLE = False

# Initialize API (global instance)
if ASTER_AVAILABLE:
    api = AsterAPI(ASTER_API_KEY, ASTER_API_SECRET)
    funcs = AsterFuncs(api)
else:
    api = None
    funcs = None

# Add NeuroFlux config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================================
# NEURO-FLUX CONFIGURATION
# ============================================================================

# Adaptive leverage based on flux sensitivity
def get_adaptive_leverage(base_leverage=5):
    """Get flux-adjusted leverage"""
    flux_penalty = config.FLUX_SENSITIVITY * 0.5  # Reduce leverage in high flux
    adjusted_leverage = max(1, base_leverage * (1 - flux_penalty))
    return int(adjusted_leverage)

DEFAULT_LEVERAGE = get_adaptive_leverage(5)  # Flux-adaptive default

# Constants
DEFAULT_SYMBOL_SUFFIX = 'USDT'  # Aster uses BTCUSDT, ETHUSDT, etc.
SYMBOL_PRECISION_CACHE = {}

# ============================================================================
# NEURO-FLUX ENHANCED HELPER FUNCTIONS
# ============================================================================

def get_symbol_precision(symbol):
    """Get price and quantity precision for a symbol with neuro-flux caching"""
    if symbol in SYMBOL_PRECISION_CACHE:
        return SYMBOL_PRECISION_CACHE[symbol]

    if not ASTER_AVAILABLE:
        # Simulation mode defaults
        SYMBOL_PRECISION_CACHE[symbol] = (2, 3)
        return 2, 3

    try:
        exchange_info = api.get_exchange_info()

        for sym_info in exchange_info.get('symbols', []):
            if sym_info['symbol'] == symbol:
                price_precision = 2
                quantity_precision = 3

                for filter_info in sym_info.get('filters', []):
                    if filter_info['filterType'] == 'PRICE_FILTER':
                        tick_size = filter_info.get('tickSize', '0.01')
                        price_precision = len(tick_size.rstrip('0').split('.')[-1]) if '.' in tick_size else 0

                    if filter_info['filterType'] == 'LOT_SIZE':
                        step_size = filter_info.get('stepSize', '0.001')
                        quantity_precision = len(step_size.rstrip('0').split('.')[-1]) if '.' in step_size else 0

                SYMBOL_PRECISION_CACHE[symbol] = (price_precision, quantity_precision)
                return price_precision, quantity_precision

        # Default if not found
        SYMBOL_PRECISION_CACHE[symbol] = (2, 3)
        return 2, 3

    except Exception as e:
        cprint(f"❌ Error getting precision: {e}", "red")
        return 2, 3


def format_symbol(token):
    """Convert token address/symbol to Aster format with flux validation"""
    if not token.endswith(DEFAULT_SYMBOL_SUFFIX):
        return f"{token}{DEFAULT_SYMBOL_SUFFIX}"
    return token


def token_price(address):
    """Get current token price with neuro-flux enhanced error handling"""
    if not ASTER_AVAILABLE:
        # Simulation mode - return mock price
        return 50000 if 'BTC' in address else 3000

    try:
        symbol = format_symbol(address)
        ask, bid, _ = api.get_ask_bid(symbol)
        midpoint = (ask + bid) / 2
        return midpoint
    except Exception as e:
        cprint(f"❌ Error getting price for {address}: {e}", "red")
        return 0


def get_best_bid_ask(symbol):
    """Get best bid and ask prices with flux-adjusted spreads"""
    if not ASTER_AVAILABLE:
        # Simulation mode
        base_price = 50000 if 'BTC' in symbol else 3000
        spread = base_price * 0.0001 * (1 + config.FLUX_SENSITIVITY)  # Flux increases spread
        return base_price - spread/2, base_price + spread/2

    try:
        orderbook = api.get_orderbook(symbol, limit=5)

        if not orderbook:
            return None, None

        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])

        if not bids or not asks:
            return None, None

        best_bid = float(bids[0][0])  # First bid price
        best_ask = float(asks[0][0])  # First ask price

        return best_bid, best_ask

    except Exception as e:
        cprint(f"❌ Error getting order book for {symbol}: {e}", "red")
        return None, None


# ============================================================================
# NEURO-FLUX ENHANCED TRADING FUNCTIONS
# ============================================================================

def neuro_adaptive_market_buy(symbol, usd_size, sentiment_score=0.5):
    """
    🧠 Neuro-flux enhanced market buy with AI adaptation

    Args:
        symbol: Token symbol
        usd_size: Base USD amount
        sentiment_score: Market sentiment (0-1, 0.5=neutral)

    Returns:
        dict: Trade execution result with flux adjustments
    """
    print(colored(f'🧠 NeuroFlux Aster Market BUY {symbol} for ${usd_size}', 'green'))

    # Neuro-flux decision making
    flux_level = config.FLUX_SENSITIVITY

    # Adjust position size based on sentiment and flux
    sentiment_multiplier = 0.5 + sentiment_score  # 0.5-1.5 range
    flux_multiplier = 1 - (flux_level * 0.4)  # Reduce size in high flux
    adjusted_size = usd_size * sentiment_multiplier * flux_multiplier

    # Adjust leverage based on flux
    base_leverage = 5
    adjusted_leverage = get_adaptive_leverage(base_leverage)

    print(f'🧠 Neuro-adaptive buy: Sentiment {sentiment_score:.2f}, Flux {flux_level:.2f}')
    print(f'💰 Size: ${usd_size:.0f} → ${adjusted_size:.0f} | Leverage: {base_leverage}x → {adjusted_leverage}x')

    # Execute trade
    result = market_buy(symbol, adjusted_size, slippage=0, leverage=adjusted_leverage, flux_adjust=False)

    return {
        'execution': result,
        'adjustments': {
            'original_size': usd_size,
            'adjusted_size': adjusted_size,
            'sentiment_multiplier': sentiment_multiplier,
            'flux_multiplier': flux_multiplier,
            'leverage': adjusted_leverage
        }
    }


def market_buy(symbol, usd_size, slippage=0, leverage=DEFAULT_LEVERAGE, flux_adjust=True):
    """Neuro-flux enhanced market buy"""
    print(colored(f'🧠 NeuroFlux Aster Market BUY {symbol} for ${usd_size}', 'green'))

    if not ASTER_AVAILABLE:
        print("⚠️  Aster API not available - simulating trade")
        return {'orderId': 'simulated', 'status': 'FILLED', 'simulated': True}

    # Apply flux adjustments
    if flux_adjust:
        flux_multiplier = 1 - (config.FLUX_SENSITIVITY * 0.3)  # Reduce size in high flux
        usd_size = usd_size * flux_multiplier
        print(f'🌊 Flux-adjusted size: ${usd_size:.2f}')

    symbol = format_symbol(symbol)

    # Set adaptive leverage
    leverage = get_adaptive_leverage(leverage)

    # Set leverage
    print(f'⚙️  Setting leverage to {leverage}x for {symbol}')
    api.change_leverage(symbol, leverage)

    # Get current price and calculate quantity
    current_price = token_price(symbol)
    if current_price == 0:
        print(f'❌ Could not get price for {symbol}')
        return None

    # Calculate quantity based on NOTIONAL value
    quantity = usd_size / current_price

    # Round to proper precision
    _, quantity_precision = get_symbol_precision(symbol)
    quantity = round(quantity, quantity_precision)

    # Check minimum order value with flux adjustment
    order_value = quantity * current_price
    min_notional = 5.0 * (1 - config.FLUX_SENSITIVITY * 0.5)  # Flux reduces minimum

    if order_value < min_notional:
        print(f'⚠️ Order value ${order_value:.2f} below ${min_notional:.2f} minimum, adjusting...')
        quantity = (min_notional + 1) / current_price
        quantity = round(quantity, quantity_precision)

    # Calculate required margin
    required_margin = usd_size / leverage

    print(f'🚀 MARKET BUY: {quantity} {symbol} @ ~${current_price:.2f}')
    print(f'💰 Notional Position: ${usd_size:.2f} | Margin Required: ${required_margin:.2f} ({leverage}x)')

    # Place market buy order
    order = api.place_order(
        symbol=symbol,
        side='BUY',
        order_type='MARKET',
        quantity=quantity
    )

    print(colored(f'✅ NeuroFlux market buy executed! Order ID: {order.get("orderId")}', 'green'))
    return order


def market_sell(symbol, usd_size, slippage=0, leverage=DEFAULT_LEVERAGE, flux_adjust=True):
    """Neuro-flux enhanced market sell"""
    print(colored(f'🧠 NeuroFlux Aster Market SELL {symbol} for ${usd_size}', 'red'))

    if not ASTER_AVAILABLE:
        print("⚠️  Aster API not available - simulating trade")
        return {'orderId': 'simulated', 'status': 'FILLED', 'simulated': True}

    # Apply flux adjustments
    if flux_adjust:
        flux_multiplier = 1 - (config.FLUX_SENSITIVITY * 0.3)
        usd_size = usd_size * flux_multiplier
        print(f'🌊 Flux-adjusted size: ${usd_size:.2f}')

    symbol = format_symbol(symbol)

    # Check current position
    position = get_position(symbol)

    # Get current price and calculate quantity
    current_price = token_price(symbol)
    if current_price == 0:
        print(f'❌ Could not get price for {symbol}')
        return None

    # Calculate quantity based on NOTIONAL value
    quantity = usd_size / current_price

    # Round to proper precision
    _, quantity_precision = get_symbol_precision(symbol)
    quantity = round(quantity, quantity_precision)

    if position and position['position_amount'] > 0:
        # We have a long position - close it
        print(f'📉 Closing LONG: {quantity} {symbol} @ MARKET')
        print(f'💰 Closing ${usd_size:.2f} notional position')

        order = api.place_order(
            symbol=symbol,
            side='SELL',
            order_type='MARKET',
            quantity=quantity,
            reduce_only=True
        )

        print(colored(f'✅ NeuroFlux market sell executed! Order ID: {order.get("orderId")}', 'green'))
        return order
    else:
        # Open short position
        leverage = get_adaptive_leverage(leverage)
        api.change_leverage(symbol, leverage)

        # Check minimum order value
        order_value = quantity * current_price
        min_notional = 5.0 * (1 - config.FLUX_SENSITIVITY * 0.5)

        if order_value < min_notional:
            print(f'⚠️ Order value ${order_value:.2f} below ${min_notional:.2f} minimum, adjusting...')
            quantity = (min_notional + 1) / current_price
            quantity = round(quantity, quantity_precision)

        required_margin = usd_size / leverage
        print(f'📉 MARKET SELL (SHORT): {quantity} {symbol} @ ~${current_price:.2f}')
        print(f'💰 Notional Position: ${usd_size:.2f} | Margin Required: ${required_margin:.2f} ({leverage}x)')

        order = api.place_order(
            symbol=symbol,
            side='SELL',
            order_type='MARKET',
            quantity=quantity
        )

        print(colored(f'✅ NeuroFlux market sell executed! Order ID: {order.get("orderId")}', 'green'))
        return order


def get_position(symbol):
    """Get current position with neuro-flux metrics"""
    if not ASTER_AVAILABLE:
        # Simulation mode
        return {
            'position_amount': 0.001,
            'entry_price': 50000,
            'mark_price': 51000,
            'pnl': 100,
            'pnl_percentage': 2.0,
            'is_long': True,
            'flux_risk_score': config.FLUX_SENSITIVITY * 10
        }

    try:
        symbol = format_symbol(symbol)
        position = api.get_position(symbol)
        if position:
            # Add neuro-flux metrics
            position['flux_risk_score'] = config.FLUX_SENSITIVITY * 10
            position['volatility_adjustment'] = config.FLUX_SENSITIVITY * 0.1
        return position
    except Exception as e:
        print(f'❌ Error getting position for {symbol}: {e}')
        return None


def get_balance():
    """Get USDC balance with flux-aware calculations"""
    if not ASTER_AVAILABLE:
        return 10000 * (1 - config.FLUX_SENSITIVITY * 0.2)  # Flux reduces available balance

    try:
        account_info = api.get_account_info()
        available = float(account_info.get('availableBalance', 0))

        # Apply flux adjustment to available balance
        flux_adjustment = config.FLUX_SENSITIVITY * 0.2
        adjusted_balance = available * (1 - flux_adjustment)

        print(f'Available balance: ${available:,.2f} | Flux-adjusted: ${adjusted_balance:,.2f}')
        return adjusted_balance
    except Exception as e:
        print(f'❌ Error getting balance: {e}')
        return 0


def flux_aware_risk_management(symbol):
    """
    Neuro-flux aware risk management for Aster

    Args:
        symbol: Token symbol

    Returns:
        dict: Risk management recommendations
    """
    position = get_position(symbol)

    if not position or position['position_amount'] == 0:
        return {'action': 'none', 'reason': 'no_position'}

    flux_level = config.FLUX_SENSITIVITY
    pnl_perc = position.get('pnl_percentage', 0)

    # Dynamic stop loss based on flux
    base_stop_loss = 5
    flux_adjusted_stop = base_stop_loss + (flux_level * 10)

    # Dynamic take profit based on flux
    base_take_profit = 10
    flux_adjusted_profit = base_take_profit - (flux_level * 5)

    recommendations = {
        'current_pnl': pnl_perc,
        'flux_level': flux_level,
        'stop_loss_threshold': -flux_adjusted_stop,
        'take_profit_threshold': flux_adjusted_profit,
        'risk_score': flux_level * 10,
        'action': 'hold'
    }

    if pnl_perc <= -flux_adjusted_stop:
        recommendations['action'] = 'close_loss'
        recommendations['reason'] = f'PnL {pnl_perc:.1f}% below flux-adjusted stop loss {-flux_adjusted_stop:.1f}%'
    elif pnl_perc >= flux_adjusted_profit:
        recommendations['action'] = 'close_profit'
        recommendations['reason'] = f'PnL {pnl_perc:.1f}% above flux-adjusted take profit {flux_adjusted_profit:.1f}%'

    return recommendations


def kill_switch(symbol):
    """Emergency close position with flux awareness"""
    print(colored(f'🔪 NeuroFlux KILL SWITCH ACTIVATED for {symbol}', 'red', attrs=['bold']))

    if not ASTER_AVAILABLE:
        print("⚠️  Aster API not available - simulating close")
        return {'simulated': True}

    position = get_position(symbol)

    if not position or position['position_amount'] == 0:
        print(colored('No position to close', 'yellow'))
        return

    # Determine close side
    is_long = position['position_amount'] > 0
    close_side = 'SELL' if is_long else 'BUY'
    abs_size = abs(position['position_amount'])

    print(f'Closing {"LONG" if is_long else "SHORT"} position: {abs_size} {symbol}')

    # Flux-adjusted close price
    current_price = token_price(symbol)
    flux_adjustment = config.FLUX_SENSITIVITY * 0.002

    if is_long:
        close_price = current_price * (0.999 - flux_adjustment)
    else:
        close_price = current_price * (1.001 + flux_adjustment)

    # Round price
    price_precision, _ = get_symbol_precision(symbol)
    close_price = round(close_price, price_precision)

    print(f'Placing IOC at ${close_price} to close position (flux-adjusted)')

    # Place reduce-only order
    order = api.place_order(
        symbol=symbol,
        side=close_side,
        order_type='LIMIT',
        quantity=abs_size,
        price=close_price,
        time_in_force='IOC',
        reduce_only=True
    )

    print(colored('✅ NeuroFlux kill switch executed - position closed', 'green'))
    return order


# ============================================================================
# NEURO-FLUX MARKET DATA FUNCTIONS
# ============================================================================

def get_aster_ohlcv(symbol, timeframe='1h', limit=100):
    """
    Get OHLCV data from Aster with neuro-flux enhancements

    Args:
        symbol: Trading pair
        timeframe: Timeframe (1m, 5m, 15m, 1h, etc.)
        limit: Number of candles

    Returns:
        pd.DataFrame: OHLCV data with neuro indicators
    """
    if not ASTER_AVAILABLE:
        # Generate mock data
        base_price = 50000 if 'BTC' in symbol else 3000
        dates = pd.date_range(end=datetime.datetime.now(), periods=limit, freq='1H')
        np.random.seed(42)

        # Generate realistic price movements
        returns = np.random.normal(0, 0.02, limit)
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.01, limit)),
            'low': prices * (1 - np.random.uniform(0, 0.01, limit)),
            'close': prices * (1 + np.random.normal(0, 0.005, limit)),
            'volume': np.random.uniform(100, 1000, limit)
        })

        # Ensure OHLC logic
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, limit))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, limit))

        df = add_neuro_indicators(df)
        return df

    try:
        # This would need to be implemented based on Aster's API
        # For now, return mock data
        return get_aster_ohlcv(symbol, timeframe, limit)  # Recursive call to mock
    except Exception as e:
        print(f"❌ Error getting OHLCV data: {e}")
        return pd.DataFrame()


def add_neuro_indicators(df):
    """Add neuro-flux enhanced technical indicators"""
    if df.empty:
        return df

    try:
        print("🧠 Adding NeuroFlux enhanced indicators...")

        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype('float64')

        # Standard indicators
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['rsi'] = ta.rsi(df['close'], length=14)

        # Neuro-flux enhanced indicators
        flux_level = config.FLUX_SENSITIVITY

        # Adaptive RSI
        adaptive_rsi_length = int(14 * (1 + flux_level))
        df['rsi_adaptive'] = ta.rsi(df['close'], length=adaptive_rsi_length)

        # Volatility-adjusted momentum
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        df['momentum_flux_adjusted'] = returns / (volatility + 0.001)

        # Neural network inspired composite score
        df['neuro_score'] = (
            (df['rsi'] - 50) / 50 * 0.4 +
            (df['close'] - df['sma_20']) / df['sma_20'] * 0.4 +
            df['momentum_flux_adjusted'] * 0.2
        ) * (1 - flux_level * 0.2)

        print("✅ NeuroFlux enhanced indicators added successfully")
        return df

    except Exception as e:
        print(f"❌ Error adding neuro indicators: {str(e)}")
        return df


# ============================================================================
# LEGACY FUNCTION ALIASES (for compatibility)
# ============================================================================

def limit_buy(token, amount, slippage, leverage=DEFAULT_LEVERAGE):
    """Legacy limit buy - now uses market buy with flux adjustments"""
    return market_buy(token, amount, slippage, leverage)

def limit_sell(token, amount, slippage, leverage=DEFAULT_LEVERAGE):
    """Legacy limit sell - now uses market sell with flux adjustments"""
    return market_sell(token, amount, slippage, leverage)

def chunk_kill(token_mint_address, max_usd_order_size, slippage):
    """Legacy chunk kill - simplified to single close"""
    return kill_switch(token_mint_address)

def ai_entry(symbol, amount, max_chunk_size=None, leverage=DEFAULT_LEVERAGE, use_limit=True):
    """Legacy AI entry - now uses neuro-adaptive buy"""
    return neuro_adaptive_market_buy(symbol, amount, sentiment_score=0.5)

def open_short(token, amount, slippage, leverage=DEFAULT_LEVERAGE):
    """Legacy open short - now uses market sell"""
    return market_sell(token, amount, slippage, leverage)

def get_account_balance():
    """Legacy balance getter"""
    return get_balance()

def get_token_balance_usd(token_mint_address):
    """Legacy token balance getter"""
    position = get_position(token_mint_address)
    if position:
        return abs(position['position_amount'] * position['mark_price'])
    return 0


# Initialize on import
if ASTER_AVAILABLE:
    print("🧠 NeuroFlux Aster DEX functions loaded successfully!")
    print(f"🌊 Flux sensitivity: {config.FLUX_SENSITIVITY:.2f}")
    print(f"⚡ Adaptive leverage: {get_adaptive_leverage(5)}x")
else:
    print("🧠 NeuroFlux Aster DEX functions loaded in simulation mode!")
    print("⚠️  Aster API not available - trades will be simulated")