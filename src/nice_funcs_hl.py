"""
🧠 NeuroFlux's HyperLiquid Trading Functions
Neuro-flux enhanced trading functions for HyperLiquid perps.

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
import json
import time
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import datetime
from datetime import timedelta
from termcolor import colored, cprint
from eth_account.signers.local import LocalAccount
import eth_account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from dotenv import load_dotenv
import traceback
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Load environment variables
load_dotenv()

# Hide all warnings
import warnings
warnings.filterwarnings('ignore')

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
BATCH_SIZE = 5000  # MAX IS 5000 FOR HYPERLIQUID
MAX_RETRIES = 3
MAX_ROWS = 5000
BASE_URL = 'https://api.hyperliquid.xyz/info'

# Global variable to store timestamp offset
timestamp_offset = None

def adjust_timestamp(dt):
    """Adjust API timestamps by subtracting the timestamp offset."""
    global timestamp_offset
    if timestamp_offset is not None:
        corrected_dt = dt - timestamp_offset
        return corrected_dt
    return dt

def ask_bid(symbol):
    """Get ask and bid prices for a symbol"""
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}

    data = {
        'type': 'l2Book',
        'coin': symbol
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()
    l2_data = l2_data['levels']

    # get bid and ask
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])

    return ask, bid, l2_data

def get_sz_px_decimals(symbol):
    """Get size and price decimals for a symbol"""
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        data = response.json()
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == symbol), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']
        else:
            print('Symbol not found')
            return 0, 0
    else:
        print('Error:', response.status_code)
        return 0, 0

    ask = ask_bid(symbol)[0]
    ask_str = str(ask)

    if '.' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0

    print(f'{symbol} price: {ask} | sz decimals: {sz_decimals} | px decimals: {px_decimals}')
    return sz_decimals, px_decimals

def get_position(symbol, account):
    """Get current position for a symbol with neuro-flux awareness"""
    print(f'{colored("Getting position for", "cyan")} {colored(symbol, "yellow")}')

    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    positions = []
    for position in user_state["assetPositions"]:
        if position["position"]["coin"] == symbol and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            coin = position["position"]["coin"]
            pos_size = float(position["position"]["szi"])
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"]) * 100
            print(f'{colored(f"{coin} position:", "green")} Size: {pos_size} | Entry: ${entry_px} | PnL: {pnl_perc:.2f}%')

    im_in_pos = len(positions) > 0

    if not im_in_pos:
        print(f'{colored("No position in", "yellow")} {symbol}')
        return positions, im_in_pos, 0, symbol, 0, 0, True

    # Return position details
    pos_size = positions[0]["szi"]
    pos_sym = positions[0]["coin"]
    entry_px = float(positions[0]["entryPx"])
    pnl_perc = float(positions[0]["returnOnEquity"]) * 100
    is_long = float(pos_size) > 0

    if is_long:
        print(f'{colored("LONG", "green")} position')
    else:
        print(f'{colored("SHORT", "red")} position')

    return positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, is_long

def set_leverage(symbol, leverage, account):
    """Set leverage for a symbol with flux validation"""
    # Validate leverage against flux conditions
    max_leverage = 50  # HyperLiquid max
    flux_adjusted_max = max_leverage * (1 - config.FLUX_SENSITIVITY)
    leverage = min(leverage, flux_adjusted_max)

    print(f'Setting leverage for {symbol} to {leverage}x (flux-adjusted)')
    exchange = Exchange(account, constants.MAINNET_API_URL)

    # Update leverage (is_cross=True for cross margin)
    result = exchange.update_leverage(leverage, symbol, is_cross=True)
    print(f'✅ Leverage set to {leverage}x for {symbol}')
    return result

def get_current_price(symbol):
    """Get current price for a symbol"""
    ask, bid, _ = ask_bid(symbol)
    mid_price = (ask + bid) / 2
    return mid_price

def get_account_value(account):
    """Get total account value"""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    account_value = float(user_state["marginSummary"]["accountValue"])
    print(f'Account value: ${account_value:,.2f}')
    return account_value

def market_buy(symbol, usd_size, account, flux_adjust=True):
    """Neuro-flux enhanced market buy"""
    print(colored(f'🧠 NeuroFlux Market BUY {symbol} for ${usd_size}', 'green'))

    # Apply flux adjustments
    if flux_adjust:
        flux_multiplier = 1 - (config.FLUX_SENSITIVITY * 0.3)  # Reduce size in high flux
        usd_size = usd_size * flux_multiplier
        print(f'🌊 Flux-adjusted size: ${usd_size:.2f}')

    # Get current ask price
    ask, bid, _ = ask_bid(symbol)

    # Overbid by 0.1% to ensure fill (market buy needs to be above ask)
    buy_price = ask * 1.001

    # Round to appropriate decimals for BTC (whole numbers)
    if symbol == 'BTC':
        buy_price = round(buy_price)
    else:
        buy_price = round(buy_price, 1)

    # Calculate position size
    pos_size = usd_size / buy_price

    # Get decimals and round
    sz_decimals, _ = get_sz_px_decimals(symbol)
    pos_size = round(pos_size, sz_decimals)

    # Ensure minimum order value
    order_value = pos_size * buy_price
    if order_value < 10:
        print(f'   ⚠️ Order value ${order_value:.2f} below $10 minimum, adjusting...')
        pos_size = 11 / buy_price  # $11 to have buffer
        pos_size = round(pos_size, sz_decimals)

    print(f'   Placing IOC buy at ${buy_price} (0.1% above ask ${ask})')
    print(f'   Position size: {pos_size} {symbol} (value: ${pos_size * buy_price:.2f})')

    # Place IOC order above ask to ensure fill
    exchange = Exchange(account, constants.MAINNET_API_URL)
    order_result = exchange.order(symbol, True, pos_size, buy_price, {"limit": {"tif": "Ioc"}}, reduce_only=False)

    print(colored(f'✅ NeuroFlux market buy executed: {pos_size} {symbol} at ${buy_price}', 'green'))
    return order_result

def market_sell(symbol, usd_size, account, flux_adjust=True):
    """Neuro-flux enhanced market sell"""
    print(colored(f'🧠 NeuroFlux Market SELL {symbol} for ${usd_size}', 'red'))

    # Apply flux adjustments
    if flux_adjust:
        flux_multiplier = 1 - (config.FLUX_SENSITIVITY * 0.3)  # Reduce size in high flux
        usd_size = usd_size * flux_multiplier
        print(f'🌊 Flux-adjusted size: ${usd_size:.2f}')

    # Get current bid price
    ask, bid, _ = ask_bid(symbol)

    # Undersell by 0.1% to ensure fill (market sell needs to be below bid)
    sell_price = bid * 0.999

    # Round to appropriate decimals for BTC (whole numbers)
    if symbol == 'BTC':
        sell_price = round(sell_price)
    else:
        sell_price = round(sell_price, 1)

    # Calculate position size
    pos_size = usd_size / sell_price

    # Get decimals and round
    sz_decimals, _ = get_sz_px_decimals(symbol)
    pos_size = round(pos_size, sz_decimals)

    # Ensure minimum order value
    order_value = pos_size * sell_price
    if order_value < 10:
        print(f'   ⚠️ Order value ${order_value:.2f} below $10 minimum, adjusting...')
        pos_size = 11 / sell_price  # $11 to have buffer
        pos_size = round(pos_size, sz_decimals)

    print(f'   Placing IOC sell at ${sell_price} (0.1% below bid ${bid})')
    print(f'   Position size: {pos_size} {symbol} (value: ${pos_size * sell_price:.2f})')

    # Place IOC order below bid to ensure fill
    exchange = Exchange(account, constants.MAINNET_API_URL)
    order_result = exchange.order(symbol, False, pos_size, sell_price, {"limit": {"tif": "Ioc"}}, reduce_only=False)

    print(colored(f'✅ NeuroFlux market sell executed: {pos_size} {symbol} at ${sell_price}', 'red'))
    return order_result

def close_position(symbol, account):
    """Close any open position for a symbol with flux-aware execution"""
    positions, im_in_pos, pos_size, _, _, pnl_perc, is_long = get_position(symbol, account)

    if not im_in_pos:
        print(f'No position to close for {symbol}')
        return None

    print(f'Closing {"LONG" if is_long else "SHORT"} position with PnL: {pnl_perc:.2f}%')

    # Flux-adjusted close: be more aggressive in high flux to reduce risk
    flux_multiplier = 1 + (config.FLUX_SENSITIVITY * 0.5)
    return kill_switch(symbol, account)

def kill_switch(symbol, account):
    """Emergency close position at market price with flux awareness"""
    print(colored(f'🔪 NeuroFlux KILL SWITCH ACTIVATED for {symbol}', 'red', attrs=['bold']))

    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    exchange = Exchange(account, constants.MAINNET_API_URL)

    # Get current position
    positions, im_in_pos, pos_size, _, _, _, is_long = get_position(symbol, account)

    if not im_in_pos:
        print(colored('No position to close', 'yellow'))
        return

    # Place market order to close
    side = not is_long  # Opposite side to close
    abs_size = abs(float(pos_size))

    print(f'Closing {"LONG" if is_long else "SHORT"} position: {abs_size} {symbol}')

    # Get current price for market order
    ask, bid, _ = ask_bid(symbol)

    # For closing positions with IOC orders:
    # - Closing long: Sell below bid (undersell)
    # - Closing short: Buy above ask (overbid)
    # Flux adjustment: be more aggressive in high flux
    flux_adjustment = config.FLUX_SENSITIVITY * 0.002  # 0.2% max adjustment

    if is_long:
        close_price = bid * (0.999 - flux_adjustment)  # More aggressive undersell in high flux
    else:
        close_price = ask * (1.001 + flux_adjustment)  # More aggressive overbid in high flux

    # Round to appropriate decimals for BTC
    if symbol == 'BTC':
        close_price = round(close_price)
    else:
        close_price = round(close_price, 1)

    print(f'   Placing IOC at ${close_price} to close position (flux-adjusted)')

    # Place reduce-only order to close
    order_result = exchange.order(symbol, side, abs_size, close_price, {"limit": {"tif": "Ioc"}}, reduce_only=True)

    print(colored('✅ NeuroFlux kill switch executed - position closed', 'green'))
    return order_result

def get_balance(account):
    """Get USDC balance with flux-aware calculations"""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    # Get withdrawable balance (free balance)
    balance = float(user_state["withdrawable"])

    # Apply flux adjustment to available balance (be more conservative in high flux)
    flux_adjustment = config.FLUX_SENSITIVITY * 0.2
    adjusted_balance = balance * (1 - flux_adjustment)

    print(f'Available balance: ${balance:,.2f} | Flux-adjusted: ${adjusted_balance:,.2f}')
    return adjusted_balance

def get_all_positions(account):
    """Get all open positions with neuro-flux metrics"""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    positions = []
    for position in user_state["assetPositions"]:
        if float(position["position"]["szi"]) != 0:
            pos_data = {
                'symbol': position["position"]["coin"],
                'size': float(position["position"]["szi"]),
                'entry_price': float(position["position"]["entryPx"]),
                'pnl_percent': float(position["position"]["returnOnEquity"]) * 100,
                'is_long': float(position["position"]["szi"]) > 0,
                'unrealized_pnl': float(position["position"]["unrealizedPnl"]),
                # Neuro-flux metrics
                'flux_risk_score': config.FLUX_SENSITIVITY * 10,  # 0-10 scale
                'volatility_adjustment': config.FLUX_SENSITIVITY * 0.1
            }
            positions.append(pos_data)

    return positions

# ============================================================================
# NEURO-FLUX ENHANCED TRADING FUNCTIONS
# ============================================================================

def neuro_adaptive_entry(symbol, usd_size, sentiment_score=0.5, account=None):
    """
    AI-enhanced entry with neuro-flux adaptation

    Args:
        symbol: Token symbol
        usd_size: Base USD amount
        sentiment_score: Market sentiment (0-1, 0.5=neutral)
        account: HyperLiquid account

    Returns:
        dict: Trade execution result with flux adjustments
    """
    if account is None:
        account = _get_account_from_env()

    # Neuro-flux decision making
    flux_level = config.FLUX_SENSITIVITY

    # Adjust position size based on sentiment and flux
    sentiment_multiplier = 0.5 + sentiment_score  # 0.5-1.5 range
    flux_multiplier = 1 - (flux_level * 0.4)  # Reduce size in high flux
    adjusted_size = usd_size * sentiment_multiplier * flux_multiplier

    # Adjust leverage based on flux
    base_leverage = 5
    adjusted_leverage = get_adaptive_leverage(base_leverage)

    print(f'🧠 Neuro-adaptive entry: Sentiment {sentiment_score:.2f}, Flux {flux_level:.2f}')
    print(f'💰 Size: ${usd_size:.0f} → ${adjusted_size:.0f} | Leverage: {base_leverage}x → {adjusted_leverage}x')

    # Set adaptive leverage
    set_leverage(symbol, adjusted_leverage, account)

    # Execute trade
    result = market_buy(symbol, adjusted_size, account, flux_adjust=False)  # Size already adjusted

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

def flux_aware_risk_management(symbol, account):
    """
    Neuro-flux aware risk management

    Args:
        symbol: Token symbol
        account: HyperLiquid account

    Returns:
        dict: Risk management recommendations
    """
    positions, im_in_pos, pos_size, _, entry_px, pnl_perc, is_long = get_position(symbol, account)

    if not im_in_pos:
        return {'action': 'none', 'reason': 'no_position'}

    flux_level = config.FLUX_SENSITIVITY

    # Dynamic stop loss based on flux
    base_stop_loss = 5  # 5% base
    flux_adjusted_stop = base_stop_loss + (flux_level * 10)  # Up to 15% in high flux

    # Dynamic take profit based on flux
    base_take_profit = 10  # 10% base
    flux_adjusted_profit = base_take_profit - (flux_level * 5)  # Down to 5% in high flux

    recommendations = {
        'current_pnl': pnl_perc,
        'flux_level': flux_level,
        'stop_loss_threshold': -flux_adjusted_stop,
        'take_profit_threshold': flux_adjusted_profit,
        'risk_score': flux_level * 10,  # 0-10 scale
        'action': 'hold'
    }

    # Decision logic
    if pnl_perc <= -flux_adjusted_stop:
        recommendations['action'] = 'close_loss'
        recommendations['reason'] = f'PnL {pnl_perc:.1f}% below flux-adjusted stop loss {-flux_adjusted_stop:.1f}%'
    elif pnl_perc >= flux_adjusted_profit:
        recommendations['action'] = 'close_profit'
        recommendations['reason'] = f'PnL {pnl_perc:.1f}% above flux-adjusted take profit {flux_adjusted_profit:.1f}%'

    return recommendations

# ============================================================================
# HELPER FUNCTIONS (adapted for NeuroFlux)
# ============================================================================

def _get_exchange():
    """Get exchange instance"""
    private_key = os.getenv('HYPER_LIQUID_ETH_PRIVATE_KEY')
    if not private_key:
        raise ValueError("HYPER_LIQUID_ETH_PRIVATE_KEY not found in .env file")
    account = eth_account.Account.from_key(private_key)
    return Exchange(account, constants.MAINNET_API_URL)

def _get_info():
    """Get info instance"""
    return Info(constants.MAINNET_API_URL, skip_ws=True)

def _get_account_from_env():
    """Initialize and return HyperLiquid account from env"""
    private_key = os.getenv('HYPER_LIQUID_ETH_PRIVATE_KEY')
    if not private_key:
        raise ValueError("HYPER_LIQUID_ETH_PRIVATE_KEY not found in .env file")
    return eth_account.Account.from_key(private_key)

# ============================================================================
# OHLCV DATA FUNCTIONS (enhanced for NeuroFlux)
# ============================================================================

def get_data(symbol, timeframe='15m', bars=100, add_indicators=True):
    """
    🧠 NeuroFlux Enhanced Hyperliquid Data Fetcher

    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC', 'ETH')
        timeframe (str): Candle timeframe (default: '15m')
        bars (int): Number of bars to fetch (default: 100, max: 5000)
        add_indicators (bool): Whether to add technical indicators

    Returns:
        pd.DataFrame: OHLCV data with neuro-flux enhanced indicators
    """
    print("\n🧠 NeuroFlux Enhanced Hyperliquid Data Fetcher")
    print(f"🎯 Symbol: {symbol}")
    print(f"⏰ Timeframe: {timeframe}")
    print(f"📊 Requested bars: {min(bars, MAX_ROWS)}")

    # Ensure we don't exceed max rows
    bars = min(bars, MAX_ROWS)

    # Calculate time window
    end_time = datetime.datetime.utcnow()
    # Add extra time to ensure we get enough bars
    start_time = end_time - timedelta(days=60)

    data = _get_ohlcv(symbol, timeframe, start_time, end_time, batch_size=bars)

    if not data:
        print("❌ No data available.")
        return pd.DataFrame()

    df = _process_data_to_df(data)

    if not df.empty:
        # Get the most recent bars
        df = df.sort_values('timestamp', ascending=False).head(bars).sort_values('timestamp')
        df = df.reset_index(drop=True)

        # Add neuro-flux enhanced technical indicators
        if add_indicators:
            df = add_neuro_indicators(df)

        print("\n📊 NeuroFlux Data summary:")
        print(f"📈 Total candles: {len(df)}")
        print(f"📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print("🧠 Neuro-enhanced indicators added!")

    return df

def add_neuro_indicators(df):
    """Add neuro-flux enhanced technical indicators"""
    if df.empty:
        return df

    try:
        print("\n🧠 Adding NeuroFlux enhanced indicators...")

        # Ensure numeric columns are float64
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype('float64')

        # Standard indicators
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['rsi'] = ta.rsi(df['close'], length=14)

        # MACD
        macd = ta.macd(df['close'])
        df = pd.concat([df, macd], axis=1)

        # Bollinger Bands
        bbands = ta.bbands(df['close'])
        df = pd.concat([df, bbands], axis=1)

        # Neuro-flux enhanced indicators
        flux_level = config.FLUX_SENSITIVITY

        # Adaptive RSI (adjusts sensitivity based on flux)
        adaptive_rsi_length = int(14 * (1 + flux_level))
        df['rsi_adaptive'] = ta.rsi(df['close'], length=adaptive_rsi_length)

        # Flux-adjusted Bollinger Bands
        bb_std_adjusted = 2 * (1 + flux_level * 0.5)  # Increase deviation in high flux
        bbands_adaptive = ta.bbands(df['close'], window_dev=bb_std_adjusted)
        df['bb_upper_adaptive'] = bbands_adaptive['BBU_' + str(int(bb_std_adjusted*10)) + '_2.0']
        df['bb_lower_adaptive'] = bbands_adaptive['BBL_' + str(int(bb_std_adjusted*10)) + '_2.0']

        # Volatility-adjusted momentum
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        df['momentum_flux_adjusted'] = returns / (volatility + 0.001)  # Avoid division by zero

        # Neural network inspired composite score
        df['neuro_score'] = (
            (df['rsi'] - 50) / 50 * 0.3 +  # RSI contribution
            (df['close'] - df['sma_20']) / df['sma_20'] * 0.4 +  # Trend contribution
            (df['MACD_12_26_9'] - df['MACDs_12_26_9']) * 0.3  # MACD contribution
        ) * (1 - flux_level * 0.2)  # Flux penalty

        print("✅ NeuroFlux enhanced indicators added successfully")
        return df

    except Exception as e:
        print(f"❌ Error adding neuro indicators: {str(e)}")
        traceback.print_exc()
        return df

# ============================================================================
# MARKET INFO FUNCTIONS (enhanced for NeuroFlux)
# ============================================================================

def get_market_info():
    """Get current market info with neuro-flux insights"""
    try:
        print("\n🔄 NeuroFlux fetching market data...")
        response = requests.post(
            BASE_URL,
            headers={'Content-Type': 'application/json'},
            json={"type": "allMids"}
        )

        if response.status_code == 200:
            data = response.json()

            # Add neuro-flux analysis
            flux_level = config.FLUX_SENSITIVITY
            analysis = {
                'timestamp': datetime.datetime.now().isoformat(),
                'flux_level': flux_level,
                'market_stability_score': 1 - flux_level,  # Inverse relationship
                'recommended_leverage': get_adaptive_leverage(5),
                'risk_multiplier': 1 + (flux_level * 0.5)
            }

            return {
                'prices': data,
                'neuro_analysis': analysis
            }
        print(f"❌ Bad status code: {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ Error getting market info: {str(e)}")
        traceback.print_exc()
        return None

# ============================================================================
# FUNDING RATE FUNCTIONS (enhanced for NeuroFlux)
# ============================================================================

def get_funding_rates(symbol):
    """
    Get current funding rate with neuro-flux implications

    Args:
        symbol (str): Trading pair symbol

    Returns:
        dict: Funding data with neuro-flux analysis
    """
    try:
        print(f"\n🔄 NeuroFlux fetching funding rate for {symbol}...")
        response = requests.post(
            BASE_URL,
            headers={'Content-Type': 'application/json'},
            json={"type": "metaAndAssetCtxs"}
        )

        if response.status_code == 200:
            data = response.json()
            if len(data) >= 2 and isinstance(data[0], dict) and isinstance(data[1], list):
                # Get universe (symbols) from first element
                universe = {coin['name']: i for i, coin in enumerate(data[0]['universe'])}

                # Check if symbol exists
                if symbol not in universe:
                    print(f"❌ Symbol {symbol} not found in Hyperliquid universe")
                    return None

                # Get funding data from second element
                funding_data = data[1]
                idx = universe[symbol]

                if idx < len(funding_data):
                    asset_data = funding_data[idx]

                    # Neuro-flux analysis of funding rate
                    funding_rate = float(asset_data['funding'])
                    flux_level = config.FLUX_SENSITIVITY

                    # Funding rate implications for neuro-flux
                    funding_signal = 'NEUTRAL'
                    if funding_rate > 0.0001:  # Positive funding > 0.01%
                        funding_signal = 'BULLISH' if flux_level < 0.5 else 'CAUTION'
                    elif funding_rate < -0.0001:  # Negative funding < -0.01%
                        funding_signal = 'BEARISH' if flux_level < 0.5 else 'CAUTION'

                    return {
                        'funding_rate': funding_rate,
                        'mark_price': float(asset_data['markPx']),
                        'open_interest': float(asset_data['openInterest']),
                        'neuro_analysis': {
                            'funding_signal': funding_signal,
                            'flux_adjusted_rate': funding_rate * (1 + flux_level * 0.2),
                            'trading_implication': 'Longs pay shorts' if funding_rate > 0 else 'Shorts pay longs',
                            'confidence': 1 - flux_level
                        }
                    }

            print("❌ Unexpected response format")
            return None
        print(f"❌ Bad status code: {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ Error getting funding rate for {symbol}: {str(e)}")
        traceback.print_exc()
        return None

# Initialize on import
print("🧠 NeuroFlux HyperLiquid functions loaded successfully!")
print(f"🌊 Flux sensitivity: {config.FLUX_SENSITIVITY:.2f}")
print(f"⚡ Adaptive leverage: {get_adaptive_leverage(5)}x")