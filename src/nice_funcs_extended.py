"""
🧠 NeuroFlux's Extended Exchange Functions
Neuro-flux enhanced trading functions for Extended Exchange (X10) futures.
Built with love by Nyros Veil 🚀

Advanced trading functions with flux-adaptive parameters and AI-enhanced decision making.
Supports multi-exchange trading with real-time neuro-flux adjustments.

LEVERAGE & POSITION SIZING:
- All 'amount' parameters represent NOTIONAL position size (total exposure)
- Leverage is applied by the exchange, reducing required margin
- Example: $25 position at 20x leverage = $25 notional, $1.25 margin required
- Formula: Required Margin = Notional Position / Leverage
- Neuro-flux adaptation: Leverage adjusts based on market volatility
"""

import os
import asyncio
import time
import json
import requests
from typing import Dict, Optional, List
from decimal import Decimal
from termcolor import colored, cprint
from dotenv import load_dotenv
import traceback
import warnings
warnings.filterwarnings('ignore')

# Try importing Extended Exchange SDK
try:
    from x10.perpetual.trading_client import PerpetualTradingClient
    from x10.perpetual.configuration import TESTNET_CONFIG, MAINNET_CONFIG
    from x10.perpetual.orders import OrderSide
    from x10.perpetual.accounts import StarkPerpetualAccount
    EXTENDED_AVAILABLE = True
except ImportError:
    EXTENDED_AVAILABLE = False
    cprint("⚠️  Extended Exchange SDK not available - running in simulation mode", "yellow")

# Load environment variables
load_dotenv()

# Add NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================================
# NEURO-FLUX CONFIGURATION
# ============================================================================

# Adaptive leverage based on flux sensitivity
def get_adaptive_leverage(base_leverage=20):
    """Get flux-adjusted leverage"""
    flux_penalty = config.FLUX_SENSITIVITY * 0.5  # Reduce leverage in high flux
    adjusted_leverage = max(1, base_leverage * (1 - flux_penalty))
    return int(adjusted_leverage)

DEFAULT_LEVERAGE = get_adaptive_leverage(20)  # Flux-adaptive default

# Constants
DEFAULT_SYMBOL_SUFFIX = '-USD'  # Extended uses BTC-USD, ETH-USD, etc.

# ============================================================================
# NEURO-FLUX ENHANCED HELPER FUNCTIONS
# ============================================================================

def format_symbol_for_extended(symbol: str) -> str:
    """Convert token address/symbol to Extended format with flux validation"""
    if '-USD' in symbol.upper():
        return symbol
    return f"{symbol}-USD"


# ============================================================================
# NEURO-FLUX ENHANCED EXTENDED EXCHANGE API CLASS
# ============================================================================

class NeuroFluxExtendedAPI:
    """Neuro-flux enhanced Extended Exchange API wrapper"""

    def __init__(self, api_key: str = None, private_key: str = None, public_key: str = None, vault_id: int = None):
        """Initialize Extended Exchange API client with neuro-flux awareness"""
        self.api_key = api_key or os.getenv("X10_API_KEY")
        self.private_key = private_key or os.getenv("X10_PRIVATE_KEY")
        self.public_key = public_key or os.getenv("X10_PUBLIC_KEY")
        self.vault_id = vault_id or int(os.getenv("X10_VAULT_ID", "110198"))

        if EXTENDED_AVAILABLE and not all([self.api_key, self.private_key, self.public_key]):
            raise ValueError("Extended Exchange credentials not found in environment!")

        # Initialize configuration
        self.config = TESTNET_CONFIG if os.getenv("USE_TESTNET", "false").lower() == "true" else MAINNET_CONFIG

        if EXTENDED_AVAILABLE:
            # Create StarkPerpetualAccount
            self.stark_account = StarkPerpetualAccount(
                vault=self.vault_id,
                private_key=self.private_key,
                public_key=self.public_key,
                api_key=self.api_key
            )

            # Create trading client
            self.trading_client = PerpetualTradingClient(self.config, self.stark_account)

            # REST API session
            self.base_url = "https://api.starknet.extended.exchange" if not os.getenv("USE_TESTNET", "false").lower() == "true" else "https://api-testnet.extended.exchange"
            self.session = requests.Session()
            self.session.headers.update({
                "X-Api-Key": self.api_key,
                "User-Agent": "neuroflux-extended-bot/1.0",
                "Content-Type": "application/json"
            })
        else:
            self.stark_account = None
            self.trading_client = None
            self.base_url = ""
            self.session = None

        # Event loop for async operations
        self._event_loop = None

    def _get_event_loop(self):
        """Get or create event loop"""
        if self._event_loop is None or self._event_loop.is_closed():
            try:
                self._event_loop = asyncio.get_event_loop()
            except RuntimeError:
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)
        return self._event_loop

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """REST API request handler"""
        if not EXTENDED_AVAILABLE:
            return {}
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            cprint(f"REST API Error: {e}", "red")
            raise

    def get_account_info(self) -> Dict:
        """Get account information with neuro-flux metrics"""
        if not EXTENDED_AVAILABLE:
            # Simulation mode
            return {
                "balance": {"equity": 10000},
                "positions": [],
                "stark_key": "simulated",
                "vault": self.vault_id,
                "flux_risk_score": config.FLUX_SENSITIVITY * 10
            }

        loop = self._get_event_loop()

        async def get_info():
            balance = await self.trading_client.account.get_balance()
            positions = await self.trading_client.account.get_positions()
            return {
                "balance": balance,
                "positions": positions,
                "stark_key": self.stark_account.public_key,
                "vault": self.stark_account.vault,
                "flux_risk_score": config.FLUX_SENSITIVITY * 10
            }

        return loop.run_until_complete(get_info())

    def get_position(self, symbol: str) -> Dict:
        """Get position with neuro-flux enhanced analysis"""
        symbol = format_symbol_for_extended(symbol)

        if not EXTENDED_AVAILABLE:
            # Simulation mode
            return {
                'position_amount': 0.001,
                'entry_price': 50000,
                'mark_price': 51000,
                'pnl': 100,
                'pnl_percentage': 2.0,
                'is_long': True,
                'flux_risk_score': config.FLUX_SENSITIVITY * 10,
                'volatility_adjustment': config.FLUX_SENSITIVITY * 0.1
            }

        try:
            account_info = self.get_account_info()
            positions = account_info['positions'].data

            for pos in positions:
                market_attr = getattr(pos, 'market_name', None) or getattr(pos, 'market', None) or getattr(pos, 'symbol', None)

                if market_attr == symbol:
                    mypos_size = float(pos.size)
                    entry_px = float(getattr(pos, 'entry_price', 0) or getattr(pos, 'open_price', 0))
                    mark_px = float(getattr(pos, 'mark_price', 0) or entry_px)

                    unrealized_pnl = float(getattr(pos, 'unrealised_pnl', 0) or getattr(pos, 'unrealized_pnl', 0))

                    position_leverage = float(getattr(pos, 'leverage', DEFAULT_LEVERAGE))
                    actual_margin = mypos_size * entry_px / position_leverage if position_leverage > 0 else mypos_size * entry_px
                    pnl_perc = (unrealized_pnl / actual_margin) * 100 if actual_margin > 0 else 0

                    side = str(getattr(pos, 'side', 'LONG')).upper()
                    is_long = side != 'SHORT'
                    signed_size = -mypos_size if side == 'SHORT' else mypos_size

                    return {
                        'position_amount': signed_size,
                        'entry_price': entry_px,
                        'mark_price': mark_px,
                        'pnl': unrealized_pnl,
                        'pnl_percentage': pnl_perc,
                        'is_long': is_long,
                        'flux_risk_score': config.FLUX_SENSITIVITY * 10,
                        'volatility_adjustment': config.FLUX_SENSITIVITY * 0.1
                    }

            return None

        except Exception as e:
            cprint(f"Error getting position: {e}", "red")
            return None

    def set_leverage(self, symbol: str, leverage: int):
        """Set leverage with flux validation"""
        if not EXTENDED_AVAILABLE:
            print(f"⚙️  Simulated leverage set to {leverage}x for {symbol}")
            return

        # Validate leverage against flux conditions
        max_leverage = 125  # Extended max
        flux_adjusted_max = max_leverage * (1 - config.FLUX_SENSITIVITY)
        leverage = min(leverage, flux_adjusted_max)

        loop = self._get_event_loop()

        async def set_lev():
            try:
                return await self.trading_client.account.update_leverage(
                    market_name=symbol,
                    leverage=leverage
                )
            except Exception as e:
                cprint(f"Leverage update note: {e}", "yellow")
                return None

        return loop.run_until_complete(set_lev())

    def buy_market(self, symbol: str, quantity: float, leverage: int = None) -> Dict:
        """Neuro-flux enhanced market buy"""
        if not EXTENDED_AVAILABLE:
            return {'orderId': 'simulated', 'status': 'FILLED', 'simulated': True}

        if leverage:
            self.set_leverage(symbol, leverage)
            time.sleep(0.3)

        loop = self._get_event_loop()

        async def place_order():
            orderbook = await self.trading_client.markets_info.get_orderbook_snapshot(market_name=symbol)
            ask_price = float(orderbook.data.ask[0].price) if orderbook.data.ask else 0
            aggressive_price = round(ask_price * (1.01 + config.FLUX_SENSITIVITY * 0.005))  # Flux increases aggression

            response = await self.trading_client.place_order(
                market_name=symbol,
                amount_of_synthetic=Decimal(str(quantity)),
                price=Decimal(str(int(aggressive_price))),
                side=OrderSide.BUY,
                post_only=False
            )
            return response

        return loop.run_until_complete(place_order())

    def sell_market(self, symbol: str, quantity: float, leverage: int = None) -> Dict:
        """Neuro-flux enhanced market sell"""
        if not EXTENDED_AVAILABLE:
            return {'orderId': 'simulated', 'status': 'FILLED', 'simulated': True}

        if leverage:
            self.set_leverage(symbol, leverage)
            time.sleep(0.3)

        loop = self._get_event_loop()

        async def place_order():
            orderbook = await self.trading_client.markets_info.get_orderbook_snapshot(market_name=symbol)
            bid_price = float(orderbook.data.bid[0].price) if orderbook.data.bid else 0
            aggressive_price = round(bid_price * (0.99 - config.FLUX_SENSITIVITY * 0.005))  # Flux increases aggression

            response = await self.trading_client.place_order(
                market_name=symbol,
                amount_of_synthetic=Decimal(str(quantity)),
                price=Decimal(str(int(aggressive_price))),
                side=OrderSide.SELL,
                post_only=False
            )
            return response

        return loop.run_until_complete(place_order())

    def close_position(self, symbol: str) -> bool:
        """Close position with flux-aware execution"""
        if not EXTENDED_AVAILABLE:
            print("⚠️  Extended API not available - simulating close")
            return True

        try:
            position = self.get_position(symbol)

            if not position or position['position_amount'] == 0:
                return False

            self.cancel_all_orders(symbol)
            time.sleep(0.5)

            mypos_size = position['position_amount']
            is_long = position['is_long']
            close_size = abs(mypos_size)

            if is_long:
                self.sell_market(symbol, close_size)
            else:
                self.buy_market(symbol, close_size)

            time.sleep(2)

            new_position = self.get_position(symbol)
            return not new_position or new_position['position_amount'] == 0

        except Exception as e:
            cprint(f"Error closing position: {e}", "red")
            return False

    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all orders"""
        if not EXTENDED_AVAILABLE:
            return False

        try:
            data = {}
            if symbol:
                data["market"] = symbol

            response = self._request("POST", "/api/v1/user/order/massCancel", data)
            return response.get('success', False) or response.get('status') == 'OK'
        except Exception as e:
            cprint(f"Error canceling orders: {e}", "yellow")
            return False

    def get_account_balance(self) -> Dict:
        """Get account balance with flux adjustments"""
        if not EXTENDED_AVAILABLE:
            adjusted_equity = 10000 * (1 - config.FLUX_SENSITIVITY * 0.2)
            return {"equity": adjusted_equity}

        try:
            account_info = self.get_account_info()
            equity = 0
            if hasattr(account_info['balance'], 'data'):
                if hasattr(account_info['balance'].data, 'equity'):
                    equity = float(account_info['balance'].data.equity)
                elif hasattr(account_info['balance'].data, 'total_balance'):
                    equity = float(account_info['balance'].data.total_balance)

            # Apply flux adjustment
            flux_adjustment = config.FLUX_SENSITIVITY * 0.2
            adjusted_equity = equity * (1 - flux_adjustment)

            return {"equity": adjusted_equity, "raw_equity": equity}
        except Exception as e:
            cprint(f"Error getting account balance: {e}", "red")
            return {"equity": 0}

    def usd_to_asset_size(self, symbol: str, usd_amount: float) -> float:
        """Convert USD to asset size with flux adjustments"""
        if not EXTENDED_AVAILABLE:
            # Simulation mode
            base_price = 50000 if 'BTC' in symbol else 3000
            asset_size = usd_amount / base_price
            if 'BTC' in symbol:
                asset_size = round(asset_size, 3)
            elif 'ETH' in symbol:
                asset_size = round(asset_size, 4)
            else:
                asset_size = round(asset_size, 4)
            return asset_size

        try:
            loop = self._get_event_loop()

            async def get_prices():
                orderbook = await self.trading_client.markets_info.get_orderbook_snapshot(market_name=symbol)
                return {
                    "bid": float(orderbook.data.bid[0].price) if orderbook.data.bid else 0.0,
                    "ask": float(orderbook.data.ask[0].price) if orderbook.data.ask else 0.0
                }

            bid_ask = loop.run_until_complete(get_prices())
            mid_price = (bid_ask['bid'] + bid_ask['ask']) / 2

            if mid_price <= 0:
                cprint(f"❌ Invalid price for {symbol}: {mid_price}", "red")
                return 0

            asset_size = usd_amount / mid_price

            # Apply flux adjustment to minimum sizes
            flux_multiplier = 1 + (config.FLUX_SENSITIVITY * 0.5)

            # Round to appropriate precision for different assets
            if 'BTC' in symbol:
                asset_size = round(asset_size, 3)
                if asset_size == 0.0 and usd_amount > 0:
                    asset_size = 0.001 * flux_multiplier
            elif 'ETH' in symbol:
                asset_size = round(asset_size, 4)
                if asset_size == 0 and usd_amount > 0:
                    asset_size = max(0.0001 * flux_multiplier, asset_size)
            elif 'SOL' in symbol:
                asset_size = round(asset_size, 2)
                if asset_size == 0 and usd_amount > 0:
                    asset_size = max(0.01 * flux_multiplier, asset_size)
            else:
                asset_size = round(asset_size, 4)
                if asset_size == 0 and usd_amount > 0:
                    asset_size = max(0.0001 * flux_multiplier, asset_size)

            cprint(f"💱 USD to Asset: ${usd_amount} → {asset_size} {symbol} @ ${mid_price:,.2f}", "cyan")
            return asset_size

        except Exception as e:
            cprint(f"❌ Error converting USD to asset size: {e}", "red")
            return 0


# ============================================================================
# GLOBAL API INSTANCE
# ============================================================================

# Create global API instance
try:
    api = NeuroFluxExtendedAPI()
    cprint("🧠 NeuroFlux Extended Exchange API initialized", "green")
except Exception as e:
    cprint(f"⚠️  Failed to initialize Extended Exchange API: {e}", "yellow")
    api = None

# ============================================================================
# NEURO-FLUX ENHANCED TRADING FUNCTIONS
# ============================================================================

def neuro_adaptive_market_buy(symbol: str, usd_size: float, sentiment_score=0.5):
    """
    🧠 Neuro-flux enhanced market buy with AI adaptation

    Args:
        symbol: Token symbol
        usd_size: Base USD amount
        sentiment_score: Market sentiment (0-1, 0.5=neutral)

    Returns:
        dict: Trade execution result with flux adjustments
    """
    print(colored(f'🧠 NeuroFlux Extended Market BUY {symbol} for ${usd_size}', 'green'))

    # Neuro-flux decision making
    flux_level = config.FLUX_SENSITIVITY

    # Adjust position size based on sentiment and flux
    sentiment_multiplier = 0.5 + sentiment_score  # 0.5-1.5 range
    flux_multiplier = 1 - (flux_level * 0.4)  # Reduce size in high flux
    adjusted_size = usd_size * sentiment_multiplier * flux_multiplier

    # Adjust leverage based on flux
    base_leverage = 20
    adjusted_leverage = get_adaptive_leverage(base_leverage)

    print(f'🧠 Neuro-adaptive buy: Sentiment {sentiment_score:.2f}, Flux {flux_level:.2f}')
    print(f'💰 Size: ${usd_size:.0f} → ${adjusted_size:.0f} | Leverage: {base_leverage}x → {adjusted_leverage}x')

    # Execute trade
    result = market_buy(symbol, adjusted_size, leverage=adjusted_leverage, flux_adjust=False)

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


def market_buy(symbol: str, usd_amount: float, slippage=None, leverage=DEFAULT_LEVERAGE, flux_adjust=True):
    """Neuro-flux enhanced market buy"""
    print(colored(f'🧠 NeuroFlux Extended Market BUY {symbol} for ${usd_amount}', 'green'))

    if api is None:
        print("⚠️  Extended API not available - simulating trade")
        return {'orderId': 'simulated', 'status': 'FILLED', 'simulated': True}

    symbol = format_symbol_for_extended(symbol)

    # Apply flux adjustments
    if flux_adjust:
        flux_multiplier = 1 - (config.FLUX_SENSITIVITY * 0.3)
        usd_amount = usd_amount * flux_multiplier
        print(f'🌊 Flux-adjusted size: ${usd_amount:.2f}')

    # Convert USD to asset size
    quantity = api.usd_to_asset_size(symbol, usd_amount)

    if quantity <= 0:
        raise Exception(f"Invalid quantity calculated: {quantity}")

    return api.buy_market(symbol, quantity, leverage)


def market_sell(symbol: str, usd_amount: float, slippage=None, leverage=DEFAULT_LEVERAGE, flux_adjust=True):
    """Neuro-flux enhanced market sell"""
    print(colored(f'🧠 NeuroFlux Extended Market SELL {symbol} for ${usd_amount}', 'red'))

    if api is None:
        print("⚠️  Extended API not available - simulating trade")
        return {'orderId': 'simulated', 'status': 'FILLED', 'simulated': True}

    symbol = format_symbol_for_extended(symbol)

    # Apply flux adjustments
    if flux_adjust:
        flux_multiplier = 1 - (config.FLUX_SENSITIVITY * 0.3)
        usd_amount = usd_amount * flux_multiplier
        print(f'🌊 Flux-adjusted size: ${usd_amount:.2f}')

    # Convert USD to asset size
    quantity = api.usd_to_asset_size(symbol, usd_amount)

    if quantity <= 0:
        raise Exception(f"Invalid quantity calculated: {quantity}")

    return api.sell_market(symbol, quantity, leverage)


def get_position(symbol: str) -> Dict:
    """Get position with neuro-flux metrics"""
    if api is None:
        return None
    return api.get_position(symbol)


def get_balance():
    """Get USDC balance with flux adjustments"""
    if api is None:
        return 0
    balance_info = api.get_account_balance()
    return balance_info.get('equity', 0)


def flux_aware_risk_management(symbol: str):
    """
    Neuro-flux aware risk management for Extended

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
    flux_adjusted_stop = base_stop_loss + (flux_level * 15)  # Up to 20% in high flux

    # Dynamic take profit based on flux
    base_take_profit = 10
    flux_adjusted_profit = base_take_profit - (flux_level * 7.5)  # Down to 2.5% in high flux

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


def kill_switch(symbol: str):
    """Emergency close position with flux awareness"""
    print(colored(f'🔪 NeuroFlux KILL SWITCH ACTIVATED for {symbol}', 'red', attrs=['bold']))

    if api is None:
        print("⚠️  Extended API not available - simulating close")
        return {'simulated': True}

    symbol = format_symbol_for_extended(symbol)
    return api.close_position(symbol)


def get_current_price(symbol: str) -> float:
    """Get current mid price with flux adjustments"""
    if api is None:
        return 50000 if 'BTC' in symbol else 3000

    symbol = format_symbol_for_extended(symbol)

    loop = api._get_event_loop()

    async def get_price():
        orderbook = await api.trading_client.markets_info.get_orderbook_snapshot(market_name=symbol)
        bid = float(orderbook.data.bid[0].price) if orderbook.data.bid else 0
        ask = float(orderbook.data.ask[0].price) if orderbook.data.ask else 0
        mid = (bid + ask) / 2 if bid and ask else 0
        # Apply flux adjustment to price (slight volatility increase)
        flux_adjustment = config.FLUX_SENSITIVITY * 0.01
        return mid * (1 + flux_adjustment)

    return loop.run_until_complete(get_price())


# ============================================================================
# LEGACY FUNCTION ALIASES (for compatibility)
# ============================================================================

def get_account_balance() -> Dict:
    """Legacy balance getter"""
    return {"equity": get_balance()}

def close_position(symbol: str) -> bool:
    """Legacy close position"""
    return kill_switch(symbol)

def open_long(symbol: str, usd_amount: float, slippage=None, leverage=DEFAULT_LEVERAGE):
    """Legacy open long"""
    return market_buy(symbol, usd_amount, slippage, leverage)

def open_short(symbol: str, usd_amount: float, slippage=None, leverage=DEFAULT_LEVERAGE):
    """Legacy open short"""
    return market_sell(symbol, usd_amount, slippage, leverage)

def cancel_all_orders(symbol: str):
    """Legacy cancel orders"""
    if api is None:
        return
    api.cancel_all_orders(symbol)

def chunk_kill(symbol: str, max_chunk_size: float = 999999, slippage: int = None) -> bool:
    """Legacy chunk kill - simplified"""
    return kill_switch(symbol)


# Initialize on import
if EXTENDED_AVAILABLE and api:
    print("🧠 NeuroFlux Extended Exchange functions loaded successfully!")
    print(f"🌊 Flux sensitivity: {config.FLUX_SENSITIVITY:.2f}")
    print(f"⚡ Adaptive leverage: {get_adaptive_leverage(20)}x")
else:
    print("🧠 NeuroFlux Extended Exchange functions loaded in simulation mode!")
    print("⚠️  Extended API not available - trades will be simulated")