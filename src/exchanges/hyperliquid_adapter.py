"""
🧠 NeuroFlux HyperLiquid Exchange Adapter
HyperLiquid exchange implementation for the unified exchange interface.

Built with love by Nyros Veil 🚀

This adapter provides HyperLiquid-specific implementations of the ExchangeAdapter
interface, with neuro-flux enhanced trading parameters and real-time data streaming.
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Any
from decimal import Decimal
from termcolor import colored, cprint
import traceback

from .base_exchange import (
    ExchangeAdapter, Order, Position, MarketData,
    OrderSide, OrderType, PositionSide
)

# Try importing HyperLiquid SDK
try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    from eth_account.signers.local import LocalAccount
    import eth_account
    HYPERLIQUID_AVAILABLE = True
except ImportError:
    HYPERLIQUID_AVAILABLE = False
    cprint("⚠️  HyperLiquid SDK not available - running in simulation mode", "yellow")


class HyperLiquidAdapter(ExchangeAdapter):
    """
    HyperLiquid exchange adapter implementing the unified ExchangeAdapter interface.

    Provides access to HyperLiquid perpetual futures with neuro-flux adaptations.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__('hyperliquid', config)

        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.wallet_address = config.get('wallet_address')

        # Initialize SDK clients
        self.info_client = None
        self.exchange_client = None
        self.account = None

        # Market data cache
        self.market_cache = {}
        self.cache_timestamp = 0
        self.cache_ttl = 60  # 1 minute cache

    async def connect(self) -> bool:
        """Establish connection to HyperLiquid"""
        if not HYPERLIQUID_AVAILABLE:
            cprint("⚠️  HyperLiquid SDK not available", "yellow")
            return False

        try:
            # Initialize info client (read-only)
            self.info_client = Info(base_url=constants.TESTNET_API_URL if self.config.get('testnet', True) else constants.MAINNET_API_URL)

            # Initialize exchange client if credentials provided
            if self.api_key and self.secret_key and self.wallet_address:
                self.account = eth_account.Account.from_key(self.secret_key)
                self.exchange_client = Exchange(
                    wallet=self.account,
                    base_url=constants.TESTNET_API_URL if self.config.get('testnet', True) else constants.MAINNET_API_URL,
                    vault_address=self.wallet_address if hasattr(self, 'vault_address') else None
                )
                cprint("✅ Connected to HyperLiquid with trading enabled", "green")
            else:
                cprint("✅ Connected to HyperLiquid (read-only mode)", "yellow")

            self.connected = True
            return True

        except Exception as e:
            cprint(f"❌ Failed to connect to HyperLiquid: {str(e)}", "red")
            return False

    async def disconnect(self) -> bool:
        """Close connection to HyperLiquid"""
        try:
            self.info_client = None
            self.exchange_client = None
            self.account = None
            self.connected = False
            cprint("✅ Disconnected from HyperLiquid", "green")
            return True
        except Exception as e:
            cprint(f"❌ Error disconnecting from HyperLiquid: {str(e)}", "red")
            return False

    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Decimal]:
        """Get account balance"""
        if not self.connected or not self.exchange_client:
            return {}

        try:
            # Get user state
            user_state = self.exchange_client.user_state(self.wallet_address)

            balances = {}
            for position in user_state.get('assetPositions', []):
                coin = position.get('position', {}).get('coin', '')
                if asset and coin != asset:
                    continue

                # Get USD value of position
                entry_price = float(position.get('position', {}).get('entryPx', 0))
                size = float(position.get('position', {}).get('szi', 0))
                usd_value = abs(entry_price * size)

                balances[coin] = Decimal(str(usd_value))

            return balances

        except Exception as e:
            cprint(f"❌ Error getting HyperLiquid balance: {str(e)}", "red")
            return {}

    async def get_positions(self) -> List[Position]:
        """Get all open positions"""
        if not self.connected or not self.exchange_client:
            return []

        try:
            user_state = self.exchange_client.user_state(self.wallet_address)
            positions = []

            for pos_data in user_state.get('assetPositions', []):
                position = pos_data.get('position', {})

                coin = position.get('coin', '')
                size = float(position.get('szi', 0))
                entry_price = float(position.get('entryPx', 0))
                unrealized_pnl = float(position.get('unrealizedPnl', 0))

                # Determine side
                side = PositionSide.LONG if size > 0 else PositionSide.SHORT

                # Get current price
                current_price = await self._get_current_price(coin)
                if not current_price:
                    current_price = entry_price

                # Calculate P&L percentage
                pnl_percentage = (unrealized_pnl / (abs(entry_price * size))) * 100 if entry_price * size != 0 else 0

                positions.append(Position(
                    symbol=coin,
                    side=side,
                    amount=Decimal(str(abs(size))),
                    entry_price=Decimal(str(entry_price)),
                    current_price=Decimal(str(current_price)),
                    pnl=Decimal(str(unrealized_pnl)),
                    pnl_percentage=Decimal(str(pnl_percentage)),
                    exchange='hyperliquid'
                ))

            return positions

        except Exception as e:
            cprint(f"❌ Error getting HyperLiquid positions: {str(e)}", "red")
            return []

    async def place_order(self, order: Order) -> Optional[str]:
        """Place an order on HyperLiquid"""
        if not self.connected or not self.exchange_client:
            cprint("❌ HyperLiquid not connected or no trading permissions", "red")
            return None

        try:
            # Convert order to HyperLiquid format
            hl_order = {
                'coin': order.symbol,
                'is_buy': order.side == OrderSide.BUY,
                'sz': float(order.amount),
                'limit_px': float(order.price) if order.price else None,
                'order_type': self._convert_order_type(order.type),
                'reduce_only': False  # TODO: Implement reduce-only logic
            }

            # Remove None values
            hl_order = {k: v for k, v in hl_order.items() if v is not None}

            # Place the order
            result = self.exchange_client.order(**hl_order)

            if result.get('status') == 'ok':
                order_id = result.get('response', {}).get('data', {}).get('statuses', [{}])[0].get('resting', {}).get('oid')
                return str(order_id) if order_id else None
            else:
                cprint(f"❌ HyperLiquid order failed: {result}", "red")
                return None

        except Exception as e:
            cprint(f"❌ Error placing HyperLiquid order: {str(e)}", "red")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        if not self.connected or not self.exchange_client:
            return False

        try:
            result = self.exchange_client.cancel(order_id, symbol)
            return result.get('status') == 'ok'
        except Exception as e:
            cprint(f"❌ Error canceling HyperLiquid order: {str(e)}", "red")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get status of a specific order"""
        if not self.connected:
            return None

        try:
            # Get open orders
            open_orders = await self.get_open_orders(symbol)
            for order in open_orders:
                if order.id == order_id:
                    return order
            return None
        except Exception as e:
            cprint(f"❌ Error getting HyperLiquid order status: {str(e)}", "red")
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        if not self.connected or not self.exchange_client:
            return []

        try:
            orders_data = self.exchange_client.open_orders(self.wallet_address)
            orders = []

            for order_data in orders_data:
                coin = order_data.get('coin', '')
                if symbol and coin != symbol:
                    continue

                # Convert HyperLiquid order to unified format
                order = Order(
                    id=str(order_data.get('oid', '')),
                    symbol=coin,
                    side=OrderSide.BUY if order_data.get('side') == 'B' else OrderSide.SELL,
                    type=self._convert_hl_order_type(order_data.get('order_type', {})),
                    amount=Decimal(str(order_data.get('sz', 0))),
                    price=Decimal(str(order_data.get('limit_px', 0))) if order_data.get('limit_px') else None,
                    status='open'
                )
                orders.append(order)

            return orders

        except Exception as e:
            cprint(f"❌ Error getting HyperLiquid open orders: {str(e)}", "red")
            return []

    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for a symbol"""
        if not self.connected:
            return None

        try:
            # Check cache first
            current_time = time.time()
            if symbol in self.market_cache and (current_time - self.cache_timestamp) < self.cache_ttl:
                return self.market_cache[symbol]

            # Get market data from HyperLiquid
            if self.info_client:
                # Get L2 book for best bid/ask
                l2_data = self.info_client.l2_snapshot(symbol)

                if l2_data and 'levels' in l2_data:
                    bids = l2_data['levels'][0]  # bids
                    asks = l2_data['levels'][1]  # asks

                    best_bid = float(bids[0]['px']) if bids else 0
                    best_ask = float(asks[0]['px']) if asks else 0

                    # Get 24h stats
                    meta_data = self.info_client.meta()
                    asset_info = next((asset for asset in meta_data.get('universe', []) if asset['name'] == symbol), {})

                    price = (best_bid + best_ask) / 2 if best_bid and best_ask else best_bid or best_ask or 0
                    volume_24h = float(asset_info.get('dayNtlVlm', '0'))
                    change_24h = 0  # HyperLiquid doesn't provide this directly

                    market_data = MarketData(
                        symbol=symbol,
                        price=Decimal(str(price)),
                        bid=Decimal(str(best_bid)),
                        ask=Decimal(str(best_ask)),
                        volume_24h=Decimal(str(volume_24h)),
                        change_24h=Decimal(str(change_24h))
                    )

                    # Cache the data
                    self.market_cache[symbol] = market_data
                    self.cache_timestamp = current_time

                    return market_data

            return None

        except Exception as e:
            cprint(f"❌ Error getting HyperLiquid market data: {str(e)}", "red")
            return None

    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Get historical OHLCV data"""
        if not self.connected or not self.info_client:
            return []

        try:
            # Convert timeframe to HyperLiquid format
            interval = self._convert_timeframe(timeframe)

            # Get candles
            candles = self.info_client.candles_snapshot(symbol, interval, limit)

            # Convert to standard OHLCV format
            ohlcv_data = []
            for candle in candles:
                ohlcv_data.append({
                    'timestamp': int(candle['t']),
                    'open': float(candle['o']),
                    'high': float(candle['h']),
                    'low': float(candle['l']),
                    'close': float(candle['c']),
                    'volume': float(candle['v'])
                })

            return ohlcv_data

        except Exception as e:
            cprint(f"❌ Error getting HyperLiquid historical data: {str(e)}", "red")
            return []

    async def subscribe_realtime_data(self, symbols: List[str], callback) -> bool:
        """Subscribe to real-time market data"""
        # HyperLiquid doesn't have WebSocket in their Python SDK
        # This would need to be implemented with their WebSocket API directly
        cprint("⚠️  Real-time data subscription not implemented for HyperLiquid", "yellow")
        return False

    async def unsubscribe_realtime_data(self, symbols: List[str]) -> bool:
        """Unsubscribe from real-time market data"""
        cprint("⚠️  Real-time data unsubscription not implemented for HyperLiquid", "yellow")
        return False

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        market_data = await self.get_market_data(symbol)
        return float(market_data.price) if market_data else None

    def _convert_order_type(self, order_type: OrderType) -> Dict:
        """Convert unified order type to HyperLiquid format"""
        if order_type == OrderType.MARKET:
            return {'limit': {'tif': 'Ioc'}}  # Immediate or Cancel for market orders
        elif order_type == OrderType.LIMIT:
            return {'limit': {'tif': 'Gtc'}}  # Good till canceled
        elif order_type == OrderType.STOP:
            return {'trigger': {'triggerPx': 0, 'isMarket': True}}  # TODO: Implement stop orders
        elif order_type == OrderType.STOP_LIMIT:
            return {'trigger': {'triggerPx': 0, 'isMarket': False}}  # TODO: Implement stop limit orders
        else:
            return {'limit': {'tif': 'Gtc'}}  # Default to limit

    def _convert_hl_order_type(self, hl_order_type: Dict) -> OrderType:
        """Convert HyperLiquid order type to unified format"""
        if 'limit' in hl_order_type:
            return OrderType.LIMIT
        elif 'trigger' in hl_order_type:
            return OrderType.STOP if hl_order_type['trigger'].get('isMarket') else OrderType.STOP_LIMIT
        else:
            return OrderType.MARKET

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to HyperLiquid interval"""
        # HyperLiquid intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d
        mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return mapping.get(timeframe, '1h')  # Default to 1h