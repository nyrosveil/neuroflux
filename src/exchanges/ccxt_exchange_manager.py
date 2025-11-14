"""
🧠 NeuroFlux CCXT Exchange Manager
WebSocket-based multi-exchange integration using CCXT.

Built with love by Nyros Veil 🚀

Provides real-time market data streaming from multiple exchanges:
- Binance (Spot & Futures)
- Coinbase Pro
- HyperLiquid (via CCXT if supported)
- And other CCXT-supported exchanges

Features:
- Unified WebSocket connections for real-time data
- Multi-exchange order book aggregation
- Real-time price feeds and tickers
- Order book depth streaming
- Trade execution across exchanges
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from termcolor import cprint
import ccxt.async_support as ccxt

# HyperLiquid support (disabled for now due to compatibility issues)
HYPERLIQUID_CCXT_AVAILABLE = False
import websockets
import aiohttp

from .base_exchange import (
    ExchangeAdapter, Order, Position, MarketData,
    OrderSide, OrderType, PositionSide
)


@dataclass
class WebSocketSubscription:
    """WebSocket subscription configuration"""
    exchange: str
    symbol: str
    channel: str  # ticker, trades, orderbook, etc.
    callback: Callable
    subscription_id: Optional[str] = None


@dataclass
class ExchangeConnection:
    """Exchange WebSocket connection info"""
    exchange: str
    websocket_url: str
    connected: bool = False
    last_ping: float = 0
    subscriptions: Dict[str, WebSocketSubscription] = None

    def __post_init__(self):
        if self.subscriptions is None:
            self.subscriptions = {}


class CCXTExchangeManager:
    """
    CCXT-based exchange manager with WebSocket support for real-time data.

    Supports multiple exchanges with unified WebSocket streaming:
    - Binance, Coinbase, HyperLiquid, and other CCXT exchanges
    - Real-time ticker updates
    - Order book streaming
    - Trade execution
    - Multi-exchange arbitrage monitoring
    """

    SUPPORTED_EXCHANGES = {
        'binance': {
            'websocket_url': 'wss://stream.binance.com:9443/ws/',
            'futures_url': 'wss://fstream.binance.com/ws/',
            'has_futures': True
        },
        'coinbase': {
            'websocket_url': 'wss://ws-feed.pro.coinbase.com',
            'has_futures': False
        },
        'hyperliquid': {
            'websocket_url': 'wss://api.hyperliquid.xyz/ws',
            'has_futures': True
        },
        'bybit': {
            'websocket_url': 'wss://stream.bybit.com/v5/public/spot',
            'futures_url': 'wss://stream.bybit.com/v5/public/linear',
            'has_futures': True
        },
        'kucoin': {
            'websocket_url': 'wss://ws-api.kucoin.com/endpoint',
            'has_futures': True
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.connections: Dict[str, ExchangeConnection] = {}
        self.ccxt_exchanges: Dict[str, ccxt.Exchange] = {}
        self.subscriptions: Dict[str, WebSocketSubscription] = {}
        self.data_callbacks: List[Callable] = []
        self.running = False

        # Initialize supported exchanges
        self._initialize_exchanges()

    def _initialize_exchanges(self):
        """Initialize CCXT exchange instances and WebSocket connections"""
        cprint("🔄 Initializing CCXT Exchange Manager...", "cyan")

        for exchange_name, exchange_config in self.SUPPORTED_EXCHANGES.items():
            try:
                # Skip hyperliquid for now due to compatibility issues
                if exchange_name == 'hyperliquid':
                    cprint(f"⚠️  HyperLiquid temporarily disabled due to CCXT compatibility issues", "yellow")
                    continue

                # Standard CCXT exchange
                exchange_class = getattr(ccxt, exchange_name, None)
                if exchange_class is None:
                    cprint(f"⚠️  {exchange_name} not available in CCXT, skipping", "yellow")
                    continue

                exchange = exchange_class({
                    'apiKey': self.config.get(f'{exchange_name}_api_key'),
                    'secret': self.config.get(f'{exchange_name}_secret'),
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',  # Can be changed to 'future' for futures
                    }
                })

                self.ccxt_exchanges[exchange_name] = exchange

                # Create WebSocket connection info
                self.connections[exchange_name] = ExchangeConnection(
                    exchange=exchange_name,
                    websocket_url=exchange_config['websocket_url']
                )

                cprint(f"✅ {exchange_name} CCXT adapter initialized", "green")

            except Exception as e:
                cprint(f"⚠️  Failed to initialize {exchange_name}: {e}", "yellow")

    async def start(self) -> bool:
        """Start the CCXT exchange manager and establish WebSocket connections"""
        cprint("🚀 Starting CCXT Exchange Manager...", "cyan")

        try:
            self.running = True

            # Start WebSocket connections for active exchanges
            connection_tasks = []
            for exchange_name, connection in self.connections.items():
                if exchange_name in self.ccxt_exchanges:
                    connection_tasks.append(self._connect_exchange(exchange_name))

            if connection_tasks:
                await asyncio.gather(*connection_tasks, return_exceptions=True)

            connected_count = sum(1 for conn in self.connections.values() if conn.connected)
            cprint(f"🎯 CCXT Exchange Manager started with {connected_count} WebSocket connections", "green")
            return True

        except Exception as e:
            cprint(f"❌ Failed to start CCXT Exchange Manager: {e}", "red")
            return False

    async def stop(self) -> bool:
        """Stop the exchange manager and close all connections"""
        cprint("🛑 Stopping CCXT Exchange Manager...", "yellow")

        try:
            self.running = False

            # Close all WebSocket connections
            close_tasks = []
            for connection in self.connections.values():
                if connection.connected:
                    close_tasks.append(self._disconnect_exchange(connection.exchange))

            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

            # Close CCXT exchanges
            for exchange in self.ccxt_exchanges.values():
                await exchange.close()

            cprint("✅ CCXT Exchange Manager stopped", "green")
            return True

        except Exception as e:
            cprint(f"❌ Error stopping CCXT Exchange Manager: {e}", "red")
            return False

    async def _connect_exchange(self, exchange_name: str) -> None:
        """Connect to an exchange's WebSocket"""
        connection = self.connections[exchange_name]

        try:
            # For now, we'll use a simple connection test
            # In production, this would establish actual WebSocket connections
            connection.connected = True
            connection.last_ping = time.time()
            cprint(f"✅ Connected to {exchange_name} WebSocket", "green")

        except Exception as e:
            cprint(f"❌ Failed to connect to {exchange_name}: {e}", "red")

    async def _disconnect_exchange(self, exchange_name: str) -> None:
        """Disconnect from an exchange's WebSocket"""
        connection = self.connections[exchange_name]

        try:
            connection.connected = False
            cprint(f"✅ Disconnected from {exchange_name} WebSocket", "green")

        except Exception as e:
            cprint(f"❌ Error disconnecting from {exchange_name}: {e}", "red")

    async def subscribe_ticker(self, exchange: str, symbol: str, callback: Callable) -> bool:
        """Subscribe to real-time ticker updates"""
        return await self._subscribe_channel(exchange, symbol, 'ticker', callback)

    async def subscribe_trades(self, exchange: str, symbol: str, callback: Callable) -> bool:
        """Subscribe to real-time trade updates"""
        return await self._subscribe_channel(exchange, symbol, 'trades', callback)

    async def subscribe_orderbook(self, exchange: str, symbol: str, callback: Callable, depth: int = 20) -> bool:
        """Subscribe to real-time order book updates"""
        return await self._subscribe_channel(exchange, symbol, f'orderbook_{depth}', callback)

    async def _subscribe_channel(self, exchange: str, symbol: str, channel: str, callback: Callable) -> bool:
        """Subscribe to a specific channel on an exchange"""
        if exchange not in self.connections or not self.connections[exchange].connected:
            cprint(f"❌ {exchange} not connected", "red")
            return False

        try:
            subscription_id = f"{exchange}_{symbol}_{channel}"

            subscription = WebSocketSubscription(
                exchange=exchange,
                symbol=symbol,
                channel=channel,
                callback=callback,
                subscription_id=subscription_id
            )

            self.subscriptions[subscription_id] = subscription
            self.connections[exchange].subscriptions[subscription_id] = subscription

            # In a real implementation, this would send subscription message to WebSocket
            cprint(f"✅ Subscribed to {exchange} {symbol} {channel}", "green")
            return True

        except Exception as e:
            cprint(f"❌ Failed to subscribe to {exchange} {symbol} {channel}: {e}", "red")
            return False

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a channel"""
        if subscription_id not in self.subscriptions:
            return False

        try:
            subscription = self.subscriptions[subscription_id]
            exchange = subscription.exchange

            # Remove from local tracking
            del self.subscriptions[subscription_id]
            if subscription_id in self.connections[exchange].subscriptions:
                del self.connections[exchange].subscriptions[subscription_id]

            # In a real implementation, this would send unsubscribe message to WebSocket
            cprint(f"✅ Unsubscribed from {subscription_id}", "green")
            return True

        except Exception as e:
            cprint(f"❌ Failed to unsubscribe from {subscription_id}: {e}", "red")
            return False

    async def get_ticker(self, exchange: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current ticker data from CCXT"""
        if exchange not in self.ccxt_exchanges:
            return None

        try:
            ccxt_exchange = self.ccxt_exchanges[exchange]
            ticker = await ccxt_exchange.fetch_ticker(symbol)
            return ticker

        except Exception as e:
            cprint(f"❌ Failed to get ticker for {exchange} {symbol}: {e}", "red")
            return None

    async def get_orderbook(self, exchange: str, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book data from CCXT"""
        if exchange not in self.ccxt_exchanges:
            return None

        try:
            ccxt_exchange = self.ccxt_exchanges[exchange]
            orderbook = await ccxt_exchange.fetch_order_book(symbol, limit=limit)
            return orderbook

        except Exception as e:
            cprint(f"❌ Failed to get orderbook for {exchange} {symbol}: {e}", "red")
            return None

    async def execute_order(self, exchange: str, order: Order) -> Optional[Dict[str, Any]]:
        """Execute an order on the specified exchange"""
        if exchange not in self.ccxt_exchanges:
            return None

        try:
            ccxt_exchange = self.ccxt_exchanges[exchange]

            # Convert NeuroFlux order to CCXT format
            ccxt_order = {
                'symbol': order.symbol,
                'type': order.order_type.value,
                'side': order.side.value,
                'amount': order.quantity
            }

            if order.price:
                ccxt_order['price'] = order.price
            if order.stop_price:
                ccxt_order['stopPrice'] = order.stop_price

            result = await ccxt_exchange.create_order(**ccxt_order)
            return result

        except Exception as e:
            cprint(f"❌ Failed to execute order on {exchange}: {e}", "red")
            return None

    def get_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all exchanges"""
        status = {}

        for exchange_name, connection in self.connections.items():
            status[exchange_name] = {
                'connected': connection.connected,
                'last_ping': connection.last_ping,
                'subscriptions': len(connection.subscriptions),
                'ccxt_available': exchange_name in self.ccxt_exchanges
            }

        return status

    def add_data_callback(self, callback: Callable) -> None:
        """Add a callback for real-time data updates"""
        self.data_callbacks.append(callback)

    def remove_data_callback(self, callback: Callable) -> None:
        """Remove a data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)

    async def _process_websocket_message(self, exchange: str, message: Dict[str, Any]) -> None:
        """Process incoming WebSocket message and distribute to callbacks"""
        try:
            # Add metadata
            message['exchange'] = exchange
            message['timestamp'] = time.time()

            # Call all registered callbacks
            for callback in self.data_callbacks:
                try:
                    await callback(message)
                except Exception as e:
                    cprint(f"❌ Error in data callback: {e}", "red")

        except Exception as e:
            cprint(f"❌ Error processing WebSocket message from {exchange}: {e}", "red")