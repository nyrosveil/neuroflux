"""
🧠 NeuroFlux Exchange Manager
Main orchestrator for multi-exchange trading operations.

Built with love by Nyros Veil 🚀

This module provides the main ExchangeManager class that coordinates
multiple exchange adapters, handles real-time data streaming, and
provides unified trading operations across all supported exchanges.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from termcolor import colored, cprint
import traceback

from .base_exchange import (
    ExchangeAdapter, Order, Position, MarketData,
    OrderSide, OrderType, PositionSide
)

# Import config if available
try:
    from ..config import EXCHANGE, ACTIVE_AGENTS
except ImportError:
    EXCHANGE = 'solana'
    ACTIVE_AGENTS = {}


@dataclass
class ExchangeStatus:
    """Status information for an exchange"""
    name: str
    connected: bool
    last_update: float
    active_symbols: List[str]
    open_positions: int
    open_orders: int


class ExchangeManager:
    """
    Main exchange manager that coordinates multiple exchange adapters.

    Provides unified interface for:
    - Multi-exchange trading operations
    - Real-time data streaming
    - Portfolio management
    - Risk monitoring across exchanges
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.exchanges: Dict[str, ExchangeAdapter] = {}
        self.active_exchanges: List[str] = []
        self.data_callbacks: List[Callable] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._running = False
        self._flux_level = 0.5  # Current neuro-flux level (0-1)

        # Initialize exchange adapters based on configuration
        self._initialize_exchanges()

    def _initialize_exchanges(self):
        """Initialize exchange adapters based on configuration"""
        cprint("🔄 Initializing Exchange Adapters...", "cyan")

        # Import exchange adapters dynamically
        try:
            from .hyperliquid_adapter import HyperLiquidAdapter
            self.exchanges['hyperliquid'] = HyperLiquidAdapter({
                'flux_sensitivity': self.config.get('flux_sensitivity', 0.8),
                'api_key': self.config.get('hyperliquid_api_key'),
                'secret_key': self.config.get('hyperliquid_secret_key'),
                'wallet_address': self.config.get('hyperliquid_wallet_address')
            })
            cprint("✅ HyperLiquid adapter initialized", "green")
        except ImportError:
            cprint("⚠️  HyperLiquid adapter not available", "yellow")

        try:
            from .extended_adapter import ExtendedAdapter
            self.exchanges['extended'] = ExtendedAdapter({
                'flux_sensitivity': self.config.get('flux_sensitivity', 0.8),
                'api_key': self.config.get('extended_api_key'),
                'secret_key': self.config.get('extended_secret_key')
            })
            cprint("✅ Extended Exchange adapter initialized", "green")
        except ImportError:
            cprint("⚠️  Extended Exchange adapter not available", "yellow")

        try:
            from .aster_adapter import AsterAdapter
            self.exchanges['aster'] = AsterAdapter({
                'flux_sensitivity': self.config.get('flux_sensitivity', 0.8),
                'api_key': self.config.get('aster_api_key'),
                'secret_key': self.config.get('aster_secret_key')
            })
            cprint("✅ Aster DEX adapter initialized", "green")
        except ImportError:
            cprint("⚠️  Aster DEX adapter not available", "yellow")

    async def start(self) -> bool:
        """Start the exchange manager and connect to exchanges"""
        cprint("🚀 Starting NeuroFlux Exchange Manager...", "cyan")

        try:
            # Connect to all configured exchanges
            for name, exchange in self.exchanges.items():
                try:
                    connected = await exchange.connect()
                    if connected:
                        self.active_exchanges.append(name)
                        cprint(f"✅ Connected to {name}", "green")
                    else:
                        cprint(f"❌ Failed to connect to {name}", "red")
                except Exception as e:
                    cprint(f"❌ Error connecting to {name}: {str(e)}", "red")

            if not self.active_exchanges:
                cprint("❌ No exchanges connected. Exchange Manager failed to start.", "red")
                return False

            self._running = True
            cprint(f"🎯 Exchange Manager started with {len(self.active_exchanges)} exchanges", "green")
            return True

        except Exception as e:
            cprint(f"❌ Failed to start Exchange Manager: {str(e)}", "red")
            return False

    async def stop(self) -> bool:
        """Stop the exchange manager and disconnect from exchanges"""
        cprint("🛑 Stopping Exchange Manager...", "yellow")

        try:
            # Disconnect from all exchanges
            disconnect_tasks = []
            for name, exchange in self.exchanges.items():
                if exchange.connected:
                    disconnect_tasks.append(exchange.disconnect())

            if disconnect_tasks:
                await asyncio.gather(*disconnect_tasks, return_exceptions=True)

            self.active_exchanges.clear()
            self._running = False
            cprint("✅ Exchange Manager stopped", "green")
            return True

        except Exception as e:
            cprint(f"❌ Error stopping Exchange Manager: {str(e)}", "red")
            return False

    def get_exchange_status(self) -> Dict[str, ExchangeStatus]:
        """Get status information for all exchanges"""
        status = {}

        for name, exchange in self.exchanges.items():
            try:
                # Get basic status info
                positions = asyncio.run(exchange.get_positions()) if exchange.connected else []
                orders = asyncio.run(exchange.get_open_orders()) if exchange.connected else []

                status[name] = ExchangeStatus(
                    name=name,
                    connected=exchange.connected,
                    last_update=time.time(),
                    active_symbols=[],  # TODO: Implement symbol tracking
                    open_positions=len(positions),
                    open_orders=len(orders)
                )
            except Exception as e:
                cprint(f"❌ Error getting status for {name}: {str(e)}", "red")
                status[name] = ExchangeStatus(
                    name=name,
                    connected=False,
                    last_update=time.time(),
                    active_symbols=[],
                    open_positions=0,
                    open_orders=0
                )

        return status

    async def get_portfolio_balance(self) -> Dict[str, float]:
        """Get combined portfolio balance across all exchanges"""
        total_balance = {}

        for name, exchange in self.exchanges.items():
            if not exchange.connected:
                continue

            try:
                balance = await exchange.get_balance()
                for asset, amount in balance.items():
                    if asset not in total_balance:
                        total_balance[asset] = 0.0
                    total_balance[asset] += float(amount)
            except Exception as e:
                cprint(f"❌ Error getting balance from {name}: {str(e)}", "red")

        return total_balance

    async def get_all_positions(self) -> List[Position]:
        """Get all positions across all exchanges"""
        all_positions = []

        for name, exchange in self.exchanges.items():
            if not exchange.connected:
                continue

            try:
                positions = await exchange.get_positions()
                # Add exchange identifier to each position
                for pos in positions:
                    pos.exchange = name
                all_positions.extend(positions)
            except Exception as e:
                cprint(f"❌ Error getting positions from {name}: {str(e)}", "red")

        return all_positions

    async def place_order(self, order: Order, exchange_preference: Optional[str] = None) -> Optional[str]:
        """
        Place an order on the preferred exchange or best available exchange

        Args:
            order: Order to place
            exchange_preference: Preferred exchange name, or None for auto-selection

        Returns:
            Order ID if successful, None otherwise
        """
        if exchange_preference and exchange_preference in self.exchanges:
            exchange = self.exchanges[exchange_preference]
            if exchange.connected:
                try:
                    return await exchange.place_order(order)
                except Exception as e:
                    cprint(f"❌ Error placing order on {exchange_preference}: {str(e)}", "red")
                    return None

        # Auto-select exchange based on symbol mapping or availability
        for name, exchange in self.exchanges.items():
            if not exchange.connected:
                continue

            try:
                return await exchange.place_order(order)
            except Exception as e:
                cprint(f"❌ Error placing order on {name}: {str(e)}", "red")
                continue

        cprint("❌ Failed to place order on any exchange", "red")
        return None

    async def cancel_order(self, order_id: str, symbol: str, exchange: Optional[str] = None) -> bool:
        """Cancel an order, optionally specifying the exchange"""
        if exchange and exchange in self.exchanges:
            ex = self.exchanges[exchange]
            if ex.connected:
                try:
                    return await ex.cancel_order(order_id, symbol)
                except Exception as e:
                    cprint(f"❌ Error canceling order on {exchange}: {str(e)}", "red")
                    return False

        # Try all exchanges if specific exchange not found or failed
        for name, ex in self.exchanges.items():
            if not ex.connected:
                continue

            try:
                if await ex.cancel_order(order_id, symbol):
                    return True
            except Exception:
                continue

        return False

    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data for a symbol from the best available exchange"""
        for name, exchange in self.exchanges.items():
            if not exchange.connected:
                continue

            try:
                data = await exchange.get_market_data(symbol)
                if data:
                    return data
            except Exception:
                continue

        return None

    async def subscribe_market_data(self, symbols: List[str], callback: Callable) -> bool:
        """Subscribe to real-time market data for symbols"""
        self.data_callbacks.append(callback)

        success_count = 0
        for name, exchange in self.exchanges.items():
            if not exchange.connected:
                continue

            try:
                if await exchange.subscribe_realtime_data(symbols, self._handle_market_data):
                    success_count += 1
            except Exception as e:
                cprint(f"❌ Error subscribing to {name}: {str(e)}", "red")

        return success_count > 0

    async def _handle_market_data(self, data: MarketData):
        """Handle incoming market data and distribute to callbacks"""
        # Distribute to all registered callbacks
        for callback in self.data_callbacks:
            try:
                await callback(data)
            except Exception as e:
                cprint(f"❌ Error in market data callback: {str(e)}", "red")

    def update_flux_level(self, level: float):
        """Update the neuro-flux level (0-1)"""
        self._flux_level = max(0.0, min(1.0, level))
        cprint(f"🧠 Neuro-flux level updated to {self._flux_level:.2f}", "cyan")

    @property
    def flux_level(self) -> float:
        """Get current neuro-flux level"""
        return self._flux_level

    @property
    def is_running(self) -> bool:
        """Check if the exchange manager is running"""
        return self._running