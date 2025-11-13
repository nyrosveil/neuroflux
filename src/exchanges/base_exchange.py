"""
🧠 NeuroFlux Exchange Manager
Unified interface for multi-exchange trading with neuro-flux enhancements.

Built with love by Nyros Veil 🚀

This module provides a unified interface for trading across multiple exchanges,
with real-time data streaming, order management, and neuro-flux adaptive behavior.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from decimal import Decimal


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Unified order representation across exchanges"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: str = "pending"
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class Position:
    """Unified position representation across exchanges"""
    symbol: str
    side: PositionSide
    amount: Decimal
    entry_price: Decimal
    current_price: Decimal
    pnl: Decimal
    pnl_percentage: Decimal
    exchange: str
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class MarketData:
    """Unified market data representation"""
    symbol: str
    price: Decimal
    bid: Decimal
    ask: Decimal
    volume_24h: Decimal
    change_24h: Decimal
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ExchangeAdapter(ABC):
    """
    Abstract base class for exchange adapters.
    All exchange implementations must inherit from this class.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.connected = False
        self._flux_sensitivity = config.get('flux_sensitivity', 0.8)

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the exchange"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to the exchange"""
        pass

    @abstractmethod
    async def get_balance(self, asset: Optional[str] = None) -> Dict[str, Decimal]:
        """Get account balance for specific asset or all assets"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all open positions"""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> Optional[str]:
        """Place a new order, return order ID if successful"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get status of a specific order"""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol"""
        pass

    @abstractmethod
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for a symbol"""
        pass

    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Get historical OHLCV data"""
        pass

    @abstractmethod
    async def subscribe_realtime_data(self, symbols: List[str], callback: Callable) -> bool:
        """Subscribe to real-time market data"""
        pass

    @abstractmethod
    async def unsubscribe_realtime_data(self, symbols: List[str]) -> bool:
        """Unsubscribe from real-time market data"""
        pass

    def adapt_to_flux(self, base_value: float, market_volatility: float) -> float:
        """
        Apply neuro-flux adaptation to trading parameters based on market conditions.

        Args:
            base_value: Original parameter value
            market_volatility: Current market volatility (0-1 scale)

        Returns:
            Adapted value with flux adjustments
        """
        # Higher volatility = more conservative adjustments
        flux_factor = 1.0 - (market_volatility * self._flux_sensitivity)

        # Apply neural network-style adaptation
        adapted_value = base_value * flux_factor

        return max(adapted_value, 0.1)  # Minimum floor to prevent zero values