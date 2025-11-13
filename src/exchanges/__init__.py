"""
🧠 NeuroFlux Exchanges Package
Unified multi-exchange trading infrastructure.

Built with love by Nyros Veil 🚀

This package provides:
- ExchangeAdapter: Abstract base class for exchange implementations
- ExchangeManager: Main orchestrator for multi-exchange operations
- Individual exchange adapters (HyperLiquid, Extended, Aster)
"""

from .base_exchange import (
    ExchangeAdapter, Order, Position, MarketData,
    OrderSide, OrderType, PositionSide
)
from .exchange_manager import ExchangeManager

__all__ = [
    'ExchangeAdapter', 'ExchangeManager',
    'Order', 'Position', 'MarketData',
    'OrderSide', 'OrderType', 'PositionSide'
]

# Try to import adapters (will fail gracefully if dependencies not available)
try:
    from .hyperliquid_adapter import HyperLiquidAdapter
    __all__.append('HyperLiquidAdapter')
except ImportError:
    pass

try:
    from .extended_adapter import ExtendedAdapter
    __all__.append('ExtendedAdapter')
except ImportError:
    pass

try:
    from .aster_adapter import AsterAdapter
    __all__.append('AsterAdapter')
except ImportError:
    pass