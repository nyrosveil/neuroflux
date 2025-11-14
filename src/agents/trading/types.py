"""
🧠 NeuroFlux Trading Agent Types
Core data structures and type definitions for trading agents.

Built with love by Nyros Veil 🚀
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class SignalType(Enum):
    """Trading signal type enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class AssetClass(Enum):
    """Asset class enumeration."""
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"

@dataclass
class MarketData:
    """Real-time market data structure."""
    symbol: str
    price: float
    volume: float
    timestamp: float
    bid: float
    ask: float
    exchange: str
    asset_class: AssetClass = AssetClass.CRYPTO

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

@dataclass
class Order:
    """Trading order structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = field(init=False)
    average_fill_price: Optional[float] = None
    fees: float = 0.0
    exchange: str = ""
    client_order_id: Optional[str] = None

    def __post_init__(self):
        self.remaining_quantity = self.quantity

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIAL]

@dataclass
class Position:
    """Portfolio position structure."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    exchange: str = ""
    asset_class: AssetClass = AssetClass.CRYPTO

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis."""
        return self.quantity * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized profit/loss percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100

@dataclass
class TradingSignal:
    """Trading signal structure."""
    signal_id: str
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    agent_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiry: Optional[float] = None  # Signal expiry timestamp

    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.expiry is None:
            return False
        return datetime.now().timestamp() > self.expiry

    def is_valid(self) -> bool:
        """Check if signal is valid."""
        return not self.is_expired() and 0.0 <= self.confidence <= 1.0

@dataclass
class Portfolio:
    """Portfolio summary structure."""
    total_value: float
    cash_balance: float
    positions: Dict[str, Position]
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    @property
    def positions_value(self) -> float:
        """Calculate total value of positions."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def total_pnl_percent(self) -> float:
        """Calculate total P&L percentage."""
        invested_value = self.cash_balance + self.positions_value - self.total_pnl
        if invested_value == 0:
            return 0.0
        return (self.total_pnl / invested_value) * 100

@dataclass
class TradingContext:
    """Context information for trading decisions."""
    symbol: str
    market_data: MarketData
    portfolio: Portfolio
    recent_signals: List[TradingSignal] = field(default_factory=list)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

@dataclass
class TradeExecution:
    """Trade execution result structure."""
    execution_id: str
    order: Order
    executed_quantity: float
    executed_price: float
    fees: float
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    exchange: str = ""

    @property
    def total_cost(self) -> float:
        """Calculate total cost including fees."""
        return (self.executed_quantity * self.executed_price) + self.fees

@dataclass
class AgentHealth:
    """Agent health status structure."""
    agent_id: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_update: float = field(default_factory=lambda: datetime.now().timestamp())
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        return self.status == "healthy"

@dataclass
class RiskLimits:
    """Risk management limits structure."""
    max_position_size: float
    max_portfolio_risk: float  # Max % of portfolio at risk
    max_drawdown: float  # Max drawdown percentage
    max_leverage: float
    max_orders_per_minute: int
    restricted_symbols: List[str] = field(default_factory=list)

    def can_open_position(self, symbol: str, position_size: float, portfolio_value: float) -> bool:
        """Check if position can be opened within risk limits."""
        if symbol in self.restricted_symbols:
            return False
        if position_size > self.max_position_size:
            return False
        risk_percentage = (position_size / portfolio_value) * 100
        return risk_percentage <= self.max_portfolio_risk