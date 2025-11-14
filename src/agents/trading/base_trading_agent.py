"""
🧠 NeuroFlux Base Trading Agent
Abstract base class for all trading agents providing common functionality.

Built with love by Nyros Veil 🚀

Features:
- Market data management and subscriptions
- Order lifecycle management
- Position tracking and P&L calculations
- Trading signal processing
- Risk management integration
- Health monitoring and metrics
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

from ..base_agent import BaseAgent
from .types import (
    MarketData, Order, Position, TradingSignal, Portfolio,
    TradingContext, TradeExecution, AgentHealth, RiskLimits,
    OrderSide, OrderType, OrderStatus, SignalType, AssetClass
)

# Load environment variables
load_dotenv()

class BaseTradingAgent(BaseAgent, ABC):
    """
    Abstract base class for all trading agents.

    Provides common functionality for:
    - Market data handling
    - Order management
    - Position tracking
    - Signal processing
    - Risk management
    - Health monitoring
    """

    def __init__(self, agent_id: str, name: Optional[str] = None, **kwargs):
        """Initialize the trading agent."""
        super().__init__(agent_id, name or f"{self.__class__.__name__}_{agent_id}", **kwargs)

        # Market data management
        self.market_subscriptions: Dict[str, bool] = {}  # symbol -> is_subscribed
        self.market_data_cache: Dict[str, MarketData] = {}  # symbol -> latest data
        self.historical_data_cache: Dict[str, Any] = {}  # symbol -> historical data

        # Order and position management
        self.active_orders: Dict[str, Order] = {}  # order_id -> order
        self.order_history: List[Order] = []
        self.positions: Dict[str, Position] = {}  # symbol -> position
        self.portfolio: Portfolio = Portfolio(
            total_value=0.0,
            cash_balance=0.0,
            positions={}
        )

        # Trading signals
        self.pending_signals: List[TradingSignal] = []
        self.signal_history: List[TradingSignal] = []

        # Risk management
        self.risk_limits: RiskLimits = RiskLimits(
            max_position_size=10000.0,
            max_portfolio_risk=10.0,  # 10% max risk
            max_drawdown=20.0,  # 20% max drawdown
            max_leverage=5.0,
            max_orders_per_minute=10
        )

        # Performance tracking
        self.trading_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

        # Agent-specific state
        self.supported_symbols: List[str] = []
        self.supported_asset_classes: List[AssetClass] = [AssetClass.CRYPTO]
        self.trading_enabled: bool = False

    # ============================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ============================================================================

    @abstractmethod
    async def subscribe_market_data(self, symbols: List[str]) -> None:
        """Subscribe to market data for given symbols."""
        pass

    @abstractmethod
    async def unsubscribe_market_data(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for given symbols."""
        pass

    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> Any:
        """Get historical market data for a symbol."""
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit an order for execution. Returns order_id."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns success status."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get the status of an order."""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""
        pass

    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def process_trading_signal(self, signal: TradingSignal) -> None:
        """Process a trading signal."""
        pass

    @abstractmethod
    async def generate_signal(self, context: TradingContext) -> Optional[TradingSignal]:
        """Generate a trading signal based on context."""
        pass

    # ============================================================================
    # Market Data Management
    # ============================================================================

    async def on_market_data_update(self, data: MarketData) -> None:
        """Handle incoming market data updates."""
        try:
            # Update cache
            self.market_data_cache[data.symbol] = data

            # Update positions with latest prices
            if data.symbol in self.positions:
                self.positions[data.symbol].current_price = data.price
                await self._update_portfolio()

            # Process any pending signals that depend on this data
            await self._process_pending_signals(data)

            # Allow subclasses to handle market data
            await self._on_market_data_received(data)

        except Exception as e:
            cprint(f"❌ Error processing market data for {data.symbol}: {str(e)}", "red")
            await self.report_error("market_data_processing", str(e))

    async def _on_market_data_received(self, data: MarketData) -> None:
        """Hook for subclasses to handle market data. Default implementation does nothing."""
        pass

    async def _process_pending_signals(self, data: MarketData) -> None:
        """Process signals that are waiting for market data."""
        # Remove expired signals
        self.pending_signals = [s for s in self.pending_signals if not s.is_expired()]

        # Process signals for this symbol
        relevant_signals = [s for s in self.pending_signals if s.symbol == data.symbol]
        for signal in relevant_signals:
            try:
                await self.process_trading_signal(signal)
                self.pending_signals.remove(signal)
                self.signal_history.append(signal)
            except Exception as e:
                cprint(f"❌ Error processing signal {signal.signal_id}: {str(e)}", "red")

    # ============================================================================
    # Order Management
    # ============================================================================

    async def create_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                          quantity: float, price: Optional[float] = None,
                          stop_price: Optional[float] = None) -> Order:
        """Create a new order object."""
        order = Order(
            order_id=f"{self.agent_id}_{int(time.time() * 1000)}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        return order

    async def execute_order(self, order: Order) -> Optional[str]:
        """Execute an order and track it."""
        try:
            # Validate order
            if not await self._validate_order(order):
                cprint(f"❌ Order validation failed: {order.order_id}", "red")
                return None

            # Submit order
            order_id = await self.submit_order(order)

            if order_id:
                # Track the order
                self.active_orders[order_id] = order
                cprint(f"📋 Order submitted: {order_id} ({order.symbol} {order.side.value} {order.quantity})", "blue")

                # Update metrics
                self.trading_metrics['total_trades'] += 1

            return order_id

        except Exception as e:
            cprint(f"❌ Error executing order: {str(e)}", "red")
            await self.report_error("order_execution", str(e))
            return None

    async def _validate_order(self, order: Order) -> bool:
        """Validate an order before submission."""
        # Check if symbol is supported
        if order.symbol not in self.supported_symbols:
            return False

        # Check risk limits
        if order.symbol in self.positions:
            current_position = self.positions[order.symbol].quantity
            new_position = current_position + (order.quantity if order.side == OrderSide.BUY else -order.quantity)
            if abs(new_position) > self.risk_limits.max_position_size:
                return False

        # Check portfolio risk
        if not self.risk_limits.can_open_position(order.symbol, order.quantity, self.portfolio.total_value):
            return False

        return True

    async def on_order_update(self, order_id: str, status: OrderStatus,
                             filled_quantity: float = 0.0, average_price: Optional[float] = None) -> None:
        """Handle order status updates."""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.status = status
                order.filled_quantity = filled_quantity
                order.average_fill_price = average_price

                if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    # Move to history
                    self.order_history.append(order)
                    del self.active_orders[order_id]

                    # Update positions if filled
                    if status == OrderStatus.FILLED and average_price:
                        await self._update_position_from_order(order, average_price)

                cprint(f"📊 Order {order_id} updated: {status.value}", "cyan")

        except Exception as e:
            cprint(f"❌ Error processing order update: {str(e)}", "red")
            await self.report_error("order_update", str(e))

    async def _update_position_from_order(self, order: Order, fill_price: float) -> None:
        """Update position based on filled order."""
        try:
            symbol = order.symbol
            quantity_change = order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity

            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=0.0,
                    avg_price=fill_price,
                    current_price=fill_price
                )

            position = self.positions[symbol]

            # Calculate new average price
            total_quantity = position.quantity + quantity_change
            if total_quantity == 0:
                # Position closed
                del self.positions[symbol]
            else:
                # Update position
                total_cost = (position.quantity * position.avg_price) + (quantity_change * fill_price)
                position.avg_price = total_cost / total_quantity
                position.quantity = total_quantity
                position.current_price = fill_price

            await self._update_portfolio()

        except Exception as e:
            cprint(f"❌ Error updating position: {str(e)}", "red")
            await self.report_error("position_update", str(e))

    # ============================================================================
    # Position & Portfolio Management
    # ============================================================================

    async def _update_portfolio(self) -> None:
        """Update portfolio summary."""
        try:
            # Update positions value
            positions_value = sum(pos.market_value for pos in self.positions.values())

            # Update portfolio
            self.portfolio.positions = self.positions.copy()
            self.portfolio.total_value = self.portfolio.cash_balance + positions_value
            self.portfolio.timestamp = time.time()

            # Update trading metrics
            self.trading_metrics['total_pnl'] = self.portfolio.total_pnl

        except Exception as e:
            cprint(f"❌ Error updating portfolio: {str(e)}", "red")
            await self.report_error("portfolio_update", str(e))

    async def calculate_portfolio_pnl(self) -> Dict[str, Any]:
        """Calculate comprehensive P&L metrics."""
        try:
            total_unrealized = self.portfolio.total_pnl
            total_realized = sum(
                pos.unrealized_pnl for pos in self.positions.values()
                if pos.quantity == 0  # Closed positions
            )

            return {
                'unrealized_pnl': total_unrealized,
                'realized_pnl': total_realized,
                'total_pnl': total_unrealized + total_realized,
                'total_pnl_percent': self.portfolio.total_pnl_percent
            }

        except Exception as e:
            cprint(f"❌ Error calculating P&L: {str(e)}", "red")
            return {'error': str(e)}

    # ============================================================================
    # Trading Signal Processing
    # ============================================================================

    async def receive_trading_signal(self, signal: TradingSignal) -> None:
        """Receive and queue a trading signal."""
        try:
            if signal.is_valid():
                self.pending_signals.append(signal)
                cprint(f"📡 Received signal: {signal.signal_type.value} {signal.symbol} (confidence: {signal.confidence:.2f})", "green")

                # Process immediately if we have market data
                if signal.symbol in self.market_data_cache:
                    await self.process_trading_signal(signal)
                    self.pending_signals.remove(signal)
                    self.signal_history.append(signal)
            else:
                cprint(f"⚠️ Invalid/expired signal received: {signal.signal_id}", "yellow")

        except Exception as e:
            cprint(f"❌ Error receiving signal: {str(e)}", "red")
            await self.report_error("signal_reception", str(e))

    async def get_signal_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[TradingSignal]:
        """Get trading signal history."""
        signals = self.signal_history
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        return signals[-limit:] if limit > 0 else signals

    # ============================================================================
    # Risk Management
    # ============================================================================

    async def check_risk_limits(self) -> Dict[str, Any]:
        """Check if current positions are within risk limits."""
        try:
            results = {
                'position_sizes': True,
                'portfolio_risk': True,
                'drawdown': True,
                'leverage': True
            }

            # Check position sizes
            for symbol, position in self.positions.items():
                if abs(position.quantity) > self.risk_limits.max_position_size:
                    results['position_sizes'] = False
                    break

            # Check portfolio risk
            portfolio_risk_pct = (self.portfolio.total_pnl / self.portfolio.total_value) * 100
            if portfolio_risk_pct > self.risk_limits.max_portfolio_risk:
                results['portfolio_risk'] = False

            # Check drawdown (simplified)
            if self.trading_metrics['max_drawdown'] > self.risk_limits.max_drawdown:
                results['drawdown'] = False

            return results

        except Exception as e:
            cprint(f"❌ Error checking risk limits: {str(e)}", "red")
            return {'error': str(e)}

    async def update_risk_limits(self, limits: Dict[str, Any]) -> None:
        """Update risk management limits."""
        try:
            for key, value in limits.items():
                if hasattr(self.risk_limits, key):
                    setattr(self.risk_limits, key, value)

            cprint(f"🔒 Risk limits updated: {limits}", "blue")

        except Exception as e:
            cprint(f"❌ Error updating risk limits: {str(e)}", "red")
            await self.report_error("risk_limits_update", str(e))

    # ============================================================================
    # Health Monitoring & Metrics
    # ============================================================================

    async def get_health_status(self) -> AgentHealth:
        """Get comprehensive agent health status."""
        try:
            # Check market data subscriptions
            active_subscriptions = sum(1 for subscribed in self.market_subscriptions.values() if subscribed)

            # Check positions
            total_positions = len(self.positions)

            # Check pending signals
            pending_signals_count = len(self.pending_signals)

            # Determine status
            issues = []
            status = "healthy"

            if active_subscriptions == 0:
                issues.append("No active market data subscriptions")
                status = "degraded"

            if pending_signals_count > 10:
                issues.append(f"High number of pending signals: {pending_signals_count}")
                status = "degraded"

            if not self.trading_enabled:
                issues.append("Trading is disabled")
                status = "degraded"

            # Risk check
            risk_status = await self.check_risk_limits()
            risk_issues = [k for k, v in risk_status.items() if not v and k != 'error']
            if risk_issues:
                issues.extend([f"Risk limit breached: {issue}" for issue in risk_issues])
                status = "unhealthy"

            metrics = {
                'active_subscriptions': active_subscriptions,
                'total_positions': total_positions,
                'pending_signals': pending_signals_count,
                'portfolio_value': self.portfolio.total_value,
                'total_pnl': self.portfolio.total_pnl,
                'win_rate': self.trading_metrics.get('win_rate', 0.0)
            }

            return AgentHealth(
                agent_id=self.agent_id,
                status=status,
                metrics=metrics,
                issues=issues
            )

        except Exception as e:
            cprint(f"❌ Error getting health status: {str(e)}", "red")
            return AgentHealth(
                agent_id=self.agent_id,
                status="error",
                issues=[str(e)]
            )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            metrics = self.trading_metrics.copy()

            # Add additional metrics
            metrics.update({
                'active_positions': len(self.positions),
                'pending_orders': len(self.active_orders),
                'portfolio_value': self.portfolio.total_value,
                'cash_balance': self.portfolio.cash_balance,
                'total_return_pct': self.portfolio.total_pnl_percent,
                'uptime_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            })

            return metrics

        except Exception as e:
            cprint(f"❌ Error getting performance metrics: {str(e)}", "red")
            return {'error': str(e)}

    # ============================================================================
    # Trading Control
    # ============================================================================

    async def enable_trading(self) -> None:
        """Enable trading operations."""
        self.trading_enabled = True
        cprint(f"🟢 Trading enabled for {self.agent_id}", "green")

    async def disable_trading(self) -> None:
        """Disable trading operations."""
        self.trading_enabled = False
        cprint(f"🔴 Trading disabled for {self.agent_id}", "red")

    async def emergency_stop(self) -> None:
        """Emergency stop - cancel all orders and disable trading."""
        try:
            cprint(f"🚨 Emergency stop initiated for {self.agent_id}", "red")

            # Cancel all active orders
            for order_id in list(self.active_orders.keys()):
                try:
                    await self.cancel_order(order_id)
                except Exception as e:
                    cprint(f"❌ Error cancelling order {order_id}: {str(e)}", "red")

            # Disable trading
            await self.disable_trading()

            # Unsubscribe from all market data
            await self.unsubscribe_market_data(list(self.market_subscriptions.keys()))

            cprint(f"✅ Emergency stop completed for {self.agent_id}", "green")

        except Exception as e:
            cprint(f"❌ Error during emergency stop: {str(e)}", "red")
            await self.report_error("emergency_stop", str(e))

    # ============================================================================
    # Lifecycle Management
    # ============================================================================

    async def start(self) -> None:
        """Start the trading agent."""
        await super().start()

        # Initialize trading state
        self.trading_enabled = True

        # Subscribe to initial market data if symbols are configured
        if self.supported_symbols:
            await self.subscribe_market_data(self.supported_symbols)

        cprint(f"🚀 Trading agent {self.agent_id} started", "green")

    async def stop(self) -> None:
        """Stop the trading agent."""
        try:
            # Emergency cleanup
            await self.emergency_stop()

            # Clear caches
            self.market_data_cache.clear()
            self.historical_data_cache.clear()
            self.pending_signals.clear()

            await super().stop()

            cprint(f"⏹️ Trading agent {self.agent_id} stopped", "green")

        except Exception as e:
            cprint(f"❌ Error stopping trading agent: {str(e)}", "red")

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def is_symbol_supported(self, symbol: str) -> bool:
        """Check if a symbol is supported by this agent."""
        return symbol in self.supported_symbols

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        return self.supported_symbols.copy()

    async def report_error(self, error_type: str, message: str) -> None:
        """Report an error with context."""
        error_data = {
            'agent_id': self.agent_id,
            'error_type': error_type,
            'message': message,
            'timestamp': time.time(),
            'active_positions': len(self.positions),
            'pending_orders': len(self.active_orders)
        }

        # Send error report via communication bus if available
        try:
            if hasattr(self, 'communication_bus') and self.communication_bus:
                await self.communication_bus.broadcast_message(
                    sender_id=self.agent_id,
                    topic="agent_error",
                    payload=error_data
                )
        except Exception:
            pass  # Ignore communication errors during error reporting

        # Log locally
        cprint(f"❌ Agent error ({error_type}): {message}", "red")
