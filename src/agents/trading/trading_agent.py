"""
🧠 NeuroFlux Trading Agent
Concrete implementation of the trading agent for order execution and position management.

Built with love by Nyros Veil 🚀

Features:
- Multi-exchange order execution
- Smart order routing and splitting
- Real-time position management
- Comprehensive trade journaling
- Performance analytics and reporting
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

from .base_trading_agent import BaseTradingAgent
from ..base_agent import BaseAgent
from .types import (
    Order, OrderSide, OrderType, OrderStatus, Position,
    TradingSignal, TradeExecution, MarketData, AssetClass
)

# Load environment variables
load_dotenv()

class ExchangeConnector:
    """
    Abstract base class for exchange connectivity.
    Provides unified interface for different exchange implementations.
    """

    def __init__(self, exchange_id: str, api_key: str = None, api_secret: str = None):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.connected = False
        self.last_heartbeat = 0

    async def connect(self) -> bool:
        """Establish connection to exchange."""
        raise NotImplementedError

    async def disconnect(self) -> None:
        """Close connection to exchange."""
        raise NotImplementedError

    async def get_balance(self, asset: str) -> float:
        """Get account balance for specific asset."""
        raise NotImplementedError

    async def place_order(self, order: Order) -> str:
        """Place an order and return order ID."""
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        raise NotImplementedError

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        raise NotImplementedError

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        raise NotImplementedError

    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        raise NotImplementedError

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data."""
        raise NotImplementedError

class CCXTExchangeConnector(ExchangeConnector):
    """
    CCXT-based exchange connector for unified API access.
    """

    def __init__(self, exchange_id: str, api_key: str = None, api_secret: str = None):
        super().__init__(exchange_id, api_key, api_secret)
        self.exchange = None
        self._initialize_exchange()

    def _initialize_exchange(self):
        """Initialize CCXT exchange instance."""
        try:
            import ccxt.async_support as ccxt

            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # or 'future' for derivatives
                }
            })
        except ImportError:
            cprint(f"⚠️ CCXT not available for {self.exchange_id}", "yellow")
        except Exception as e:
            cprint(f"❌ Failed to initialize {self.exchange_id}: {str(e)}", "red")

    async def connect(self) -> bool:
        """Connect to exchange."""
        if not self.exchange:
            return False

        try:
            await self.exchange.loadMarkets()
            self.connected = True
            self.last_heartbeat = time.time()
            cprint(f"✅ Connected to {self.exchange_id}", "green")
            return True
        except Exception as e:
            cprint(f"❌ Failed to connect to {self.exchange_id}: {str(e)}", "red")
            return False

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self.exchange:
            await self.exchange.close()
        self.connected = False
        cprint(f"🔌 Disconnected from {self.exchange_id}", "yellow")

    async def get_balance(self, asset: str) -> float:
        """Get account balance."""
        if not self.connected or not self.exchange:
            return 0.0

        try:
            balance = await self.exchange.fetchBalance()
            return float(balance.get(asset, {}).get('free', 0.0))
        except Exception as e:
            cprint(f"❌ Failed to get balance for {asset}: {str(e)}", "red")
            return 0.0

    async def place_order(self, order: Order) -> str:
        """Place an order."""
        if not self.connected or not self.exchange:
            raise Exception(f"Not connected to {self.exchange_id}")

        try:
            # Convert order to CCXT format
            ccxt_order = {
                'symbol': order.symbol,
                'type': order.order_type.value,
                'side': order.side.value,
                'amount': order.quantity,
            }

            if order.price:
                ccxt_order['price'] = order.price

            if order.stop_price:
                ccxt_order['stopPrice'] = order.stop_price

            # Place order
            result = await self.exchange.createOrder(**ccxt_order)

            # Update order with exchange details
            order.exchange = self.exchange_id
            order.status = OrderStatus.OPEN

            return str(result['id'])

        except Exception as e:
            cprint(f"❌ Failed to place order: {str(e)}", "red")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.connected or not self.exchange:
            return False

        try:
            await self.exchange.cancelOrder(order_id)
            return True
        except Exception as e:
            cprint(f"❌ Failed to cancel order {order_id}: {str(e)}", "red")
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if not self.connected or not self.exchange:
            return OrderStatus.PENDING

        try:
            order_info = await self.exchange.fetchOrder(order_id)
            status = order_info.get('status', 'open')

            # Map CCXT status to our enum
            status_mapping = {
                'open': OrderStatus.OPEN,
                'closed': OrderStatus.FILLED,
                'canceled': OrderStatus.CANCELLED,
                'expired': OrderStatus.EXPIRED,
                'rejected': OrderStatus.REJECTED,
            }

            return status_mapping.get(status, OrderStatus.PENDING)

        except Exception as e:
            cprint(f"❌ Failed to get order status: {str(e)}", "red")
            return OrderStatus.PENDING

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        if not self.connected or not self.exchange:
            return []

        try:
            orders = await self.exchange.fetchOpenOrders(symbol)
            return [self._ccxt_order_to_order(order) for order in orders]
        except Exception as e:
            cprint(f"❌ Failed to get open orders: {str(e)}", "red")
            return []

    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        # For spot trading, positions are just balances
        # For futures, this would return actual positions
        positions = {}

        try:
            balance = await self.exchange.fetchBalance()
            for asset, data in balance.items():
                if asset != 'info' and data.get('total', 0) > 0:
                    positions[asset] = Position(
                        symbol=f"{asset}/USDT",  # Assuming USDT pairs
                        quantity=float(data.get('total', 0)),
                        avg_price=0.0,  # Would need to track this separately
                        current_price=0.0,  # Would need market data
                        exchange=self.exchange_id
                    )
        except Exception as e:
            cprint(f"❌ Failed to get positions: {str(e)}", "red")

        return positions

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data."""
        if not self.connected or not self.exchange:
            raise Exception(f"Not connected to {self.exchange_id}")

        try:
            ticker = await self.exchange.fetchTicker(symbol)
            return MarketData(
                symbol=symbol,
                price=float(ticker['last']),
                volume=float(ticker['quoteVolume']),
                timestamp=time.time(),
                bid=float(ticker['bid']),
                ask=float(ticker['ask']),
                exchange=self.exchange_id
            )
        except Exception as e:
            cprint(f"❌ Failed to get market data for {symbol}: {str(e)}", "red")
            raise

    def _ccxt_order_to_order(self, ccxt_order: Dict[str, Any]) -> Order:
        """Convert CCXT order format to our Order format."""
        return Order(
            order_id=str(ccxt_order['id']),
            symbol=ccxt_order['symbol'],
            side=OrderSide(ccxt_order['side']),
            order_type=OrderType(ccxt_order['type']),
            quantity=float(ccxt_order['amount']),
            price=float(ccxt_order.get('price', 0)) if ccxt_order.get('price') else None,
            status=OrderStatus.OPEN,  # Assume open if fetched
            exchange=self.exchange_id
        )

class OrderRouter:
    """
    Intelligent order routing system.
    Routes orders to best exchange based on liquidity, fees, and execution quality.
    """

    def __init__(self, connectors: Dict[str, ExchangeConnector]):
        self.connectors = connectors
        self.routing_rules = {
            'liquidity_weight': 0.4,
            'fee_weight': 0.3,
            'latency_weight': 0.3
        }

    async def route_order(self, order: Order) -> Tuple[ExchangeConnector, str]:
        """
        Route order to best exchange.
        Returns (connector, reason)
        """
        if len(self.connectors) == 1:
            connector = list(self.connectors.values())[0]
            return connector, "Only one exchange available"

        # Score each exchange
        scores = {}
        for exchange_id, connector in self.connectors.items():
            if not connector.connected:
                continue

            try:
                score = await self._score_exchange(connector, order)
                scores[exchange_id] = score
            except Exception as e:
                cprint(f"⚠️ Failed to score {exchange_id}: {str(e)}", "yellow")

        if not scores:
            # Fallback to first available
            for connector in self.connectors.values():
                if connector.connected:
                    return connector, "Fallback routing"

        # Select best exchange
        best_exchange = max(scores.items(), key=lambda x: x[1])[0]
        connector = self.connectors[best_exchange]

        return connector, f"Best score: {scores[best_exchange]:.2f}"

    async def _score_exchange(self, connector: ExchangeConnector, order: Order) -> float:
        """Score an exchange for order execution."""
        score = 0.0

        try:
            # Liquidity score (based on order book depth)
            market_data = await connector.get_market_data(order.symbol)
            spread = market_data.spread
            liquidity_score = 1.0 / (1.0 + spread)  # Lower spread = higher score
            score += liquidity_score * self.routing_rules['liquidity_weight']

            # Fee score (simplified - would need actual fee data)
            fee_score = 0.8  # Placeholder
            score += fee_score * self.routing_rules['fee_weight']

            # Latency score (simplified - would need ping measurements)
            latency_score = 0.9  # Placeholder
            score += latency_score * self.routing_rules['latency_weight']

        except Exception as e:
            cprint(f"⚠️ Error scoring {connector.exchange_id}: {str(e)}", "yellow")
            score = 0.1  # Low score for problematic exchanges

        return score

    async def split_order(self, order: Order, max_size: float) -> List[Order]:
        """Split large order into smaller chunks."""
        if order.quantity <= max_size:
            return [order]

        chunks = []
        remaining_quantity = order.quantity

        while remaining_quantity > 0:
            chunk_size = min(remaining_quantity, max_size)
            chunk_order = Order(
                order_id=f"{order.order_id}_chunk_{len(chunks)}",
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=chunk_size,
                price=order.price,
                stop_price=order.stop_price
            )
            chunks.append(chunk_order)
            remaining_quantity -= chunk_size

        return chunks

class TradeJournal:
    """
    Comprehensive trade journaling and analytics system.
    """

    def __init__(self, storage_path: str = "trades.db"):
        self.storage_path = storage_path
        self.trades: List[TradeExecution] = []
        self._lock = asyncio.Lock()

    async def record_trade(self, trade: TradeExecution) -> None:
        """Record a completed trade."""
        async with self._lock:
            self.trades.append(trade)
            cprint(f"📝 Recorded trade: {trade.execution_id} ({trade.order.symbol} {trade.executed_quantity}@{trade.executed_price})", "blue")

            # TODO: Persist to database
            await self._persist_trade(trade)

    async def get_trade_history(self,
                              symbol: Optional[str] = None,
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None,
                              limit: int = 1000) -> List[TradeExecution]:
        """Get trade history with optional filters."""
        async with self._lock:
            trades = self.trades.copy()

            # Apply filters
            if symbol:
                trades = [t for t in trades if t.order.symbol == symbol]

            if start_time:
                trades = [t for t in trades if t.timestamp >= start_time]

            if end_time:
                trades = [t for t in trades if t.timestamp <= end_time]

            # Sort by timestamp (newest first)
            trades.sort(key=lambda x: x.timestamp, reverse=True)

            return trades[:limit]

    async def calculate_performance(self,
                                  symbol: Optional[str] = None,
                                  start_time: Optional[float] = None,
                                  end_time: Optional[float] = None) -> Dict[str, Any]:
        """Calculate performance metrics."""
        trades = await self.get_trade_history(symbol, start_time, end_time, limit=10000)

        if not trades:
            return {'error': 'No trades found'}

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.total_cost > 0])  # Simplified
        losing_trades = total_trades - winning_trades

        total_volume = sum(t.executed_quantity for t in trades)
        total_fees = sum(t.fees for t in trades)

        # Calculate time-weighted metrics
        if len(trades) > 1:
            time_span = trades[0].timestamp - trades[-1].timestamp
            trades_per_hour = (total_trades / time_span) * 3600 if time_span > 0 else 0
        else:
            trades_per_hour = 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_volume': total_volume,
            'total_fees': total_fees,
            'trades_per_hour': trades_per_hour,
            'avg_trade_size': total_volume / total_trades if total_trades > 0 else 0,
            'time_range': {
                'start': trades[-1].timestamp if trades else None,
                'end': trades[0].timestamp if trades else None
            }
        }

    async def _persist_trade(self, trade: TradeExecution) -> None:
        """Persist trade to storage (database/file)."""
        # TODO: Implement actual persistence
        # For now, just keep in memory
        pass

class TradingAgent(BaseTradingAgent):
    """
    Concrete trading agent implementation.
    Handles order execution, position management, and trade journaling.
    """

    def __init__(self, agent_id: str, exchanges_config: Dict[str, Dict[str, str]] = None, **kwargs):
        super().__init__(agent_id, **kwargs)

        # Exchange configuration
        self.exchanges_config = exchanges_config or {}
        self.connectors: Dict[str, ExchangeConnector] = {}
        self.order_router = None

        # Trading components
        self.trade_journal = TradeJournal()

        # Supported symbols (will be populated from exchanges)
        self.supported_symbols = []

        # Order splitting configuration
        self.max_order_size = 1000.0  # Maximum order size before splitting
        self.min_order_split = 100.0   # Minimum chunk size

        # Initialize exchanges
        self._initialize_exchanges()

    def _initialize_exchanges(self):
        """Initialize exchange connectors."""
        for exchange_id, config in self.exchanges_config.items():
            try:
                connector = CCXTExchangeConnector(
                    exchange_id=exchange_id,
                    api_key=config.get('api_key'),
                    api_secret=config.get('api_secret')
                )
                self.connectors[exchange_id] = connector
                cprint(f"🔌 Initialized connector for {exchange_id}", "blue")
            except Exception as e:
                cprint(f"❌ Failed to initialize {exchange_id}: {str(e)}", "red")

        # Initialize order router
        if self.connectors:
            self.order_router = OrderRouter(self.connectors)

    async def start(self) -> None:
        """Start the trading agent."""
        # Initialize agent first
        if not await self._initialize_agent():
            raise Exception(f"Failed to initialize trading agent {self.agent_id}")

        # Call parent start (which is not async in base agent)
        super(BaseTradingAgent, self).start()  # Call BaseAgent.start directly

        # Connect to exchanges
        connected_count = 0
        for exchange_id, connector in self.connectors.items():
            if await connector.connect():
                connected_count += 1

        if connected_count == 0:
            cprint("⚠️ No exchanges connected - trading agent in limited mode", "yellow")
        else:
            cprint(f"✅ Connected to {connected_count}/{len(self.connectors)} exchanges", "green")

        # Update supported symbols
        await self._update_supported_symbols()

    async def stop(self) -> None:
        """Stop the trading agent."""
        # Disconnect from exchanges
        for connector in self.connectors.values():
            await connector.disconnect()

        # Call cleanup
        self._cleanup_agent()

        # Call parent stop (not async)
        super(BaseTradingAgent, self).stop()

    async def _update_supported_symbols(self):
        """Update list of supported symbols from connected exchanges."""
        all_symbols = set()

        for connector in self.connectors.values():
            if connector.connected and hasattr(connector, 'exchange'):
                try:
                    markets = list(connector.exchange.markets.keys())
                    all_symbols.update(markets)
                except Exception as e:
                    cprint(f"⚠️ Failed to get markets from {connector.exchange_id}: {str(e)}", "yellow")

        self.supported_symbols = list(all_symbols)
        cprint(f"📊 Updated supported symbols: {len(self.supported_symbols)} total", "blue")

    # ============================================================================
    # Abstract Method Implementations
    # ============================================================================

    async def subscribe_market_data(self, symbols: List[str]) -> None:
        """Subscribe to market data (simplified - would need WebSocket connections)."""
        for symbol in symbols:
            if symbol in self.market_subscriptions:
                continue

            self.market_subscriptions[symbol] = True
            cprint(f"📡 Subscribed to market data for {symbol}", "blue")

            # Start periodic market data fetching
            asyncio.create_task(self._periodic_market_data_fetch(symbol))

    async def unsubscribe_market_data(self, symbols: List[str]) -> None:
        """Unsubscribe from market data."""
        for symbol in symbols:
            if symbol in self.market_subscriptions:
                self.market_subscriptions[symbol] = False
                cprint(f"📡 Unsubscribed from market data for {symbol}", "blue")

    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> Any:
        """Get historical market data."""
        # Find a connected exchange that supports this symbol
        for connector in self.connectors.values():
            if connector.connected and hasattr(connector, 'exchange'):
                try:
                    # Use CCXT OHLCV data
                    ohlcv = await connector.exchange.fetchOHLCV(symbol, timeframe, limit=limit)
                    return ohlcv  # Returns list of [timestamp, open, high, low, close, volume]
                except Exception as e:
                    cprint(f"⚠️ Failed to get historical data from {connector.exchange_id}: {str(e)}", "yellow")

        return []

    async def submit_order(self, order: Order) -> str:
        """Submit an order for execution."""
        # Route order to best exchange
        if not self.order_router:
            raise Exception("No order router available")

        connector, reason = await self.order_router.route_order(order)
        cprint(f"🎯 Routed order to {connector.exchange_id}: {reason}", "blue")

        # Split order if too large
        order_chunks = await self.order_router.split_order(order, self.max_order_size)

        if len(order_chunks) > 1:
            cprint(f"✂️ Split order into {len(order_chunks)} chunks", "blue")

        # Submit chunks
        order_ids = []
        for chunk in order_chunks:
            try:
                order_id = await connector.place_order(chunk)
                order_ids.append(order_id)

                # Track the order
                self.active_orders[order_id] = chunk

            except Exception as e:
                cprint(f"❌ Failed to submit order chunk: {str(e)}", "red")
                # Continue with other chunks

        # Return primary order ID (first chunk)
        return order_ids[0] if order_ids else ""

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        # Find which exchange has this order
        for connector in self.connectors.values():
            if connector.connected:
                try:
                    success = await connector.cancel_order(order_id)
                    if success:
                        if order_id in self.active_orders:
                            self.active_orders[order_id].status = OrderStatus.CANCELLED
                        return True
                except Exception as e:
                    cprint(f"⚠️ Failed to cancel on {connector.exchange_id}: {str(e)}", "yellow")

        return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        # Check local tracking first
        if order_id in self.active_orders:
            return self.active_orders[order_id].status

        # Query exchanges
        for connector in self.connectors.values():
            if connector.connected:
                try:
                    status = await connector.get_order_status(order_id)
                    if status != OrderStatus.PENDING:  # Found it
                        return status
                except Exception as e:
                    cprint(f"⚠️ Failed to get status from {connector.exchange_id}: {str(e)}", "yellow")

        return OrderStatus.PENDING

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        all_orders = []

        for connector in self.connectors.values():
            if connector.connected:
                try:
                    orders = await connector.get_open_orders(symbol)
                    all_orders.extend(orders)
                except Exception as e:
                    cprint(f"⚠️ Failed to get orders from {connector.exchange_id}: {str(e)}", "yellow")

        return all_orders

    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions across all exchanges."""
        all_positions = {}

        for connector in self.connectors.values():
            if connector.connected:
                try:
                    positions = await connector.get_positions()
                    # Merge positions (simplified - would need proper netting)
                    for symbol, position in positions.items():
                        if symbol in all_positions:
                            # Net positions across exchanges
                            existing = all_positions[symbol]
                            net_quantity = existing.quantity + position.quantity
                            if net_quantity != 0:
                                all_positions[symbol] = Position(
                                    symbol=symbol,
                                    quantity=net_quantity,
                                    avg_price=(existing.avg_price * existing.quantity +
                                             position.avg_price * position.quantity) / abs(net_quantity),
                                    current_price=max(existing.current_price, position.current_price),
                                    exchange="NETTED"
                                )
                            else:
                                del all_positions[symbol]
                        else:
                            all_positions[symbol] = position

                except Exception as e:
                    cprint(f"⚠️ Failed to get positions from {connector.exchange_id}: {str(e)}", "yellow")

        return all_positions

    async def process_trading_signal(self, signal: TradingSignal) -> None:
        """Process a trading signal by creating and submitting orders."""
        try:
            cprint(f"🎯 Processing signal: {signal.signal_type.value} {signal.symbol} (confidence: {signal.confidence:.2f})", "green")

            # Validate signal
            if not signal.is_valid():
                cprint(f"⚠️ Invalid signal: {signal.signal_id}", "yellow")
                return

            # Create order based on signal
            if signal.signal_type in [signal.signal_type.BUY, signal.signal_type.STRONG_BUY]:
                side = OrderSide.BUY
            elif signal.signal_type in [signal.signal_type.SELL, signal.signal_type.STRONG_SELL]:
                side = OrderSide.SELL
            else:  # HOLD
                cprint(f"📊 Signal indicates HOLD - no action taken", "blue")
                return

            # Get current market data for pricing
            if signal.symbol in self.market_data_cache:
                market_data = self.market_data_cache[signal.symbol]
                price = market_data.price
            else:
                cprint(f"⚠️ No market data available for {signal.symbol}", "yellow")
                return

            # Calculate order size based on signal confidence and risk limits
            order_size = await self._calculate_order_size(signal, price)

            if order_size <= 0:
                cprint(f"⚠️ Order size too small: {order_size}", "yellow")
                return

            # Create order
            order = await self.create_order(
                symbol=signal.symbol,
                side=side,
                order_type=OrderType.MARKET,  # Use market orders for signals
                quantity=order_size,
                price=None  # Market order
            )

            # Submit order
            order_id = await self.execute_order(order)

            if order_id:
                cprint(f"✅ Signal executed: {signal.signal_id} -> Order {order_id}", "green")
            else:
                cprint(f"❌ Failed to execute signal: {signal.signal_id}", "red")

        except Exception as e:
            cprint(f"❌ Error processing signal {signal.signal_id}: {str(e)}", "red")
            await self.report_error("signal_processing", str(e))

    async def generate_signal(self, context: Any) -> Optional[TradingSignal]:
        """Generate trading signals (placeholder - would use ML models)."""
        # This would typically use the ModelFactory from Phase 3
        # For now, return None (signals come from analysis agents)
        return None

    # ============================================================================
    # Trading Agent Specific Methods
    # ============================================================================

    async def _calculate_order_size(self, signal: TradingSignal, price: float) -> float:
        """Calculate appropriate order size based on signal and risk limits."""
        try:
            # Base size on signal confidence
            base_size = 100.0  # Base order size
            confidence_multiplier = signal.confidence

            order_size = base_size * confidence_multiplier

            # Apply risk limits
            max_position = self.risk_limits.max_position_size
            current_position = 0.0

            if signal.symbol in self.positions:
                current_position = abs(self.positions[signal.symbol].quantity)

            available_position = max_position - current_position
            order_size = min(order_size, available_position)

            # Check portfolio risk
            portfolio_value = self.portfolio.total_value
            if portfolio_value > 0:
                risk_percentage = (order_size * price / portfolio_value) * 100
                max_risk_pct = self.risk_limits.max_portfolio_risk
                if risk_percentage > max_risk_pct:
                    order_size = (max_risk_pct / 100) * portfolio_value / price

            return max(0.0, order_size)

        except Exception as e:
            cprint(f"❌ Error calculating order size: {str(e)}", "red")
            return 0.0

    async def _periodic_market_data_fetch(self, symbol: str):
        """Periodically fetch market data for subscribed symbols."""
        while self.market_subscriptions.get(symbol, False) and self.trading_enabled:
            try:
                # Find a connected exchange
                for connector in self.connectors.values():
                    if connector.connected:
                        market_data = await connector.get_market_data(symbol)
                        await self.on_market_data_update(market_data)
                        break

            except Exception as e:
                cprint(f"⚠️ Failed to fetch market data for {symbol}: {str(e)}", "yellow")

            await asyncio.sleep(1.0)  # Fetch every second

    async def get_trading_performance(self) -> Dict[str, Any]:
        """Get comprehensive trading performance metrics."""
        try:
            # Get trade journal metrics
            journal_metrics = await self.trade_journal.calculate_performance()

            # Add position metrics
            positions = await self.get_positions()
            total_exposure = sum(abs(pos.market_value) for pos in positions.values())

            # Add portfolio metrics
            pnl_metrics = await self.calculate_portfolio_pnl()

            return {
                'journal': journal_metrics,
                'positions': {
                    'total_positions': len(positions),
                    'total_exposure': total_exposure,
                    'positions': {k: v.__dict__ for k, v in positions.items()}
                },
                'portfolio': {
                    'total_value': self.portfolio.total_value,
                    'cash_balance': self.portfolio.cash_balance,
                    'total_pnl': pnl_metrics.get('total_pnl', 0),
                    'total_pnl_percent': pnl_metrics.get('total_pnl_percent', 0)
                },
                'risk': {
                    'risk_limits': self.risk_limits.__dict__,
                    'risk_check': await self.check_risk_limits()
                }
            }

        except Exception as e:
            cprint(f"❌ Error getting trading performance: {str(e)}", "red")
            return {'error': str(e)}

    async def execute_emergency_stop(self) -> None:
        """Execute emergency stop - close all positions."""
        try:
            cprint(f"🚨 Executing emergency stop for {self.agent_id}", "red")

            positions = await self.get_positions()

            for symbol, position in positions.items():
                if position.quantity != 0:
                    # Create closing order
                    side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                    quantity = abs(position.quantity)

                    order = await self.create_order(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=quantity
                    )

                    # Execute immediately
                    await self.execute_order(order)
                    cprint(f"🛑 Emergency close: {symbol} {quantity} {side.value}", "red")

            cprint(f"✅ Emergency stop completed", "green")

        except Exception as e:
            cprint(f"❌ Error during emergency stop: {str(e)}", "red")
            await self.report_error("emergency_stop", str(e))

    # ============================================================================
    # Abstract Method Implementations from BaseAgent
    # ============================================================================

    async def _initialize_agent(self) -> bool:
        """Agent-specific initialization logic."""
        try:
            # Initialize exchange connections
            self._initialize_exchanges()

            # Initialize trading components
            self.trade_journal = TradeJournal()

            cprint(f"✅ Trading agent {self.agent_id} initialized successfully", "green")
            return True

        except Exception as e:
            cprint(f"❌ Failed to initialize trading agent {self.agent_id}: {str(e)}", "red")
            return False

    def _execute_agent_cycle(self):
        """Execute one cycle of agent logic."""
        # Trading agents are primarily reactive (responding to signals)
        # This method can be used for periodic maintenance tasks
        pass

    def _cleanup_agent(self):
        """Agent-specific cleanup logic."""
        try:
            # Close all positions if configured
            if hasattr(self, 'emergency_stop_on_shutdown') and self.emergency_stop_on_shutdown:
                asyncio.create_task(self.execute_emergency_stop())

            # Disconnect from exchanges
            for connector in self.connectors.values():
                asyncio.create_task(connector.disconnect())

            cprint(f"🧹 Trading agent {self.agent_id} cleaned up", "blue")

        except Exception as e:
            cprint(f"❌ Error during cleanup: {str(e)}", "red")
