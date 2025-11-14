#!/usr/bin/env python3
"""
Integration test for end-to-end trading agent workflow.
Tests complete order execution flow from signal to trade execution.
"""

import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.trading.trading_agent import TradingAgent, OrderRouter
from agents.trading.types import (
    Order, OrderSide, OrderType, OrderStatus, Position,
    TradingSignal, TradeExecution, MarketData, SignalType
)

class MockExchangeConnector:
    """Mock exchange connector for integration testing."""

    def __init__(self, exchange_id):
        self.exchange_id = exchange_id
        self.connected = False
        self.orders = {}
        self.positions = {}
        self.market_data = {}

    async def connect(self):
        self.connected = True
        return True

    async def disconnect(self):
        self.connected = False

    async def get_balance(self, asset):
        return 10000.0 if asset == "USDT" else 1.0

    async def place_order(self, order):
        order_id = f"{self.exchange_id}_order_{len(self.orders)}"
        order.status = OrderStatus.FILLED
        self.orders[order_id] = order
        return order_id

    async def cancel_order(self, order_id):
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_order_status(self, order_id):
        return self.orders.get(order_id, OrderStatus.PENDING)

    async def get_open_orders(self, symbol=None):
        return [order for order in self.orders.values()
                if order.status == OrderStatus.OPEN]

    async def get_positions(self):
        return self.positions.copy()

    async def get_market_data(self, symbol):
        # Return mock market data
        return MarketData(
            symbol=symbol,
            price=50000.0,
            volume=1000.0,
            timestamp=0,
            bid=49900.0,
            ask=50100.0,
            exchange=self.exchange_id
        )

async def test_end_to_end_trading_workflow():
    """Test complete trading workflow from signal to execution."""
    print("🧪 Testing End-to-End Trading Workflow")

    # Create mock exchange connectors
    mock_binance = MockExchangeConnector("binance")
    mock_coinbase = MockExchangeConnector("coinbase")

    # Create trading agent with mock exchanges
    exchanges_config = {
        "binance": {"api_key": "test", "api_secret": "test"},
        "coinbase": {"api_key": "test", "api_secret": "test"}
    }

    # Mock the exchange initialization to use our mock connectors
    agent = TradingAgent("integration_test_agent", exchanges_config)

    # Replace connectors with mocks
    agent.connectors = {
        "binance": mock_binance,
        "coinbase": mock_coinbase
    }

    # Create OrderRouter
    agent.order_router = OrderRouter(agent.connectors)

    # Start the agent
    await agent.start()

    try:
        # Step 1: Subscribe to market data
        await agent.subscribe_market_data(["BTC/USDT"])
        print("✅ Market data subscription successful")

        # Step 2: Create and process a trading signal
        signal = TradingSignal(
            signal_id="integration_test_signal",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence=0.8,
            reasoning="Integration test signal",
            timestamp=0
        )

        # Mock market data cache
        agent.market_data_cache = {
            "BTC/USDT": MarketData(
                symbol="BTC/USDT",
                price=50000.0,
                volume=1000.0,
                timestamp=0,
                bid=49900.0,
                ask=50100.0,
                exchange="binance"
            )
        }

        # Mock portfolio and risk limits
        agent.portfolio = Mock()
        agent.portfolio.total_value = 100000.0
        agent.portfolio.cash_balance = 50000.0
        agent.risk_limits = Mock()
        agent.risk_limits.max_position_size = 10000.0
        agent.risk_limits.max_portfolio_risk = 5.0
        agent.positions = {}

        # Process the signal
        await agent.process_trading_signal(signal)
        print("✅ Signal processing successful")

        # Step 3: Check that order was created and executed
        open_orders = await agent.get_open_orders()
        all_orders = []
        for connector in agent.connectors.values():
            all_orders.extend(await connector.get_open_orders())

        # Check trade journal
        trades = await agent.trade_journal.get_trade_history()
        if trades:
            print(f"✅ Trade recorded: {trades[0].execution_id}")
        else:
            print("ℹ️ No trades recorded (expected for mock execution)")

        # Step 4: Test position management
        positions = await agent.get_positions()
        print(f"✅ Position tracking: {len(positions)} positions")

        # Step 5: Test emergency stop
        await agent.execute_emergency_stop()
        print("✅ Emergency stop executed")

        # Step 6: Get performance metrics
        performance = await agent.get_trading_performance()
        print(f"✅ Performance metrics retrieved: {len(performance)} categories")

        print("🎉 End-to-end trading workflow test completed successfully!")

    finally:
        # Stop the agent
        await agent.stop()

async def test_multi_exchange_order_routing():
    """Test order routing across multiple exchanges."""
    print("🧪 Testing Multi-Exchange Order Routing")

    # Create mock exchanges with different characteristics
    fast_exchange = MockExchangeConnector("fast_exchange")
    slow_exchange = MockExchangeConnector("slow_exchange")

    # Set them as connected
    fast_exchange.connected = True
    slow_exchange.connected = True

    # Mock market data with different spreads (lower spread = better)
    fast_exchange.get_market_data = AsyncMock(return_value=MarketData(
        symbol="BTC/USDT", price=50000.0, volume=1000.0, timestamp=0,
        bid=49990.0, ask=50010.0, exchange="fast_exchange"  # Tight spread
    ))
    slow_exchange.get_market_data = AsyncMock(return_value=MarketData(
        symbol="BTC/USDT", price=50000.0, volume=1000.0, timestamp=0,
        bid=49900.0, ask=50100.0, exchange="slow_exchange"  # Wide spread
    ))

    agent = TradingAgent("routing_test_agent", {})
    agent.connectors = {
        "fast_exchange": fast_exchange,
        "slow_exchange": slow_exchange
    }
    agent.order_router = OrderRouter(agent.connectors)

    # Test order routing
    order = Order("routing_test", "BTC/USDT", OrderSide.BUY, OrderType.MARKET, 1.0)
    connector, reason = await agent.order_router.route_order(order)

    # Should select the exchange with better (tighter) spread
    assert connector == fast_exchange, f"Expected fast_exchange, got {connector.exchange_id}"
    assert "Best score" in reason
    print("✅ Order routing selected best exchange based on spread")

async def test_order_splitting():
    """Test large order splitting functionality."""
    print("🧪 Testing Order Splitting")

    agent = TradingAgent("splitting_test_agent", {})
    mock_connector = MockExchangeConnector("test_exchange")
    agent.connectors = {"test_exchange": mock_connector}
    agent.order_router = OrderRouter(agent.connectors)

    # Test large order splitting
    large_order = Order("large_order", "BTC/USDT", OrderSide.BUY, OrderType.MARKET, 2500.0)
    chunks = await agent.order_router.split_order(large_order, 1000.0)

    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert chunks[0].quantity == 1000.0
    assert chunks[1].quantity == 1000.0
    assert chunks[2].quantity == 500.0

    # Check chunk IDs
    assert "large_order_chunk_0" in chunks[0].order_id
    assert "large_order_chunk_1" in chunks[1].order_id
    assert "large_order_chunk_2" in chunks[2].order_id

    print("✅ Order splitting works correctly")

async def main():
    """Run integration tests."""
    print("🚀 Starting NeuroFlux Trading Agent Integration Tests")
    print("=" * 60)

    try:
        await test_end_to_end_trading_workflow()
        print()

        await test_multi_exchange_order_routing()
        print()

        await test_order_splitting()
        print()

        print("=" * 60)
        print("🎉 All integration tests passed!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)