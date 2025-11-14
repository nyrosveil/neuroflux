#!/usr/bin/env python3
"""
Simple test runner for trading agent components.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.trading.trading_agent import TradingAgent, ExchangeConnector, CCXTExchangeConnector, OrderRouter, TradeJournal
from agents.trading.types import Order, OrderSide, OrderType, OrderStatus, Position, TradingSignal, TradeExecution, MarketData

async def test_exchange_connector():
    """Test ExchangeConnector abstract class."""
    print("Testing ExchangeConnector...")

    # Test abstract methods raise NotImplementedError
    connector = ExchangeConnector("test_exchange")

    try:
        await connector.connect()
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass

    try:
        await connector.get_balance("BTC")
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass

    print("✅ ExchangeConnector abstract methods work correctly")

async def test_ccxt_exchange_connector():
    """Test CCXTExchangeConnector."""
    print("Testing CCXTExchangeConnector...")

    connector = CCXTExchangeConnector("binance", "test_key", "test_secret")
    assert connector.exchange_id == "binance"
    assert connector.api_key == "test_key"
    assert connector.api_secret == "test_secret"

    print("✅ CCXTExchangeConnector initialization works")

async def test_order_router():
    """Test OrderRouter."""
    print("Testing OrderRouter...")

    # Mock connector
    class MockConnector(ExchangeConnector):
        def __init__(self, exchange_id):
            super().__init__(exchange_id)
            self.connected = True

        async def connect(self): return True
        async def disconnect(self): pass
        async def get_balance(self, asset): return 1000.0
        async def place_order(self, order): return "order_123"
        async def cancel_order(self, order_id): return True
        async def get_order_status(self, order_id): return OrderStatus.FILLED
        async def get_open_orders(self, symbol=None): return []
        async def get_positions(self): return {}
        async def get_market_data(self, symbol): return MarketData(symbol, 50000.0, 1000.0, 0, 49900.0, 50100.0, self.exchange_id)

    connectors = {"test_exchange": MockConnector("test_exchange")}
    router = OrderRouter(connectors)

    order = Order("test_order", "BTC/USDT", OrderSide.BUY, OrderType.MARKET, 1.0)

    # Test routing
    selected_connector, reason = await router.route_order(order)
    assert selected_connector == connectors["test_exchange"]
    assert "Only one exchange available" in reason

    # Test order splitting
    large_order = Order("large_order", "BTC/USDT", OrderSide.BUY, OrderType.MARKET, 2500.0)
    chunks = await router.split_order(large_order, 1000.0)
    assert len(chunks) == 3
    assert chunks[0].quantity == 1000.0
    assert chunks[1].quantity == 1000.0
    assert chunks[2].quantity == 500.0

    print("✅ OrderRouter works correctly")

async def test_trade_journal():
    """Test TradeJournal."""
    print("Testing TradeJournal...")

    journal = TradeJournal()

    # Create test trade
    order = Order("test_order", "BTC/USDT", OrderSide.BUY, OrderType.MARKET, 1.0)
    trade = TradeExecution("trade_1", order, 1.0, 50000.0, 5.0, 1000000.0)

    # Record trade
    await journal.record_trade(trade)
    assert len(journal.trades) == 1

    # Get trade history
    history = await journal.get_trade_history()
    assert len(history) == 1
    assert history[0].execution_id == "trade_1"

    # Calculate performance
    performance = await journal.calculate_performance()
    assert performance['total_trades'] == 1
    assert performance['total_volume'] == 1.0
    assert performance['total_fees'] == 5.0

    print("✅ TradeJournal works correctly")

async def test_trading_agent():
    """Test TradingAgent basic functionality."""
    print("Testing TradingAgent...")

    # Mock the base class to avoid complex initialization
    from unittest.mock import Mock

    # Create a mock base agent
    mock_base = Mock()
    mock_base.agent_id = "test_agent"
    mock_base.trading_enabled = True
    mock_base.market_subscriptions = {}
    mock_base.active_orders = {}
    mock_base.market_data_cache = {}
    mock_base.positions = {}
    mock_base.portfolio = Mock()
    mock_base.portfolio.total_value = 100000.0
    mock_base.portfolio.cash_balance = 50000.0
    mock_base.risk_limits = Mock()
    mock_base.risk_limits.max_position_size = 10000.0
    mock_base.risk_limits.max_portfolio_risk = 5.0

    # Patch the base class
    import agents.trading.trading_agent
    original_base = agents.trading.trading_agent.BaseTradingAgent
    agents.trading.trading_agent.BaseTradingAgent = lambda *args, **kwargs: mock_base

    try:
        agent = TradingAgent("test_agent", {})
        assert agent.agent_id == "test_agent"
        assert isinstance(agent.trade_journal, TradeJournal)

        print("✅ TradingAgent initialization works")
    finally:
        # Restore original
        agents.trading.trading_agent.BaseTradingAgent = original_base

async def main():
    """Run all tests."""
    print("🧪 Running Trading Agent Component Tests")
    print("=" * 50)

    try:
        await test_exchange_connector()
        await test_ccxt_exchange_connector()
        await test_order_router()
        await test_trade_journal()
        await test_trading_agent()

        print("=" * 50)
        print("✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)