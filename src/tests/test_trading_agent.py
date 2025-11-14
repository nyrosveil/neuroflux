"""
🧪 Test Suite for NeuroFlux Trading Agent System
Comprehensive testing of trading agent, exchange connectors, order routing, and trade journaling.

Built with love by Nyros Veil 🚀

Tests Cover:
- TradingAgent core functionality and abstract method implementations
- ExchangeConnector abstract base and CCXT implementation
- OrderRouter intelligent routing and order splitting
- TradeJournal comprehensive trade logging and analytics
- Integration testing of end-to-end trading workflows
"""

import asyncio
import time
import uuid
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional
from decimal import Decimal

# Import the modules to test
from agents.trading.trading_agent import (
    TradingAgent, ExchangeConnector, CCXTExchangeConnector,
    OrderRouter, TradeJournal
)
from agents.trading.types import (
    Order, OrderSide, OrderType, OrderStatus, Position,
    TradingSignal, TradeExecution, MarketData, AssetClass,
    SignalType
)
from agents.trading.base_trading_agent import BaseTradingAgent


class TestExchangeConnector:
    """Test suite for ExchangeConnector abstract base class."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock ExchangeConnector for testing."""
        connector = Mock(spec=ExchangeConnector)
        connector.exchange_id = "test_exchange"
        connector.connected = False
        connector.last_heartbeat = 0
        return connector

    def test_initialization(self):
        """Test ExchangeConnector initialization."""
        connector = ExchangeConnector("test_exchange", "api_key", "secret")
        assert connector.exchange_id == "test_exchange"
        assert connector.api_key == "api_key"
        assert connector.api_secret == "secret"
        assert not connector.connected
        assert connector.last_heartbeat == 0

    @pytest.mark.asyncio
    async def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        connector = ExchangeConnector("test_exchange")

        with pytest.raises(NotImplementedError):
            await connector.connect()

        with pytest.raises(NotImplementedError):
            await connector.disconnect()

        with pytest.raises(NotImplementedError):
            await connector.get_balance("BTC")

        with pytest.raises(NotImplementedError):
            await connector.place_order(Order())

        with pytest.raises(NotImplementedError):
            await connector.cancel_order("order_id")

        with pytest.raises(NotImplementedError):
            await connector.get_order_status("order_id")

        with pytest.raises(NotImplementedError):
            await connector.get_open_orders()

        with pytest.raises(NotImplementedError):
            await connector.get_positions()

        with pytest.raises(NotImplementedError):
            await connector.get_market_data("BTC/USDT")


class TestCCXTExchangeConnector:
    """Test suite for CCXT-based exchange connector."""

    @pytest.fixture
    def connector(self):
        """Create a CCXTExchangeConnector instance."""
        return CCXTExchangeConnector("binance", "test_key", "test_secret")

    @patch('agents.trading.trading_agent.ccxt')
    def test_initialization(self, mock_ccxt, connector):
        """Test CCXTExchangeConnector initialization."""
        mock_ccxt.async_support.binance.assert_called_once()
        assert connector.exchange_id == "binance"
        assert connector.api_key == "test_key"
        assert connector.api_secret == "test_secret"
        assert not connector.connected

    @patch('agents.trading.trading_agent.ccxt')
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_ccxt, connector):
        """Test successful connection to exchange."""
        mock_exchange = Mock()
        mock_exchange.loadMarkets = AsyncMock()
        connector.exchange = mock_exchange

        result = await connector.connect()

        assert result is True
        assert connector.connected is True
        assert connector.last_heartbeat > 0
        mock_exchange.loadMarkets.assert_called_once()

    @patch('agents.trading.trading_agent.ccxt')
    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_ccxt, connector):
        """Test connection failure."""
        mock_exchange = Mock()
        mock_exchange.loadMarkets = AsyncMock(side_effect=Exception("Connection failed"))
        connector.exchange = mock_exchange

        result = await connector.connect()

        assert result is False
        assert connector.connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnecting from exchange."""
        connector.connected = True
        mock_exchange = Mock()
        mock_exchange.close = AsyncMock()
        connector.exchange = mock_exchange

        await connector.disconnect()

        assert connector.connected is False
        mock_exchange.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_balance(self, connector):
        """Test getting account balance."""
        connector.connected = True
        mock_exchange = Mock()
        mock_exchange.fetchBalance = AsyncMock(return_value={
            'BTC': {'free': 1.5, 'used': 0.5, 'total': 2.0},
            'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0}
        })
        connector.exchange = mock_exchange

        balance = await connector.get_balance("BTC")
        assert balance == 1.5

        balance_usdt = await connector.get_balance("USDT")
        assert balance_usdt == 1000.0

    @pytest.mark.asyncio
    async def test_get_balance_not_connected(self, connector):
        """Test getting balance when not connected."""
        connector.connected = False
        balance = await connector.get_balance("BTC")
        assert balance == 0.0

    @pytest.mark.asyncio
    async def test_place_order(self, connector):
        """Test placing an order."""
        connector.connected = True
        mock_exchange = Mock()
        mock_exchange.createOrder = AsyncMock(return_value={'id': '12345'})
        connector.exchange = mock_exchange

        order = Order(
            order_id="test_order",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )

        order_id = await connector.place_order(order)

        assert order_id == "12345"
        assert order.exchange == "binance"
        assert order.status == OrderStatus.OPEN

        mock_exchange.createOrder.assert_called_once_with(
            symbol="BTC/USDT",
            type="limit",
            side="buy",
            amount=1.0,
            price=50000.0
        )

    @pytest.mark.asyncio
    async def test_cancel_order(self, connector):
        """Test canceling an order."""
        connector.connected = True
        mock_exchange = Mock()
        mock_exchange.cancelOrder = AsyncMock()
        connector.exchange = mock_exchange

        result = await connector.cancel_order("12345")

        assert result is True
        mock_exchange.cancelOrder.assert_called_once_with("12345")

    @pytest.mark.asyncio
    async def test_get_order_status(self, connector):
        """Test getting order status."""
        connector.connected = True
        mock_exchange = Mock()
        mock_exchange.fetchOrder = AsyncMock(return_value={
            'id': '12345',
            'status': 'closed'
        })
        connector.exchange = mock_exchange

        status = await connector.get_order_status("12345")

        assert status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_get_market_data(self, connector):
        """Test getting market data."""
        connector.connected = True
        mock_exchange = Mock()
        mock_exchange.fetchTicker = AsyncMock(return_value={
            'last': 50000.0,
            'bid': 49950.0,
            'ask': 50050.0,
            'quoteVolume': 1000000.0
        })
        connector.exchange = mock_exchange

        market_data = await connector.get_market_data("BTC/USDT")

        assert market_data.symbol == "BTC/USDT"
        assert market_data.price == 50000.0
        assert market_data.bid == 49950.0
        assert market_data.ask == 50050.0
        assert market_data.volume == 1000000.0
        assert market_data.exchange == "binance"


class TestOrderRouter:
    """Test suite for OrderRouter class."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock exchange connector."""
        connector = Mock(spec=ExchangeConnector)
        connector.exchange_id = "test_exchange"
        connector.connected = True
        return connector

    @pytest.fixture
    def order_router(self, mock_connector):
        """Create an OrderRouter instance."""
        connectors = {"test_exchange": mock_connector}
        return OrderRouter(connectors)

    @pytest.fixture
    def sample_order(self):
        """Create a sample order."""
        return Order(
            order_id="test_order",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )

    @pytest.mark.asyncio
    async def test_route_order_single_exchange(self, order_router, mock_connector, sample_order):
        """Test routing order with single exchange."""
        connector, reason = await order_router.route_order(sample_order)

        assert connector == mock_connector
        assert "Only one exchange available" in reason

    @pytest.mark.asyncio
    async def test_route_order_multiple_exchanges(self, order_router, mock_connector, sample_order):
        """Test routing order with multiple exchanges."""
        # Add another connector
        mock_connector2 = Mock(spec=ExchangeConnector)
        mock_connector2.exchange_id = "test_exchange2"
        mock_connector2.connected = True
        order_router.connectors["test_exchange2"] = mock_connector2

        # Mock market data for scoring
        mock_connector.get_market_data = AsyncMock(return_value=Mock(spread=0.001))
        mock_connector2.get_market_data = AsyncMock(return_value=Mock(spread=0.002))

        connector, reason = await order_router.route_order(sample_order)

        # Should select the exchange with better score (lower spread)
        assert connector in [mock_connector, mock_connector2]
        assert "Best score" in reason

    @pytest.mark.asyncio
    async def test_split_order_no_split(self, order_router, sample_order):
        """Test order splitting when order is small enough."""
        sample_order.quantity = 500.0  # Below max_order_size (1000.0)
        chunks = await order_router.split_order(sample_order, 1000.0)

        assert len(chunks) == 1
        assert chunks[0] == sample_order

    @pytest.mark.asyncio
    async def test_split_order_with_split(self, order_router, sample_order):
        """Test order splitting when order is too large."""
        sample_order.quantity = 2500.0  # Above max_order_size (1000.0)
        chunks = await order_router.split_order(sample_order, 1000.0)

        assert len(chunks) == 3
        assert chunks[0].quantity == 1000.0
        assert chunks[1].quantity == 1000.0
        assert chunks[2].quantity == 500.0

        # Check chunk IDs
        assert chunks[0].order_id == "test_order_chunk_0"
        assert chunks[1].order_id == "test_order_chunk_1"
        assert chunks[2].order_id == "test_order_chunk_2"


class TestTradeJournal:
    """Test suite for TradeJournal class."""

    @pytest.fixture
    def trade_journal(self):
        """Create a TradeJournal instance."""
        return TradeJournal()

    @pytest.fixture
    def sample_trade(self):
        """Create a sample trade execution."""
        return TradeExecution(
            execution_id="test_execution",
            order=Order(
                order_id="test_order",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1.0
            ),
            executed_quantity=1.0,
            executed_price=50000.0,
            fees=5.0,
            timestamp=time.time()
        )

    @pytest.mark.asyncio
    async def test_record_trade(self, trade_journal, sample_trade):
        """Test recording a trade."""
        await trade_journal.record_trade(sample_trade)

        assert len(trade_journal.trades) == 1
        assert trade_journal.trades[0] == sample_trade

    @pytest.mark.asyncio
    async def test_get_trade_history(self, trade_journal, sample_trade):
        """Test getting trade history."""
        await trade_journal.record_trade(sample_trade)

        # Add another trade
        trade2 = sample_trade
        trade2.execution_id = "test_execution2"
        trade2.timestamp = time.time() + 1
        await trade_journal.record_trade(trade2)

        # Get all trades
        history = await trade_journal.get_trade_history()
        assert len(history) == 2

        # Get trades for specific symbol
        btc_trades = await trade_journal.get_trade_history(symbol="BTC/USDT")
        assert len(btc_trades) == 2

        # Get trades for non-existent symbol
        eth_trades = await trade_journal.get_trade_history(symbol="ETH/USDT")
        assert len(eth_trades) == 0

        # Test time filtering
        recent_trades = await trade_journal.get_trade_history(
            start_time=time.time() - 10,
            end_time=time.time() + 10
        )
        assert len(recent_trades) == 2

    @pytest.mark.asyncio
    async def test_calculate_performance_empty(self, trade_journal):
        """Test performance calculation with no trades."""
        result = await trade_journal.calculate_performance()
        assert result['error'] == 'No trades found'

    @pytest.mark.asyncio
    async def test_calculate_performance(self, trade_journal):
        """Test performance calculation with trades."""
        # Create winning and losing trades
        winning_trade = TradeExecution(
            execution_id="win1",
            order=Order(order_id="o1", symbol="BTC/USDT", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=1.0),
            executed_quantity=1.0,
            executed_price=50000.0,
            fees=5.0,
            timestamp=time.time() - 3600  # 1 hour ago
        )

        losing_trade = TradeExecution(
            execution_id="loss1",
            order=Order(order_id="o2", symbol="BTC/USDT", side=OrderSide.SELL,
                       order_type=OrderType.MARKET, quantity=1.0),
            executed_quantity=1.0,
            executed_price=45000.0,
            fees=4.5,
            timestamp=time.time()  # Now
        )

        await trade_journal.record_trade(winning_trade)
        await trade_journal.record_trade(losing_trade)

        performance = await trade_journal.calculate_performance()

        assert performance['total_trades'] == 2
        assert performance['winning_trades'] == 1  # Simplified logic
        assert performance['losing_trades'] == 1
        assert performance['win_rate'] == 0.5
        assert performance['total_volume'] == 2.0
        assert 'total_fees' in performance
        assert 'trades_per_hour' in performance
        assert 'time_range' in performance


class TestTradingAgent:
    """Test suite for TradingAgent class."""

    @pytest.fixture
    def mock_base_agent(self):
        """Create a mock BaseTradingAgent."""
        base_agent = Mock(spec=BaseTradingAgent)
        base_agent.agent_id = "test_agent"
        base_agent.trading_enabled = True
        base_agent.market_subscriptions = {}
        base_agent.active_orders = {}
        base_agent.market_data_cache = {}
        base_agent.positions = {}
        base_agent.portfolio = Mock()
        base_agent.portfolio.total_value = 100000.0
        base_agent.portfolio.cash_balance = 50000.0
        base_agent.risk_limits = Mock()
        base_agent.risk_limits.max_position_size = 10000.0
        base_agent.risk_limits.max_portfolio_risk = 5.0
        base_agent.start = AsyncMock()
        base_agent.stop = AsyncMock()
        return base_agent

    @pytest.fixture
    def trading_agent(self, mock_base_agent):
        """Create a TradingAgent instance with mocked base class."""
        with patch('agents.trading.trading_agent.BaseTradingAgent', return_value=mock_base_agent):
            agent = TradingAgent("test_agent", {})
            agent._initialize_exchanges = Mock()  # Mock exchange initialization
            return agent

    def test_initialization(self, trading_agent):
        """Test TradingAgent initialization."""
        assert trading_agent.agent_id == "test_agent"
        assert isinstance(trading_agent.trade_journal, TradeJournal)
        assert trading_agent.max_order_size == 1000.0
        assert trading_agent.min_order_split == 100.0

    @pytest.mark.asyncio
    async def test_start_agent(self, trading_agent):
        """Test starting the trading agent."""
        # Mock connectors
        mock_connector = Mock(spec=ExchangeConnector)
        mock_connector.connect = AsyncMock(return_value=True)
        mock_connector.exchange_id = "test_exchange"
        trading_agent.connectors = {"test_exchange": mock_connector}

        await trading_agent.start()

        mock_connector.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_agent(self, trading_agent):
        """Test stopping the trading agent."""
        mock_connector = Mock(spec=ExchangeConnector)
        mock_connector.disconnect = AsyncMock()
        trading_agent.connectors = {"test_exchange": mock_connector}

        await trading_agent.stop()

        mock_connector.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_order(self, trading_agent):
        """Test submitting an order."""
        # Mock order router
        mock_router = Mock(spec=OrderRouter)
        mock_connector = Mock(spec=ExchangeConnector)
        mock_router.route_order = AsyncMock(return_value=(mock_connector, "test_reason"))
        mock_router.split_order = AsyncMock(return_value=[Mock()])
        trading_agent.order_router = mock_router

        # Mock connector
        mock_connector.place_order = AsyncMock(return_value="order_123")
        mock_connector.exchange_id = "test_exchange"

        order = Order(
            order_id="test_order",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )

        order_id = await trading_agent.submit_order(order)

        assert order_id == "order_123"
        mock_router.route_order.assert_called_once()
        mock_router.split_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order(self, trading_agent):
        """Test canceling an order."""
        mock_connector = Mock(spec=ExchangeConnector)
        mock_connector.cancel_order = AsyncMock(return_value=True)
        trading_agent.connectors = {"test_exchange": mock_connector}

        result = await trading_agent.cancel_order("order_123")

        assert result is True
        mock_connector.cancel_order.assert_called_once_with("order_123")

    @pytest.mark.asyncio
    async def test_get_order_status(self, trading_agent):
        """Test getting order status."""
        # Test cached status
        order = Order(order_id="test_order", symbol="BTC/USDT", side=OrderSide.BUY,
                     order_type=OrderType.MARKET, quantity=1.0, status=OrderStatus.OPEN)
        trading_agent.active_orders["test_order"] = order

        status = await trading_agent.get_order_status("test_order")
        assert status == OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_get_positions(self, trading_agent):
        """Test getting positions."""
        mock_connector = Mock(spec=ExchangeConnector)
        mock_connector.get_positions = AsyncMock(return_value={
            "BTC": Position(symbol="BTC", quantity=1.0, avg_price=50000.0,
                          current_price=51000.0, exchange="test_exchange")
        })
        trading_agent.connectors = {"test_exchange": mock_connector}

        positions = await trading_agent.get_positions()

        assert "BTC" in positions
        assert positions["BTC"].quantity == 1.0

    @pytest.mark.asyncio
    async def test_process_trading_signal_buy(self, trading_agent):
        """Test processing a buy trading signal."""
        # Mock market data
        trading_agent.market_data_cache["BTC/USDT"] = MarketData(
            symbol="BTC/USDT", price=50000.0, volume=1000.0, timestamp=time.time(),
            bid=49900.0, ask=50100.0, exchange="test_exchange"
        )

        # Mock order creation and execution
        trading_agent.create_order = AsyncMock(return_value=Mock())
        trading_agent.execute_order = AsyncMock(return_value="order_123")

        signal = TradingSignal(
            signal_id="test_signal",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=time.time()
        )

        await trading_agent.process_trading_signal(signal)

        trading_agent.create_order.assert_called_once()
        trading_agent.execute_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_trading_signal_hold(self, trading_agent):
        """Test processing a hold trading signal."""
        signal = TradingSignal(
            signal_id="test_signal",
            symbol="BTC/USDT",
            signal_type=SignalType.HOLD,
            confidence=0.8,
            timestamp=time.time()
        )

        await trading_agent.process_trading_signal(signal)

        # Should not create or execute any orders
        trading_agent.create_order.assert_not_called()
        trading_agent.execute_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_emergency_stop(self, trading_agent):
        """Test emergency stop functionality."""
        # Mock positions and order creation
        trading_agent.get_positions = AsyncMock(return_value={
            "BTC": Position(symbol="BTC", quantity=1.0, avg_price=50000.0,
                          current_price=51000.0, exchange="test_exchange")
        })
        trading_agent.create_order = AsyncMock(return_value=Mock())
        trading_agent.execute_order = AsyncMock()

        await trading_agent.execute_emergency_stop()

        # Should create and execute a sell order
        trading_agent.create_order.assert_called_once()
        trading_agent.execute_order.assert_called_once()


# Integration tests
@pytest.mark.integration
class TestTradingAgentIntegration:
    """Integration tests for trading agent system."""

    @pytest.fixture
    async def mock_trading_agent(self):
        """Create a mock trading agent for integration testing."""
        agent = Mock(spec=TradingAgent)
        agent.agent_id = "integration_test_agent"
        agent.connectors = {}
        agent.order_router = None
        agent.trade_journal = TradeJournal()
        agent.max_order_size = 1000.0
        return agent

    @pytest.mark.asyncio
    async def test_end_to_end_order_flow(self, mock_trading_agent):
        """Test end-to-end order submission and tracking."""
        # Mock exchange connector
        mock_connector = Mock(spec=ExchangeConnector)
        mock_connector.exchange_id = "test_exchange"
        mock_connector.connected = True
        mock_connector.place_order = AsyncMock(return_value="order_123")
        mock_connector.get_order_status = AsyncMock(return_value=OrderStatus.FILLED)

        mock_trading_agent.connectors = {"test_exchange": mock_connector}
        mock_trading_agent.order_router = OrderRouter(mock_trading_agent.connectors)

        # Create and submit order
        order = Order(
            order_id="integration_test_order",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.5
        )

        # Mock the submit_order method
        async def mock_submit_order(order):
            connector, _ = await mock_trading_agent.order_router.route_order(order)
            order_id = await connector.place_order(order)
            mock_trading_agent.active_orders[order_id] = order
            return order_id

        mock_trading_agent.submit_order = mock_submit_order

        order_id = await mock_trading_agent.submit_order(order)
        assert order_id == "order_123"

        # Check order status
        status = await mock_trading_agent.get_order_status(order_id)
        assert status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_trade_journal_integration(self, mock_trading_agent):
        """Test trade journal integration."""
        # Create and record a trade
        trade = TradeExecution(
            execution_id="integration_trade",
            order=Order(order_id="o1", symbol="BTC/USDT", side=OrderSide.BUY,
                       order_type=OrderType.MARKET, quantity=1.0),
            executed_quantity=1.0,
            executed_price=50000.0,
            fees=5.0,
            timestamp=time.time()
        )

        await mock_trading_agent.trade_journal.record_trade(trade)

        # Verify trade was recorded
        history = await mock_trading_agent.trade_journal.get_trade_history()
        assert len(history) == 1
        assert history[0].execution_id == "integration_trade"

        # Check performance metrics
        performance = await mock_trading_agent.trade_journal.calculate_performance()
        assert performance['total_trades'] == 1
        assert performance['total_volume'] == 1.0
        assert performance['total_fees'] == 5.0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])