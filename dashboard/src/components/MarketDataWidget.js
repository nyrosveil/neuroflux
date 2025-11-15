import React, { useState, useEffect, useRef } from 'react';
import './MarketDataWidget.css';

function MarketDataWidget({ socket }) {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const [marketData, setMarketData] = useState({});
  const [orderBook, setOrderBook] = useState({ bids: [], asks: [] });
  const [priceHistory, setPriceHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [selectedExchange, setSelectedExchange] = useState('binance');
  const [isLoading, setIsLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState('connecting');

  const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT'];
  const exchanges = ['binance', 'bybit', 'kucoin'];

  // WebSocket listeners for market data
  useEffect(() => {
    if (!socket) return;

    const handleMarketData = (data) => {
      try {
        const marketUpdate = data.data;
        setMarketData(prev => ({
          ...prev,
          [marketUpdate.symbol]: {
            ...prev[marketUpdate.symbol],
            ...marketUpdate,
            lastUpdate: new Date().toISOString()
          }
        }));

        // Update price history for chart
        if (marketUpdate.price) {
          setPriceHistory(prev => {
            const newPoint = {
              time: new Date().toLocaleTimeString(),
              price: marketUpdate.price,
              volume: marketUpdate.volume || 0
            };
            return [...prev.slice(-49), newPoint]; // Keep last 50 points
          });
        }

        setIsLoading(false);
      } catch (error) {
        console.error('Error processing market data:', error);
      }
    };

    const handleOrderBook = (data) => {
      try {
        setOrderBook(data.data);
      } catch (error) {
        console.error('Error processing order book:', error);
      }
    };

    const handleConnect = () => setConnectionStatus('connected');
    const handleDisconnect = () => setConnectionStatus('disconnected');
    const handleConnectError = () => setConnectionStatus('error');

    socket.on('market_data', handleMarketData);
    socket.on('orderbook_update', handleOrderBook);
    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('connect_error', handleConnectError);

    // Set initial connection status
    if (socket.connected) {
      setConnectionStatus('connected');
    }

    return () => {
      socket.off('market_data', handleMarketData);
      socket.off('orderbook_update', handleOrderBook);
      socket.off('connect', handleConnect);
      socket.off('disconnect', handleDisconnect);
      socket.off('connect_error', handleConnectError);
    };
  }, [socket]);

  // Price alert system
  const addPriceAlert = (type, price) => {
    const alert = {
      id: Date.now(),
      symbol: selectedSymbol,
      type,
      price,
      created: new Date().toISOString(),
      triggered: false
    };
    setAlerts(prev => [...prev, alert]);
  };

  const removeAlert = (alertId) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  // Check alerts against current price
  useEffect(() => {
    const currentData = marketData[selectedSymbol];
    if (!currentData || !currentData.price) return;

    const currentPrice = currentData.price;
    setAlerts(prev => prev.map(alert => {
      if (alert.symbol !== selectedSymbol || alert.triggered) return alert;

      const triggered = alert.type === 'above' ? currentPrice >= alert.price :
                       alert.type === 'below' ? currentPrice <= alert.price : false;

      if (triggered) {
        // Could play sound or show notification here
        console.log(`🚨 Price alert triggered: ${selectedSymbol} ${alert.type} ${alert.price}`);
      }

      return { ...alert, triggered };
    }));
  }, [marketData, selectedSymbol]);

  const currentData = marketData[selectedSymbol] || {};
  const priceChange = currentData.change_24h || 0;
  const priceChangePercent = currentData.change_percentage || 0;

  if (isLoading) {
    return (
      <div className="market-data-widget loading">
        <h3>📈 Market Data</h3>
        <div className="connection-status">
          <span
            className="status-indicator"
            style={{ backgroundColor: connectionStatus === 'connected' ? '#6bcf7f' : '#ffd93d' }}
          ></span>
          <span className="status-text">{connectionStatus}</span>
        </div>
        <p>Loading market data...</p>
      </div>
    );
  }

  return (
    <div className="market-data-widget">
      <div className="widget-header">
        <h3>📈 Market Data</h3>
        <div className="connection-status">
          <span
            className="status-indicator"
            style={{ backgroundColor: connectionStatus === 'connected' ? '#6bcf7f' : '#ffd93d' }}
          ></span>
          <span className="status-text">{connectionStatus}</span>
        </div>
      </div>

      {/* Symbol and Exchange Selection */}
      <div className="controls-section">
        <div className="control-group">
          <label>Symbol:</label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
          >
            {symbols.map(symbol => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Exchange:</label>
          <select
            value={selectedExchange}
            onChange={(e) => setSelectedExchange(e.target.value)}
          >
            {exchanges.map(exchange => (
              <option key={exchange} value={exchange}>
                {exchange.charAt(0).toUpperCase() + exchange.slice(1)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Price Display */}
      <div className="price-display">
        <div className="main-price">
          <div className="price-value">
            ${currentData.price ? currentData.price.toLocaleString() : 'N/A'}
          </div>
          <div className={`price-change ${priceChange >= 0 ? 'positive' : 'negative'}`}>
            {priceChange >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%
            ({priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)})
          </div>
        </div>

        <div className="price-details">
          <div className="detail-item">
            <span className="label">24h High:</span>
            <span className="value">${currentData.high_24h ? currentData.high_24h.toLocaleString() : 'N/A'}</span>
          </div>
          <div className="detail-item">
            <span className="label">24h Low:</span>
            <span className="value">${currentData.low_24h ? currentData.low_24h.toLocaleString() : 'N/A'}</span>
          </div>
          <div className="detail-item">
            <span className="label">Volume:</span>
            <span className="value">{currentData.volume ? currentData.volume.toLocaleString() : 'N/A'}</span>
          </div>
        </div>
      </div>

      {/* Price Chart */}
      <div className="price-chart-section">
        <h4>Price History (Last 50 Updates)</h4>
        <div className="mini-chart">
          {priceHistory.length > 0 ? (
            <div className="price-points">
              {priceHistory.map((point, index) => {
                const maxPrice = Math.max(...priceHistory.map(p => p.price));
                const minPrice = Math.min(...priceHistory.map(p => p.price));
                const range = maxPrice - minPrice;
                const height = range > 0 ? ((point.price - minPrice) / range) * 100 : 50;

                return (
                  <div key={index} className="price-point" title={`${point.time}: $${point.price}`}>
                    <div
                      className="price-bar"
                      style={{ height: `${height}%` }}
                    ></div>
                    <div className="volume-bar" style={{ height: `${Math.min(point.volume / 1000000 * 20, 20)}%` }}></div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="no-data">No price data available</div>
          )}
        </div>
      </div>

      {/* Order Book */}
      <div className="orderbook-section">
        <h4>Order Book</h4>
        <div className="orderbook">
          <div className="orderbook-side">
            <h5>Bids (Buy)</h5>
            <div className="orders">
              {orderBook.bids && orderBook.bids.slice(0, 10).map((bid, index) => (
                <div key={index} className="order-row bid">
                  <span className="price">${bid.price ? bid.price.toFixed(2) : 'N/A'}</span>
                  <span className="amount">{bid.amount ? bid.amount.toFixed(4) : 'N/A'}</span>
                  <span className="total">${bid.total ? bid.total.toFixed(2) : 'N/A'}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="orderbook-side">
            <h5>Asks (Sell)</h5>
            <div className="orders">
              {orderBook.asks && orderBook.asks.slice(0, 10).map((ask, index) => (
                <div key={index} className="order-row ask">
                  <span className="price">${ask.price ? ask.price.toFixed(2) : 'N/A'}</span>
                  <span className="amount">{ask.amount ? ask.amount.toFixed(4) : 'N/A'}</span>
                  <span className="total">${ask.total ? ask.total.toFixed(2) : 'N/A'}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Price Alerts */}
      <div className="alerts-section">
        <h4>🚨 Price Alerts</h4>
        <div className="alert-controls">
          <button
            className="alert-btn above"
            onClick={() => addPriceAlert('above', currentData.price * 1.01)}
          >
            Alert Above (+1%)
          </button>
          <button
            className="alert-btn below"
            onClick={() => addPriceAlert('below', currentData.price * 0.99)}
          >
            Alert Below (-1%)
          </button>
        </div>

        <div className="alerts-list">
          {alerts.filter(alert => alert.symbol === selectedSymbol).map(alert => (
            <div key={alert.id} className={`alert-item ${alert.triggered ? 'triggered' : ''}`}>
              <span className="alert-symbol">{alert.symbol}</span>
              <span className="alert-condition">
                {alert.type} ${alert.price.toFixed(2)}
              </span>
              {alert.triggered && <span className="alert-triggered">🚨 TRIGGERED</span>}
              <button
                className="remove-alert"
                onClick={() => removeAlert(alert.id)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Market Stats */}
      <div className="market-stats">
        <div className="stat-item">
          <span className="stat-label">Spread:</span>
          <span className="stat-value">
            {orderBook.spread ? `$${orderBook.spread.toFixed(2)}` : 'N/A'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Liquidity:</span>
          <span className="stat-value">
            {orderBook.liquidity ? orderBook.liquidity.toLocaleString() : 'N/A'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Last Update:</span>
          <span className="stat-value">
            {currentData.lastUpdate ? new Date(currentData.lastUpdate).toLocaleTimeString() : 'Never'}
          </span>
        </div>
      </div>
    </div>
  );
}

export default MarketDataWidget;