import React, { useState, useEffect } from 'react';
import './MarketAnalysisWidget.css';

const MarketAnalysisWidget = ({ symbol = 'BTC' }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAnalysis();
    const interval = setInterval(fetchAnalysis, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, [symbol]);

  const fetchAnalysis = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/market-analysis/${symbol}`);
      if (!response.ok) throw new Error('Failed to fetch analysis');
      const data = await response.json();
      setAnalysis(data);
      setError(null);
    } catch (err) {
      setError(err.message);
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'buy': return '#00ff88';
      case 'sell': return '#ff4444';
      default: return '#888888';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#00ff88';
    if (confidence >= 0.6) return '#ffff44';
    if (confidence >= 0.4) return '#ffaa44';
    return '#ff4444';
  };

  if (loading) {
    return (
      <div className="market-analysis-widget">
        <div className="widget-header">
          <h3>🧠 Market Analysis - {symbol}</h3>
        </div>
        <div className="loading">Loading analysis...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="market-analysis-widget">
        <div className="widget-header">
          <h3>🧠 Market Analysis - {symbol}</h3>
        </div>
        <div className="error">Error: {error}</div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="market-analysis-widget">
        <div className="widget-header">
          <h3>🧠 Market Analysis - {symbol}</h3>
        </div>
        <div className="no-data">No analysis data available</div>
      </div>
    );
  }

  const { technical_analysis, market_regime, risk_metrics, trading_signals } = analysis;

  return (
    <div className="market-analysis-widget">
      <div className="widget-header">
        <h3>🧠 Market Analysis - {symbol}</h3>
        <div className="last-updated">
          Updated: {new Date(analysis.timestamp).toLocaleTimeString()}
        </div>
      </div>

      <div className="analysis-content">
        {/* Trading Signals */}
        <div className="analysis-section">
          <h4>📊 Trading Signals</h4>
          <div className="signal-display">
            <div
              className="main-signal"
              style={{ backgroundColor: getSignalColor(trading_signals.overall_signal) }}
            >
              {trading_signals.overall_signal.toUpperCase()}
            </div>
            <div
              className="confidence-bar"
              style={{ backgroundColor: getConfidenceColor(trading_signals.confidence) }}
            >
              Confidence: {(trading_signals.confidence * 100).toFixed(1)}%
            </div>
          </div>

          {trading_signals.entry_price && (
            <div className="trade-levels">
              <div>Entry: ${trading_signals.entry_price.toFixed(2)}</div>
              <div>Stop Loss: ${trading_signals.stop_loss.toFixed(2)}</div>
              <div>Take Profit: ${trading_signals.take_profit.toFixed(2)}</div>
              <div>Risk/Reward: {trading_signals.risk_reward_ratio.toFixed(2)}</div>
            </div>
          )}
        </div>

        {/* Technical Indicators */}
        <div className="analysis-section">
          <h4>📈 Technical Indicators</h4>
          <div className="indicators-grid">
            <div className="indicator">
              <span className="label">RSI:</span>
              <span className={`value ${technical_analysis.rsi < 30 ? 'oversold' : technical_analysis.rsi > 70 ? 'overbought' : 'neutral'}`}>
                {technical_analysis.rsi.toFixed(1)}
              </span>
            </div>

            <div className="indicator">
              <span className="label">MACD:</span>
              <span className={`value ${technical_analysis.macd_signal === 'bullish' ? 'bullish' : technical_analysis.macd_signal === 'bearish' ? 'bearish' : 'neutral'}`}>
                {technical_analysis.macd_signal}
              </span>
            </div>

            <div className="indicator">
              <span className="label">Trend:</span>
              <span className={`value ${technical_analysis.moving_avg_trend.includes('up') ? 'bullish' : technical_analysis.moving_avg_trend.includes('down') ? 'bearish' : 'neutral'}`}>
                {technical_analysis.moving_avg_trend.replace('_', ' ')}
              </span>
            </div>

            <div className="indicator">
              <span className="label">Volume:</span>
              <span className="value">{technical_analysis.volume_analysis.replace('_', ' ')}</span>
            </div>
          </div>
        </div>

        {/* Market Regime */}
        {market_regime && (
          <div className="analysis-section">
            <h4>🌊 Market Regime</h4>
            <div className="regime-info">
              <div className="regime-name">{market_regime.name.replace('_', ' ').toUpperCase()}</div>
              <div className="regime-metrics">
                <div>Volatility: {(market_regime.volatility * 100).toFixed(1)}%</div>
                <div>Trend Strength: {(market_regime.trend_strength * 100).toFixed(1)}%</div>
                <div>Liquidity: {(market_regime.liquidity * 100).toFixed(1)}%</div>
                <div>Risk Level: <span className={`risk-${market_regime.risk_level}`}>{market_regime.risk_level.toUpperCase()}</span></div>
              </div>
            </div>
          </div>
        )}

        {/* Risk Metrics */}
        {risk_metrics && (
          <div className="analysis-section">
            <h4>⚠️ Risk Metrics</h4>
            <div className="risk-grid">
              <div className="metric">
                <span className="label">Sharpe Ratio:</span>
                <span className="value">{risk_metrics.sharpe_ratio.toFixed(2)}</span>
              </div>

              <div className="metric">
                <span className="label">Max Drawdown:</span>
                <span className="value">{(risk_metrics.max_drawdown * 100).toFixed(2)}%</span>
              </div>

              <div className="metric">
                <span className="label">VaR (95%):</span>
                <span className="value">{(risk_metrics.value_at_risk_95 * 100).toFixed(2)}%</span>
              </div>

              <div className="metric">
                <span className="label">Volatility:</span>
                <span className="value">{(risk_metrics.volatility * 100).toFixed(2)}%</span>
              </div>

              <div className="metric">
                <span className="label">Total Return:</span>
                <span className="value">{risk_metrics.total_return.toFixed(2)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* Support/Resistance */}
        {technical_analysis.support_resistance && (
          <div className="analysis-section">
            <h4>🎯 Key Levels</h4>
            <div className="levels">
              <div>Support: ${technical_analysis.support_resistance.support.toFixed(2)}</div>
              <div>Resistance: ${technical_analysis.support_resistance.resistance.toFixed(2)}</div>
              <div>Current: ${technical_analysis.support_resistance.current_price.toFixed(2)}</div>
              <div>Pivot: ${technical_analysis.support_resistance.pivot_point.toFixed(2)}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MarketAnalysisWidget;