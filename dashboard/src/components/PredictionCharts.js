import React, { useState, useEffect, useRef } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { useTouchGestures } from '../hooks/useTouchGestures';
import './PredictionCharts.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function PredictionCharts({ data, socket }) {
  const [predictionData, setPredictionData] = useState(data);
  const [lastUpdate, setLastUpdate] = useState(data?.last_update || null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [updateCount, setUpdateCount] = useState(0);
  const [showNotification, setShowNotification] = useState(false);
  const [chartZoom, setChartZoom] = useState(1);
  const chartRefs = useRef({});

  // Update local state when data prop changes
  useEffect(() => {
    if (data) {
      setPredictionData(data);
      setLastUpdate(data.last_update);
    }
  }, [data]);

  // WebSocket event listeners
  useEffect(() => {
    if (!socket) return;

    const handlePredictionUpdate = (updateData) => {
      try {
        const newData = updateData.data;
        console.log('📊 Received prediction update:', updateData);

        setPredictionData(prevData => ({
          ...prevData,
          ...newData,
          last_update: new Date().toISOString()
        }));

        setLastUpdate(new Date().toISOString());
        setUpdateCount(prev => prev + 1);

        // Add visual feedback for update
        showUpdateNotification();

      } catch (error) {
        console.error('❌ Error processing prediction update:', error);
      }
    };

    const handleConnect = () => {
      setConnectionStatus('connected');
      console.log('🔗 PredictionCharts connected to WebSocket');
    };

    const handleDisconnect = () => {
      setConnectionStatus('disconnected');
      console.log('🔌 PredictionCharts disconnected from WebSocket');
    };

    const handleConnectError = () => {
      setConnectionStatus('error');
      console.log('❌ PredictionCharts WebSocket connection error');
    };

    const handleSentimentUpdate = (updateData) => {
      try {
        const sentimentData = updateData.data;
        console.log('📊 Received sentiment update:', updateData);

        // Update prediction data with sentiment information
        setPredictionData(prevData => ({
          ...prevData,
          sentiment_chart: formatSentimentChartData(sentimentData),
          last_update: new Date().toISOString()
        }));

        setLastUpdate(new Date().toISOString());
        setUpdateCount(prev => prev + 1);

        showUpdateNotification();

      } catch (error) {
        console.error('❌ Error processing sentiment update:', error);
      }
    };

    const handleVolatilityUpdate = (updateData) => {
      try {
        const volatilityData = updateData.data;
        console.log('📊 Received volatility update:', updateData);

        // Update prediction data with volatility information
        setPredictionData(prevData => ({
          ...prevData,
          volatility_chart: formatVolatilityChartData(volatilityData),
          last_update: new Date().toISOString()
        }));

        setLastUpdate(new Date().toISOString());
        setUpdateCount(prev => prev + 1);

        showUpdateNotification();

      } catch (error) {
        console.error('❌ Error processing volatility update:', error);
      }
    };

    // Subscribe to all prediction-related events
    socket.on('prediction_update', handlePredictionUpdate);
    socket.on('sentiment_update', handleSentimentUpdate);
    socket.on('volatility_update', handleVolatilityUpdate);
    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('connect_error', handleConnectError);

    // Set initial connection status
    if (socket.connected) {
      setConnectionStatus('connected');
    }

    // Cleanup
    return () => {
      socket.off('prediction_update', handlePredictionUpdate);
      socket.off('sentiment_update', handleSentimentUpdate);
      socket.off('volatility_update', handleVolatilityUpdate);
      socket.off('connect', handleConnect);
      socket.off('disconnect', handleDisconnect);
      socket.off('connect_error', handleConnectError);
    };
  }, [socket]);

  // Pinch-to-zoom for charts
  const { bindPinch } = useTouchGestures({
    onPinch: (scale) => {
      const newZoom = Math.max(0.5, Math.min(2, scale));
      setChartZoom(newZoom);
    },
  });

  // Show visual notification for updates
  const showUpdateNotification = () => {
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 3000);
  };

  // Format last update time
  const formatLastUpdate = (timestamp) => {
    if (!timestamp) return 'Never';

    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return date.toLocaleDateString();
  };

  // Get connection status color
  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return '#6bcf7f';
      case 'connecting': return '#ffd93d';
      case 'disconnected': return '#ff6b6b';
      case 'error': return '#ff6b6b';
      default: return '#bdc3c7';
    }
  };

  // Get data freshness class for styling
  const getFreshnessClass = (timestamp) => {
    if (!timestamp) return 'stale';

    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = diffMs / 60000;

    if (diffMins < 5) return 'fresh';
    if (diffMins < 30) return 'recent';
    if (diffMins < 120) return 'stale';
    return 'very-stale';
  };

  // Format sentiment data for charts
  const formatSentimentChartData = (sentimentData) => {
    const timestamps = [];
    const sentiment = [];
    const predictedSentiment = [];

    // Generate time series data (last 24 hours)
    const now = new Date();
    for (let i = 23; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 60 * 1000);
      timestamps.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));

      // Use real sentiment data if available, otherwise generate
      const score = sentimentData.overall_score || (Math.random() - 0.5) * 0.4;
      sentiment.push(score);
      predictedSentiment.push(score + (Math.random() - 0.5) * 0.1);
    }

    return {
      timestamps,
      sentiment,
      predicted_sentiment: predictedSentiment
    };
  };

  // Format volatility data for charts
  const formatVolatilityChartData = (volatilityData) => {
    const timestamps = [];
    const volatility = [];
    const predictedVolatility = [];

    // Generate time series data (last 12 hours)
    const now = new Date();
    for (let i = 11; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 60 * 1000);
      timestamps.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));

      // Use real volatility data if available, otherwise generate
      const vol = volatilityData.volatility_pct || (0.01 + Math.random() * 0.07);
      volatility.push(vol * 100); // Convert to percentage
      predictedVolatility.push((vol + (Math.random() - 0.5) * 0.01) * 100);
    }

    return {
      timestamps,
      volatility,
      predicted_volatility: predictedVolatility
    };
  };

  if (!predictionData) {
    return (
      <div className="prediction-charts loading">
        <h3>🔮 AI Predictions</h3>
        <div className="connection-status">
          <span
            className="status-indicator"
            style={{ backgroundColor: getStatusColor() }}
          ></span>
          <span className="status-text">{connectionStatus}</span>
        </div>
        <p>Loading prediction data...</p>
      </div>
    );
  }

  // Prepare price prediction chart data
  const priceLabels = data.price_chart?.timestamps?.map(ts =>
    new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  ) || [];

  const priceData = {
    labels: priceLabels,
    datasets: [
      {
        label: 'Actual Price',
        data: data.price_chart?.actual_prices || [],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        tension: 0.1,
        pointRadius: 3,
      },
      {
        label: 'AI Prediction',
        data: data.price_chart?.predictions || [],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        tension: 0.1,
        pointRadius: 3,
      },
      {
        label: 'Confidence Upper',
        data: data.price_chart?.confidence_upper || [],
        borderColor: 'rgba(255, 99, 132, 0.3)',
        backgroundColor: 'rgba(255, 99, 132, 0.05)',
        borderDash: [5, 5],
        fill: '+1',
        tension: 0.1,
        pointRadius: 0,
      },
      {
        label: 'Confidence Lower',
        data: data.price_chart?.confidence_lower || [],
        borderColor: 'rgba(255, 99, 132, 0.3)',
        backgroundColor: 'rgba(255, 99, 132, 0.05)',
        borderDash: [5, 5],
        fill: false,
        tension: 0.1,
        pointRadius: 0,
      },
    ],
  };

  // Prepare volatility chart data
  const volatilityLabels = data.volatility_chart?.timestamps?.map(ts =>
    new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  ) || [];

  const volatilityData = {
    labels: volatilityLabels,
    datasets: [
      {
        label: 'Actual Volatility',
        data: data.volatility_chart?.volatility || [],
        borderColor: 'rgb(255, 205, 86)',
        backgroundColor: 'rgba(255, 205, 86, 0.2)',
        tension: 0.1,
      },
      {
        label: 'Predicted Volatility',
        data: data.volatility_chart?.predicted_volatility || [],
        borderColor: 'rgb(153, 102, 255)',
        backgroundColor: 'rgba(153, 102, 255, 0.2)',
        tension: 0.1,
      },
    ],
  };

  // Prepare sentiment chart data
  const sentimentLabels = data.sentiment_chart?.timestamps?.map(ts =>
    new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  ) || [];

  const sentimentData = {
    labels: sentimentLabels,
    datasets: [
      {
        label: 'Market Sentiment',
        data: data.sentiment_chart?.sentiment || [],
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.1,
        fill: true,
      },
      {
        label: 'Predicted Sentiment',
        data: data.sentiment_chart?.predicted_sentiment || [],
        borderColor: 'rgb(255, 159, 64)',
        backgroundColor: 'rgba(255, 159, 64, 0.2)',
        tension: 0.1,
        fill: true,
      },
    ],
  };

  // Prepare model performance data
  const modelLabels = ['ARIMA', 'LSTM', 'Ensemble'];
  const modelData = {
    labels: modelLabels,
    datasets: [
      {
        label: 'Model Accuracy',
        data: [
          data.model_performance?.arima_accuracy || 0,
          data.model_performance?.lstm_accuracy || 0,
          data.model_performance?.ensemble_accuracy || 0,
        ],
        backgroundColor: [
          'rgba(255, 99, 132, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 205, 86, 0.8)',
        ],
        borderColor: [
          'rgb(255, 99, 132)',
          'rgb(54, 162, 235)',
          'rgb(255, 205, 86)',
        ],
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 1000,
      easing: 'easeInOutQuart',
    },
    transitions: {
      active: {
        animation: {
          duration: 500,
        },
      },
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 15,
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
        },
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
        },
      },
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false,
    },
  };

  const barOptions = {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        beginAtZero: true,
        max: 1,
        ticks: {
          ...chartOptions.scales.y.ticks,
          callback: function(value) {
            return (value * 100).toFixed(0) + '%';
          },
        },
      },
    },
  };

  return (
    <div className="prediction-charts">
      {showNotification && (
        <div className="update-notification">
          🔄 Prediction data updated
        </div>
      )}
      <div className="prediction-header">
        <h3>🔮 AI Price Predictions</h3>
        <div className="prediction-meta">
          <div className="connection-status">
            <span
              className="status-indicator"
              style={{ backgroundColor: getStatusColor() }}
            ></span>
            <span className="status-text">{connectionStatus}</span>
          </div>
          <div className="last-update">
            <span className="update-label">Last update:</span>
            <span className="update-time">{formatLastUpdate(lastUpdate)}</span>
            {updateCount > 0 && (
              <span className="update-count">({updateCount} updates)</span>
            )}
          </div>
        </div>
      </div>

      <div className="charts-grid">
        {/* Price Prediction Chart */}
        <div className="chart-container" {...bindPinch()}>
          <h4>Price Prediction with Confidence Intervals</h4>
          <div
            className="chart-wrapper"
            style={{
              transform: `scale(${chartZoom})`,
              transformOrigin: 'center center'
            }}
          >
            <Line data={priceData} options={chartOptions} />
          </div>
          {chartZoom !== 1 && (
            <div className="zoom-controls">
              <button onClick={() => setChartZoom(1)}>🔍 Reset Zoom</button>
              <span className="zoom-level">{Math.round(chartZoom * 100)}%</span>
            </div>
          )}
        </div>

        {/* Volatility Chart */}
        <div className="chart-container" {...bindPinch()}>
          <h4>Market Volatility Analysis</h4>
          <div
            className="chart-wrapper"
            style={{
              transform: `scale(${chartZoom})`,
              transformOrigin: 'center center'
            }}
          >
            <Line data={volatilityData} options={chartOptions} />
          </div>
        </div>

        {/* Sentiment Chart */}
        <div className="chart-container" {...bindPinch()}>
          <h4>Market Sentiment Tracking</h4>
          <div
            className="chart-wrapper"
            style={{
              transform: `scale(${chartZoom})`,
              transformOrigin: 'center center'
            }}
          >
            <Line data={sentimentData} options={chartOptions} />
          </div>
        </div>

        {/* Model Performance */}
        <div className="chart-container" {...bindPinch()}>
          <h4>AI Model Performance</h4>
          <div
            className="chart-wrapper"
            style={{
              transform: `scale(${chartZoom})`,
              transformOrigin: 'center center'
            }}
          >
            <Bar data={modelData} options={barOptions} />
          </div>
        </div>
      </div>

      {/* Prediction Summary */}
      <div className="prediction-summary">
        <div className="summary-item">
          <span className="label">Latest Prediction</span>
          <span className="value">
            ${data.price_chart?.predictions?.[data.price_chart.predictions.length - 1]?.toLocaleString() || 'N/A'}
          </span>
        </div>

        <div className="summary-item">
          <span className="label">Confidence Range</span>
          <span className="value">
            ${(data.price_chart?.confidence_lower?.[data.price_chart.confidence_lower.length - 1]?.toLocaleString() || 'N/A')} - ${(data.price_chart?.confidence_upper?.[data.price_chart.confidence_upper.length - 1]?.toLocaleString() || 'N/A')}
          </span>
        </div>

        <div className="summary-item">
          <span className="label">Best Model</span>
          <span className="value">
            {data.model_performance?.ensemble_accuracy > data.model_performance?.lstm_accuracy &&
             data.model_performance?.ensemble_accuracy > data.model_performance?.arima_accuracy
              ? 'Ensemble'
              : data.model_performance?.lstm_accuracy > data.model_performance?.arima_accuracy
              ? 'LSTM'
              : 'ARIMA'
            } ({Math.max(
              data.model_performance?.arima_accuracy || 0,
              data.model_performance?.lstm_accuracy || 0,
              data.model_performance?.ensemble_accuracy || 0
            ) * 100}%)
          </span>
        </div>

        <div className="summary-item">
          <span className="label">Data Freshness</span>
          <span className="value freshness">
            <span className={`freshness-indicator ${getFreshnessClass(lastUpdate)}`}></span>
            {formatLastUpdate(lastUpdate)}
          </span>
        </div>

        <div className="summary-item">
          <span className="label">Connection</span>
          <span className="value">
            <span
              className="connection-dot"
              style={{ backgroundColor: getStatusColor() }}
            ></span>
            {connectionStatus}
          </span>
        </div>
      </div>
    </div>
  );
}

export default PredictionCharts;