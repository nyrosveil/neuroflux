import React, { useState, useEffect } from 'react';
import './App.css';
import SystemStatus from './components/SystemStatus';
import AgentMetrics from './components/AgentMetrics';
import AgentControlPanel from './components/AgentControlPanel';
import MarketDataWidget from './components/MarketDataWidget';
import PerformanceCharts from './components/PerformanceCharts';
import PredictionCharts from './components/PredictionCharts';
import RealTimeNotifications from './components/RealTimeNotifications';
import ErrorDisplay from './components/ErrorDisplay';
import NotificationToast from './components/NotificationToast';
import OnboardingTour, { useOnboarding } from './components/OnboardingTour';
import { ErrorProvider, ErrorBoundary } from './contexts/ErrorContext';
import { useTouchGestures, usePullToRefresh } from './hooks/useTouchGestures';
import { registerServiceWorker, requestNotificationPermission, setupInstallPrompt, isPWA } from './utils/pwaUtils';
import { io } from 'socket.io-client';

function App() {
  const [systemData, setSystemData] = useState({
    status: 'loading',
    uptime: 0,
    version: '3.2.0',
    orchestrator: { connected: false },
    ml_models: { available: false },
    exchanges: { available: false }
  });
  const [agentData, setAgentData] = useState([]);
  const [predictionData, setPredictionData] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [socket, setSocket] = useState(null);
  const [currentView, setCurrentView] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [navCollapsed, setNavCollapsed] = useState(false);
  const [showViewSelector, setShowViewSelector] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Onboarding tour
  const { showTour, completeTour } = useOnboarding();

  // Component views for swipe navigation
  const views = [
    {
      id: 'overview',
      name: 'Overview',
      icon: '📊',
      components: ['SystemStatus', 'AgentMetrics'],
      description: 'System status & agent overview'
    },
    {
      id: 'agents',
      name: 'Agents',
      icon: '🤖',
      components: ['AgentControlPanel', 'AgentMetrics'],
      description: 'Agent control & monitoring'
    },
    {
      id: 'markets',
      name: 'Markets',
      icon: '📈',
      components: ['MarketDataWidget'],
      description: 'Live market data & trading'
    },
    {
      id: 'analytics',
      name: 'Analytics',
      icon: '📊',
      components: ['PerformanceCharts', 'PredictionCharts'],
      description: 'Performance & predictions'
    },
    {
      id: 'notifications',
      name: 'Alerts',
      icon: '🔔',
      components: ['RealTimeNotifications'],
      description: 'System alerts & notifications'
    }
  ];

  useEffect(() => {
    // Initialize WebSocket connection (proxied to API server)
    // Temporarily disabled to test HTTP-only functionality
    const newSocket = null; // io('/');
    setSocket(newSocket);

    // Fetch initial data
    const loadData = async () => {
      await Promise.all([fetchSystemData(), fetchAgentData(), fetchPredictionData()]);
      setIsLoading(false);
    };
    loadData();

    // Initialize PWA features
    if (!isPWA()) {
      registerServiceWorker();
      requestNotificationPermission();
      setupInstallPrompt();
    }

    // WebSocket event listeners - disabled
    // if (newSocket) {
    //   newSocket.on('system_update', (data) => {
    //     setSystemData(data);
    //   });

    //   newSocket.on('agent_update', (data) => {
    //     setAgentData(prev => [...prev, data]);
    //   });

    //   newSocket.on('notification', (notification) => {
    //     setNotifications(prev => [notification, ...prev.slice(0, 9)]); // Keep last 10
    //   });
    // }

    return () => {
      // if (newSocket) newSocket.close();
    };
  }, []);

  const fetchSystemData = async () => {
    try {
      console.log('Fetching system data...');
      const response = await fetch('/api/status');
      console.log('Response status:', response.status);
      const data = await response.json();
      console.log('System data:', data);
      setSystemData(data);
    } catch (error) {
      console.error('Failed to fetch system data:', error);
    }
  };

  const fetchAgentData = async () => {
    try {
      console.log('Fetching agent data...');
      const response = await fetch('/api/agents');
      console.log('Agent response status:', response.status);
      const data = await response.json();
      console.log('Agent data:', data);
      setAgentData(data);
    } catch (error) {
      console.error('Failed to fetch agent data:', error);
    }
  };

  const fetchPredictionData = async () => {
    try {
      console.log('Fetching prediction data...');
      const response = await fetch('/api/dashboard/predictions');
      console.log('Prediction response status:', response.status);
      const data = await response.json();
      console.log('Prediction data:', data);
      setPredictionData(data);
    } catch (error) {
      console.error('Failed to fetch prediction data:', error);
      // Use mock data for development
      setPredictionData(getMockPredictionData());
    }
  };

  // Pull to refresh handler
  const handlePullToRefresh = async () => {
    setIsRefreshing(true);
    try {
      // Refresh all data
      await Promise.all([
        fetchSystemData(),
        fetchAgentData(),
        fetchPredictionData()
      ]);
      console.log('🔄 Data refreshed via pull-to-refresh');
    } catch (error) {
      console.error('❌ Pull-to-refresh failed:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  // Touch gesture handlers for navigation
  const { bindSwipe } = useTouchGestures({
    onSwipeLeft: () => {
      setCurrentView(prev => Math.min(prev + 1, views.length - 1));
    },
    onSwipeRight: () => {
      setCurrentView(prev => Math.max(prev - 1, 0));
    },
    swipeThreshold: 75, // Require more deliberate swipes
  });

  // Pull to refresh binding
  const bindPullToRefresh = usePullToRefresh(handlePullToRefresh);

  const getMockPredictionData = () => {
    const timestamps = [];
    const prices = [];
    const predictions = [];
    const confidenceUpper = [];
    const confidenceLower = [];

    const baseTime = new Date();
    const basePrice = 45000;

    for (let i = 0; i < 24; i++) {
      const timestamp = new Date(baseTime.getTime() + i * 60 * 60 * 1000).toISOString();
      timestamps.push(timestamp);

      const trend = 0.001 * i;
      const noise = (Math.random() - 0.5) * 0.04;
      const price = basePrice * (1 + trend + noise);
      prices.push(Math.round(price * 100) / 100);

      const predNoise = (Math.random() - 0.5) * 0.03;
      const prediction = price * (1 + predNoise);
      predictions.push(Math.round(prediction * 100) / 100);

      const confWidth = price * 0.05;
      confidenceUpper.push(Math.round((prediction + confWidth) * 100) / 100);
      confidenceLower.push(Math.round((prediction - confWidth) * 100) / 100);
    }

    return {
      price_chart: {
        timestamps,
        actual_prices: prices,
        predictions,
        confidence_upper: confidenceUpper,
        confidence_lower: confidenceLower
      },
      volatility_chart: {
        timestamps: timestamps.slice(-12),
        volatility: Array.from({length: 12}, () => Math.round((0.01 + Math.random() * 0.07) * 10000) / 10000),
        predicted_volatility: Array.from({length: 12}, () => Math.round((0.01 + Math.random() * 0.07) * 10000) / 10000)
      },
      sentiment_chart: {
        timestamps: timestamps.slice(-24),
        sentiment: Array.from({length: 24}, () => Math.round((0.2 + Math.random() * 0.6) * 100) / 100),
        predicted_sentiment: Array.from({length: 24}, () => Math.round((0.2 + Math.random() * 0.6) * 100) / 100)
      },
      model_performance: {
        arima_accuracy: Math.round((0.65 + Math.random() * 0.2) * 100) / 100,
        lstm_accuracy: Math.round((0.70 + Math.random() * 0.2) * 100) / 100,
        ensemble_accuracy: Math.round((0.75 + Math.random() * 0.25) * 100) / 100
      },
      last_update: new Date().toISOString()
    };
  };

  const renderCurrentView = () => {
    const currentViewData = views[currentView];

    return (
      <div className="dashboard-grid">
        {currentViewData.components.includes('SystemStatus') && (
          <SystemStatus data={systemData} />
        )}
        {currentViewData.components.includes('AgentMetrics') && (
          <AgentMetrics data={agentData} socket={socket} />
        )}
        {currentViewData.components.includes('AgentControlPanel') && (
          <AgentControlPanel agents={agentData} socket={socket} />
        )}
        {currentViewData.components.includes('MarketDataWidget') && (
          <MarketDataWidget socket={socket} />
        )}
        {currentViewData.components.includes('PerformanceCharts') && (
          <PerformanceCharts data={agentData} />
        )}
        {currentViewData.components.includes('PredictionCharts') && (
          <PredictionCharts data={predictionData} socket={socket} />
        )}
        {currentViewData.components.includes('RealTimeNotifications') && (
          <RealTimeNotifications notifications={notifications} />
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="App loading">
        <div className="loading-screen">
          <div className="loading-spinner">🧠</div>
          <h2>Loading NeuroFlux...</h2>
          <p>Initializing AI trading system</p>
        </div>
      </div>
    );
  }

  // Debug: Show data in console
  console.log('App render - isLoading:', isLoading, 'systemData:', systemData, 'agentData:', agentData);

  return (
    <ErrorProvider>
      <ErrorBoundary>
        <div className="App" {...bindSwipe()} {...bindPullToRefresh()}>
          {/* Pull to refresh indicator */}
          {isRefreshing && (
            <div className="pull-refresh-indicator">
              <div className="refresh-spinner">🔄</div>
              <span>Refreshing...</span>
            </div>
          )}

          {/* Error Display */}
          <ErrorDisplay />

          {/* Notification Toast */}
          <NotificationToast />

          {/* Onboarding Tour */}
          <OnboardingTour isVisible={showTour} onComplete={completeTour} />

          <header className="app-header">
            <h1>🧠 NeuroFlux Trading Dashboard</h1>
            <div className="header-controls">
              <button className="header-btn" onClick={fetchSystemData}>
                🔄 Refresh
              </button>
            </div>
          </header>

          {/* Mobile Navigation */}
          <nav className={`mobile-nav ${navCollapsed ? 'collapsed' : ''}`}>
        <div className="nav-header">
          <button
            className="nav-toggle"
            onClick={() => setNavCollapsed(!navCollapsed)}
          >
            {navCollapsed ? '☰' : '×'}
          </button>
          {!navCollapsed && (
            <div className="current-view-info">
              <span className="current-icon">{views[currentView].icon}</span>
              <span className="current-name">{views[currentView].name}</span>
            </div>
          )}
        </div>

        {!navCollapsed && (
          <div className="nav-tabs">
            {views.map((view, index) => (
              <button
                key={view.id}
                className={`nav-tab ${currentView === index ? 'active' : ''}`}
                onClick={() => setCurrentView(index)}
              >
                <span className="nav-icon">{view.icon}</span>
                <div className="nav-content">
                  <span className="nav-label">{view.name}</span>
                  <span className="nav-desc">{view.description}</span>
                </div>
              </button>
            ))}
          </div>
        )}

        {/* Quick actions */}
        {!navCollapsed && (
          <div className="nav-actions">
            <button
              className="quick-action"
              onClick={handlePullToRefresh}
              disabled={isRefreshing}
            >
              {isRefreshing ? '🔄' : '🔄'} Refresh
            </button>
            <button
              className="quick-action"
              onClick={() => setShowViewSelector(!showViewSelector)}
            >
              👁️ Views
            </button>
          </div>
        )}
      </nav>

      {/* View Selector Overlay */}
      {showViewSelector && (
        <div className="view-selector-overlay" onClick={() => setShowViewSelector(false)}>
          <div className="view-selector" onClick={e => e.stopPropagation()}>
            <h4>Select View</h4>
            <div className="view-grid">
              {views.map((view, index) => (
                <button
                  key={view.id}
                  className={`view-option ${currentView === index ? 'active' : ''}`}
                  onClick={() => {
                    setCurrentView(index);
                    setShowViewSelector(false);
                  }}
                >
                  <span className="view-icon">{view.icon}</span>
                  <span className="view-name">{view.name}</span>
                  <span className="view-desc">{view.description}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <main className="app-main">
        {renderCurrentView()}
      </main>

      <footer className="app-footer">
        <p>NeuroFlux v3.2.0 - Real-time AI Trading System</p>
          <div className="swipe-hint">
            👆 Swipe left/right to navigate • Pull down to refresh
          </div>
        </footer>
      </div>
    </ErrorBoundary>
  </ErrorProvider>
  );
}

export default App;
