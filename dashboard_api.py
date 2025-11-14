"""
🧠 NeuroFlux Dashboard API Server
Flask + Socket.IO server for real-time dashboard data

Built with love by Nyros Veil 🚀
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import configuration (handles environment loading automatically)
from config import config

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import time
import random
from datetime import datetime, timedelta
import threading
import asyncio
import time
import random
from datetime import datetime, timedelta
import threading
import numpy as np
import pandas as pd

class EventLoopManager:
    """Global event loop manager to prevent 'Event loop is closed' errors"""

    def __init__(self):
        self._loop = None
        self._lock = threading.Lock()

    def get_loop(self):
        """Get or create the global event loop thread-safely"""
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = event_loop_manager.get_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop

    async def run_async(self, coro):
        """Run async coroutine safely with error handling"""
        try:
            return await coro
        except Exception as e:
            print(f"❌ Async operation error: {e}")
            raise

# Global event loop manager instance
event_loop_manager = EventLoopManager()

def get_ccxt_manager():
    """Get or initialize CCXT manager on-demand"""
    global ccxt_manager
    if ccxt_manager is None and CCXT_AVAILABLE:
        try:
            # Create config dict for CCXT manager
            ccxt_config = {
                'binance_api_key': config.BINANCE_API_KEY,
                'binance_secret': config.BINANCE_API_SECRET,
                'coinbase_api_key': config.COINBASE_API_KEY,
                'coinbase_secret': config.COINBASE_API_SECRET,
                'bybit_api_key': config.BYBIT_API_KEY,
                'bybit_secret': config.BYBIT_API_SECRET,
                'kucoin_api_key': config.KUCOIN_API_KEY,
                'kucoin_secret': config.KUCOIN_API_SECRET,
            }

            ccxt_manager = CCXTExchangeManager(ccxt_config)
            print("✅ CCXT Exchange Manager initialized on-demand")
        except Exception as e:
            print(f"❌ Failed to initialize CCXT Manager: {e}")
            ccxt_manager = None
    return ccxt_manager

# Import NeuroFlux orchestration components
try:
    from src.neuroflux_orchestrator_v32 import NeuroFluxOrchestratorV32
    from src.orchestration.agent_registry import AgentRegistry, AgentStatus, AgentCapability
    from src.orchestration.communication_bus import CommunicationBus, Message, MessageType, MessagePriority
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Orchestrator modules not available: {e}")
    ORCHESTRATOR_AVAILABLE = False

# Import CCXT Exchange Manager
try:
    from src.exchanges.ccxt_exchange_manager import CCXTExchangeManager
    CCXT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CCXT Exchange Manager not available: {e}")
    CCXT_AVAILABLE = False

# Import Real-Time Agent Bus
try:
    from src.realtime_agent_bus import RealTimeAgentBus
    RT_BUS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Real-Time Agent Bus not available: {e}")
    RT_BUS_AVAILABLE = False

# Import ML modules
try:
    from src.ml.models import (
        BasePredictor,
        TimeSeriesPredictor,
        PredictionResult,
        ARIMAPredictor,
        ExponentialSmoothingPredictor,
        SimpleMovingAveragePredictor
    )
    ML_AVAILABLE = True
except ImportError:
    print("⚠️  ML modules not available - running without ML features")
    ML_AVAILABLE = False

# Configure Flask to serve React build files
app = Flask(__name__,
            static_folder='dashboard/build',
            static_url_path='/')

# Load configuration from config module
app.config.update(
    SECRET_KEY=config.SECRET_KEY,
    DEBUG=config.DEBUG,
    TESTING=config.TESTING,
    ENV=config.ENV,
    HOST=config.HOST,
    PORT=config.PORT,
    SESSION_TIMEOUT=config.SESSION_TIMEOUT,
    MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
)

# CORS configuration
CORS(app, origins=config.CORS_ORIGINS)
socketio = SocketIO(app, cors_allowed_origins=config.CORS_ORIGINS, async_mode='threading')

# Configure logging
import logging
from logging.handlers import RotatingFileHandler

# Configure root logger
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add file handler if log file is specified
if config.LOG_FILE:
    try:
        os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
        file_handler = RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=config.LOG_MAX_SIZE,
            backupCount=config.LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
        print(f"📝 Logging to file: {config.LOG_FILE}")
    except (OSError, PermissionError) as e:
        print(f"⚠️  Could not set up file logging: {e}")
        print("📝 Using console logging only")

# Reduce noise from other libraries
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)

# Log configuration info
print(f"🧠 NeuroFlux {config.ENV} server starting...")
print(f"📍 Host: {config.HOST}:{config.PORT}")
print(f"🐍 Python: {sys.version.split()[0]}")
print(f"📦 Conda: {config.CONDA_ENV_NAME or 'None'}")
print(f"🏠 Venv: {config.VENV_PATH or 'None'}")

# Global instances
orchestrator = None
orchestrator_task = None
ccxt_manager = None
rt_agent_bus = None

# CCXT Manager will be initialized on-demand in API endpoints

# Mock data for development
mock_data = {
    'flux_state': {
        'level': 0.75,
        'market_stability': 0.82,
        'last_update': datetime.now().isoformat()
    },
    'initialized_agents': 8,
    'total_agents': 12,
    'analytics': {
        'enabled': True
    },
    'predictions': {
        'price_forecast': {
            'model': 'ARIMA',
            'next_hour': 45230.50,
            'confidence': 0.78,
            'trend': 'bullish',
            'last_update': datetime.now().isoformat()
        },
        'volatility_forecast': {
            'model': 'GARCH',
            'next_hour': 0.023,
            'confidence': 0.65,
            'risk_level': 'moderate',
            'last_update': datetime.now().isoformat()
        },
        'market_sentiment': {
            'model': 'LSTM',
            'current': 0.72,
            'prediction': 0.68,
            'confidence': 0.82,
            'trend': 'neutral',
            'last_update': datetime.now().isoformat()
        }
    }
}

mock_agents = [
    {
        'agent_name': 'TradingAgent_1',
        'status': 'active',
        'success': True,
        'execution_time': 1.2
    },
    {
        'agent_name': 'RiskAgent_1',
        'status': 'active',
        'success': True,
        'execution_time': 0.8
    },
    {
        'agent_name': 'SentimentAgent_1',
        'status': 'active',
        'success': False,
        'execution_time': 2.1
    },
    {
        'agent_name': 'ResearchAgent_1',
        'status': 'idle',
        'success': True,
        'execution_time': 1.5
    }
]

def get_real_system_status():
    """Get real system status from orchestrator"""
    if not ORCHESTRATOR_AVAILABLE or not orchestrator:
        return mock_data

    try:
        # Get real agent data from orchestrator
        agent_registry = orchestrator.agent_registry
        agents = list(agent_registry.agents.values())

        # Calculate system metrics
        active_agents = len([a for a in agents if a.status == AgentStatus.ACTIVE])
        total_agents = len(agents)

        # Get flux state from running tasks/stats
        flux_level = min(1.0, orchestrator.stats['tasks_completed'] / max(1, orchestrator.stats['tasks_created']))
        market_stability = 0.8  # Would come from market data analysis

        return {
            'flux_state': {
                'level': flux_level,
                'market_stability': market_stability,
                'last_update': datetime.now().isoformat()
            },
            'initialized_agents': active_agents,
            'total_agents': total_agents,
            'analytics': {
                'enabled': True
            },
            'predictions': mock_data['predictions'],  # Keep mock predictions for now
            'orchestrator_stats': orchestrator.stats
        }
    except Exception as e:
        print(f"Error getting real system status: {e}")
        return mock_data

@app.route('/api/status')
def get_system_status():
    """Get current system status"""
    return jsonify(get_real_system_status())

def get_real_agent_data():
    """Get real agent data from orchestrator"""
    if not ORCHESTRATOR_AVAILABLE or not orchestrator:
        return mock_agents

    try:
        agent_registry = orchestrator.agent_registry
        agents = list(agent_registry.agents.values())

        real_agents = []
        for agent in agents:
            real_agents.append({
                'agent_name': agent.agent_id,
                'status': agent.status.value,
                'success': agent.health_score > 0.7,
                'execution_time': agent.performance_metrics.get('avg_response_time', 1.0),
                'capabilities': [cap.value for cap in agent.capabilities],
                'health_score': agent.health_score,
                'load_factor': agent.load_factor,
                'last_heartbeat': datetime.fromtimestamp(agent.last_heartbeat).isoformat()
            })

        return real_agents
    except Exception as e:
        print(f"Error getting real agent data: {e}")
        return mock_agents

@app.route('/api/agents')
def get_agents():
    """Get current agent data"""
    return jsonify(get_real_agent_data())

@app.route('/api/health')
def health_check():
    """System health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': time.time() - start_time,
        'ml_available': ML_AVAILABLE
    })

# React App Routes - Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    """Serve React app for all non-API routes"""
    # Skip API routes - let Flask handle them normally
    if path.startswith('api/') or path.startswith('socket.io/'):
        from flask import abort
        abort(404)

    # Get the React build directory path
    react_build_dir = os.path.join(os.path.dirname(__file__), 'dashboard', 'build')

    # Serve static files from React build
    if path and os.path.exists(os.path.join(react_build_dir, path)):
        return send_from_directory(react_build_dir, path)

    # Serve index.html for client-side routing
    return send_from_directory(react_build_dir, 'index.html')

# ML Prediction Endpoints
@app.route('/api/ml/models')
def get_available_models():
    """Get list of available ML models"""
    if not ML_AVAILABLE:
        return jsonify({'error': 'ML modules not available'}), 503

    models = [
        {
            'name': 'ARIMA',
            'description': 'AutoRegressive Integrated Moving Average for time series forecasting',
            'type': 'time_series',
            'class': 'ARIMAPredictor'
        },
        {
            'name': 'Exponential Smoothing',
            'description': 'Exponential smoothing for trend and seasonal forecasting',
            'type': 'time_series',
            'class': 'ExponentialSmoothingPredictor'
        },
        {
            'name': 'Simple Moving Average',
            'description': 'Simple moving average baseline predictor',
            'type': 'time_series',
            'class': 'SimpleMovingAveragePredictor'
        }
    ]
    return jsonify(models)

@app.route('/api/ml/predict/<model_type>', methods=['POST'])
def make_prediction(model_type):
    """Make a prediction using specified model"""
    if not ML_AVAILABLE:
        return jsonify({'error': 'ML modules not available'}), 503

    try:
        data = request.get_json()

        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400

        # Create model instance
        model_config = data.get('config', {})
        predictor = None

        if model_type == 'arima':
            predictor = ARIMAPredictor(config=model_config)
        elif model_type == 'exp_smoothing':
            predictor = ExponentialSmoothingPredictor(config=model_config)
        elif model_type == 'sma':
            predictor = SimpleMovingAveragePredictor(config=model_config)
        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400

        # Prepare data
        input_data = pd.DataFrame(data['data'])
        target_column = data.get('target_column', input_data.columns[0])

        # Train model if requested
        if data.get('train', False):
            metrics = predictor.fit(input_data, target_column)
        elif not predictor.is_trained:
            return jsonify({'error': 'Model not trained and training not requested'}), 400

        # Make prediction
        if 'prediction_data' in data:
            pred_data = pd.DataFrame(data['prediction_data'])
            result = predictor.predict_from_data(pred_data)
        else:
            # Use training data for in-sample prediction
            result = predictor.predict_from_data(input_data)

        # Convert result to JSON-serializable format
        response = {
            'model_type': model_type,
            'predictions': result.predictions.tolist() if hasattr(result.predictions, 'tolist') else result.predictions,
            'confidence_scores': result.confidence_scores.tolist() if result.confidence_scores is not None and hasattr(result.confidence_scores, 'tolist') else result.confidence_scores,
            'metadata': result.metadata,
            'timestamp': result.timestamp.isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/train/<model_type>', methods=['POST'])
def train_model(model_type):
    """Train a model with provided data"""
    if not ML_AVAILABLE:
        return jsonify({'error': 'ML modules not available'}), 503

    try:
        data = request.get_json()

        if not data or 'data' not in data:
            return jsonify({'error': 'No training data provided'}), 400

        # Create model instance
        model_config = data.get('config', {})
        predictor = None

        if model_type == 'arima':
            predictor = ARIMAPredictor(config=model_config)
        elif model_type == 'exp_smoothing':
            predictor = ExponentialSmoothingPredictor(config=model_config)
        elif model_type == 'sma':
            predictor = SimpleMovingAveragePredictor(config=model_config)
        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400

        # Prepare data
        input_data = pd.DataFrame(data['data'])
        target_column = data.get('target_column', input_data.columns[0])

        # Train model
        metrics = predictor.fit(input_data, target_column)

        return jsonify({
            'model_type': model_type,
            'trained': True,
            'metrics': metrics,
            'model_info': predictor.get_model_info()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/status')
def get_ml_status():
    """Get ML system status"""
    if not ML_AVAILABLE:
        return jsonify({
            'available': False,
            'message': 'ML modules not available'
        })

    return jsonify({
        'available': True,
        'models_loaded': 3,  # ARIMA, Exp Smoothing, SMA
        'supported_types': ['time_series'],
        'features': ['prediction', 'training', 'confidence_scores']
    })

# CCXT Exchange Manager Endpoints
@app.route('/api/exchanges/status')
def get_exchange_status():
    """Get status of all connected exchanges"""
    ccxt_mgr = get_ccxt_manager()
    if not CCXT_AVAILABLE or not ccxt_mgr:
        return jsonify({
            'available': False,
            'message': 'CCXT Exchange Manager not available'
        })

    status = ccxt_mgr.get_exchange_status()
    return jsonify({
        'available': True,
        'exchanges': status
    })

@app.route('/api/exchanges/ticker/<exchange>/<symbol>')
def get_ticker(exchange, symbol):
    """Get real-time ticker data from an exchange"""
    ccxt_mgr = get_ccxt_manager()
    if not CCXT_AVAILABLE or not ccxt_mgr:
        return jsonify({'error': 'CCXT Exchange Manager not available'}), 503

    try:
        # Use global event loop manager
        loop = event_loop_manager.get_loop()
        ticker = loop.run_until_complete(ccxt_mgr.get_ticker(exchange, symbol))

        if ticker:
            return jsonify(ticker)
        else:
            return jsonify({'error': f'Ticker not available for {exchange}:{symbol}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exchanges/orderbook/<exchange>/<symbol>')
def get_orderbook(exchange, symbol):
    """Get order book data from an exchange"""
    ccxt_mgr = get_ccxt_manager()
    if not CCXT_AVAILABLE or not ccxt_mgr:
        return jsonify({'error': 'CCXT Exchange Manager not available'}), 503

    try:
        limit = int(request.args.get('limit', 20))

        # Use global event loop manager
        loop = event_loop_manager.get_loop()
        orderbook = loop.run_until_complete(ccxt_mgr.get_orderbook(exchange, symbol, limit))

        if orderbook:
            return jsonify(orderbook)
        else:
            return jsonify({'error': f'Orderbook not available for {exchange}:{symbol}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exchanges/subscribe/ticker', methods=['POST'])
def subscribe_ticker():
    """Subscribe to real-time ticker updates"""
    ccxt_mgr = get_ccxt_manager()
    if not CCXT_AVAILABLE or not ccxt_mgr:
        return jsonify({'error': 'CCXT Exchange Manager not available'}), 503

    data = request.get_json()
    if not data or 'exchange' not in data or 'symbol' not in data:
        return jsonify({'error': 'Missing exchange or symbol'}), 400

    exchange = data['exchange']
    symbol = data['symbol']

    # For now, return subscription info (real WebSocket subscription would be implemented)
    return jsonify({
        'subscribed': True,
        'exchange': exchange,
        'symbol': symbol,
        'channel': 'ticker'
    })

@app.route('/api/exchanges/subscribe/orderbook', methods=['POST'])
def subscribe_orderbook():
    """Subscribe to real-time orderbook updates"""
    ccxt_mgr = get_ccxt_manager()
    if not CCXT_AVAILABLE or not ccxt_mgr:
        return jsonify({'error': 'CCXT Exchange Manager not available'}), 503

    data = request.get_json()
    if not data or 'exchange' not in data or 'symbol' not in data:
        return jsonify({'error': 'Missing exchange or symbol'}), 400

    exchange = data['exchange']
    symbol = data['symbol']
    depth = data.get('depth', 20)

    # For now, return subscription info
    return jsonify({
        'subscribed': True,
        'exchange': exchange,
        'symbol': symbol,
        'channel': 'orderbook',
        'depth': depth
    })

@app.route('/api/marketdata/multi-exchange/<symbol>')
def get_multi_exchange_data(symbol):
    """Get market data from multiple exchanges for comparison"""
    ccxt_mgr = get_ccxt_manager()
    if not CCXT_AVAILABLE or not ccxt_mgr:
        return jsonify({'error': 'CCXT Exchange Manager not available'}), 503

    exchanges = ['binance', 'coinbase', 'hyperliquid']  # Supported exchanges
    market_data = {}

    # Use global event loop managers
    loop = event_loop_manager.get_loop()

    for exchange in exchanges:
        try:
            ticker = loop.run_until_complete(ccxt_mgr.get_ticker(exchange, symbol))
            if ticker:
                market_data[exchange] = {
                    'price': ticker.get('last'),
                    'bid': ticker.get('bid'),
                    'ask': ticker.get('ask'),
                    'volume': ticker.get('quoteVolume'),
                    'timestamp': ticker.get('timestamp')
                }
        except Exception as e:
            market_data[exchange] = {'error': str(e)}


    return jsonify({
        'symbol': symbol,
        'data': market_data,
        'timestamp': datetime.now().isoformat()
    })

# Real-Time Agent Bus Endpoints
@app.route('/api/realtime/stats')
def get_realtime_bus_stats():
    """Get real-time agent bus statistics"""
    if not RT_BUS_AVAILABLE or not rt_agent_bus:
        return jsonify({'error': 'Real-Time Agent Bus not available'}), 503

    stats = rt_agent_bus.get_bus_stats()
    return jsonify(stats)

@app.route('/api/realtime/subscribe/<topic>', methods=['POST'])
def subscribe_topic(topic):
    """Subscribe to a real-time topic"""
    if not RT_BUS_AVAILABLE or not rt_agent_bus:
        return jsonify({'error': 'Real-Time Agent Bus not available'}), 503

    data = request.get_json() or {}
    subscriber_id = data.get('subscriber_id', f"dashboard_{topic}")

    # Use global event loop manager
    loop = event_loop_manager.get_loop()
    success = loop.run_until_complete(rt_agent_bus.subscribe_topic(subscriber_id, topic))

    if success:
        return jsonify({'subscribed': True, 'topic': topic, 'subscriber_id': subscriber_id})
    else:
        return jsonify({'error': f'Failed to subscribe to {topic}'}), 500

@app.route('/api/realtime/broadcast/<topic>', methods=['POST'])
def broadcast_event(topic):
    """Broadcast an event through the real-time bus"""
    if not RT_BUS_AVAILABLE or not rt_agent_bus:
        return jsonify({'error': 'Real-Time Agent Bus not available'}), 503

    data = request.get_json()
    if not data or 'payload' not in data:
        return jsonify({'error': 'Missing payload'}), 400

    payload = data['payload']
    priority = data.get('priority', 'medium')

    # Use global event loop manager
    loop = event_loop_manager.get_loop()
    success = loop.run_until_complete(rt_agent_bus.broadcast_event(topic, payload))

    if success:
        return jsonify({'broadcast': True, 'topic': topic})
    else:
        return jsonify({'error': f'Failed to broadcast to {topic}'}), 500

@app.route('/api/realtime/signal/trading', methods=['POST'])
def send_trading_signal():
    """Send a trading signal through the real-time bus"""
    if not RT_BUS_AVAILABLE or not rt_agent_bus:
        return jsonify({'error': 'Real-Time Agent Bus not available'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing signal data'}), 400

    signal_type = data.get('signal_type', 'unknown')
    symbol = data.get('symbol', 'UNKNOWN')
    confidence = data.get('confidence', 0.5)
    metadata = data.get('metadata', {})

    # Use global event loop manager
    loop = event_loop_manager.get_loop()
    success = loop.run_until_complete(rt_agent_bus.send_trading_signal(signal_type, symbol, confidence, metadata))

    if success:
        return jsonify({'signal_sent': True, 'type': signal_type, 'symbol': symbol})
    else:
        return jsonify({'error': 'Failed to send trading signal'}), 500

@app.route('/api/realtime/alert/risk', methods=['POST'])
def send_risk_alert():
    """Send a risk alert through the real-time bus"""
    if not RT_BUS_AVAILABLE or not rt_agent_bus:
        return jsonify({'error': 'Real-Time Agent Bus not available'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing alert data'}), 400

    alert_type = data.get('alert_type', 'unknown')
    severity = data.get('severity', 'medium')
    message = data.get('message', 'Risk alert')
    agent_id = data.get('agent_id')

    # Use global event loop manager
    loop = event_loop_manager.get_loop()
    success = loop.run_until_complete(rt_agent_bus.send_risk_alert(alert_type, severity, message, agent_id))

    if success:
        return jsonify({'alert_sent': True, 'type': alert_type, 'severity': severity})
    else:
        return jsonify({'error': 'Failed to send risk alert'}), 500

# Advanced Features Endpoints
@app.route('/api/arbitrage/opportunities/<symbol>')
def get_arbitrage_opportunities(symbol):
    """Get arbitrage opportunities across exchanges"""
    ccxt_mgr = get_ccxt_manager()
    if not CCXT_AVAILABLE or not ccxt_mgr:
        return jsonify({'error': 'CCXT Exchange Manager not available'}), 503

    exchanges = ['binance', 'coinbase', 'hyperliquid', 'bybit', 'kucoin']
    prices = {}

    # Use global event loop managers
    loop = event_loop_manager.get_loop()

    for exchange in exchanges:
        try:
            ticker = loop.run_until_complete(ccxt_mgr.get_ticker(exchange, symbol))
            if ticker and ticker.get('last'):
                prices[exchange] = ticker['last']
        except Exception as e:
            continue


    if len(prices) < 2:
        return jsonify({'opportunities': [], 'message': 'Need at least 2 exchanges with price data'})

    # Find arbitrage opportunities
    opportunities = []
    exchanges_list = list(prices.keys())

    for i, exchange1 in enumerate(exchanges_list):
        for exchange2 in exchanges_list[i+1:]:
            price1 = prices[exchange1]
            price2 = prices[exchange2]

            spread = abs(price1 - price2)
            spread_percentage = (spread / min(price1, price2)) * 100

            if spread_percentage > 0.1:  # 0.1% minimum spread
                opportunities.append({
                    'exchange1': exchange1,
                    'exchange2': exchange2,
                    'price1': price1,
                    'price2': price2,
                    'spread': spread,
                    'spread_percentage': round(spread_percentage, 4),
                    'direction': 'buy_low_sell_high' if price1 < price2 else 'buy_high_sell_low',
                    'profit_potential': spread_percentage > 0.5,  # Profitable if > 0.5%
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                })

    # Sort by spread percentage (highest first)
    opportunities.sort(key=lambda x: x['spread_percentage'], reverse=True)

    return jsonify({
        'symbol': symbol,
        'opportunities': opportunities[:10],  # Top 10 opportunities
        'total_exchanges': len(prices),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/risk/portfolio')
def get_portfolio_risk():
    """Get portfolio risk assessment across exchanges"""
    if not ORCHESTRATOR_AVAILABLE or not orchestrator:
        return jsonify({'error': 'Orchestrator not available'}), 503

    # Mock portfolio risk data (would be calculated from real positions)
    risk_data = {
        'total_exposure': 125000.50,
        'positions': [
            {
                'symbol': 'BTC/USDT',
                'exchange': 'binance',
                'size': 2.5,
                'entry_price': 43500.00,
                'current_price': 45230.50,
                'pnl': 4407.50,
                'pnl_percentage': 10.15,
                'risk_level': 'medium'
            },
            {
                'symbol': 'ETH/USDT',
                'exchange': 'coinbase',
                'size': 15.0,
                'entry_price': 2650.00,
                'current_price': 2780.25,
                'pnl': 1953.75,
                'pnl_percentage': 7.37,
                'risk_level': 'low'
            }
        ],
        'risk_metrics': {
            'var_95': -2500.00,  # Value at Risk 95%
            'expected_shortfall': -3800.00,
            'max_drawdown': -1200.00,
            'sharpe_ratio': 1.85,
            'volatility': 0.023
        },
        'alerts': [
            {
                'type': 'high_volatility',
                'message': 'BTC/USDT showing increased volatility',
                'severity': 'medium',
                'timestamp': datetime.now().isoformat()
            }
        ],
        'timestamp': datetime.now().isoformat()
    }

    return jsonify(risk_data)

@app.route('/api/trading/routes/<symbol>')
def get_trading_routes(symbol):
    """Get optimal trading routes across exchanges"""
    ccxt_mgr = get_ccxt_manager()
    if not CCXT_AVAILABLE or not ccxt_mgr:
        return jsonify({'error': 'CCXT Exchange Manager not available'}), 503

    # Analyze liquidity and fees across exchanges
    routes = []

    exchanges = ['binance', 'coinbase', 'hyperliquid', 'bybit', 'kucoin']

    # Use global event loop managers
    loop = event_loop_manager.get_loop()

    for exchange in exchanges:
        try:
            ticker = loop.run_until_complete(ccxt_mgr.get_ticker(exchange, symbol))
            orderbook = loop.run_until_complete(ccxt_mgr.get_orderbook(exchange, symbol, limit=10))

            if ticker and orderbook:
                # Calculate liquidity metrics
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])

                bid_liquidity = sum(vol for price, vol in bids[:5]) if bids else 0
                ask_liquidity = sum(vol for price, vol in asks[:5]) if asks else 0

                routes.append({
                    'exchange': exchange,
                    'symbol': symbol,
                    'price': ticker.get('last'),
                    'bid_liquidity': bid_liquidity,
                    'ask_liquidity': ask_liquidity,
                    'spread': (asks[0][0] - bids[0][0]) if asks and bids else 0,
                    'volume_24h': ticker.get('quoteVolume', 0),
                    'recommended': bid_liquidity > 100 and ask_liquidity > 100  # Simple recommendation
                })

        except Exception as e:
            continue


    # Sort by liquidity and volume
    routes.sort(key=lambda x: (x['bid_liquidity'] + x['ask_liquidity']) * x.get('volume_24h', 0), reverse=True)

    return jsonify({
        'symbol': symbol,
        'routes': routes,
        'optimal_route': routes[0] if routes else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/market/analysis/<symbol>')
def get_market_analysis(symbol):
    """Get comprehensive market analysis"""
    ccxt_mgr = get_ccxt_manager()
    if not CCXT_AVAILABLE or not ccxt_mgr:
        return jsonify({'error': 'CCXT Exchange Manager not available'}), 503

    analysis = {
        'symbol': symbol,
        'price_analysis': {},
        'volume_analysis': {},
        'sentiment': {},
        'predictions': {},
        'recommendations': []
    }

    # Get data from multiple exchanges
    exchanges = ['binance', 'coinbase']

    # Use global event loop managers
    loop = event_loop_manager.get_loop()

    prices = []
    volumes = []

    for exchange in exchanges:
        try:
            ticker = loop.run_until_complete(ccxt_mgr.get_ticker(exchange, symbol))
            if ticker:
                price = ticker.get('last')
                volume = ticker.get('quoteVolume', 0)

                if price:
                    prices.append(price)
                    volumes.append(volume)

                analysis['price_analysis'][exchange] = {
                    'price': price,
                    'change_24h': ticker.get('percentage'),
                    'high_24h': ticker.get('high'),
                    'low_24h': ticker.get('low'),
                    'volume': volume
                }

        except Exception as e:
            continue


    # Calculate aggregate metrics
    if prices:
        analysis['price_analysis']['aggregate'] = {
            'avg_price': sum(prices) / len(prices),
            'min_price': min(prices),
            'max_price': max(prices),
            'price_spread': max(prices) - min(prices),
            'spread_percentage': ((max(prices) - min(prices)) / min(prices)) * 100
        }

    if volumes:
        analysis['volume_analysis'] = {
            'total_volume': sum(volumes),
            'avg_volume': sum(volumes) / len(volumes)
        }

    # Mock sentiment and predictions (would come from ML agents)
    analysis['sentiment'] = {
        'overall': 'bullish',
        'score': 0.72,
        'sources': ['news', 'social_media', 'technical']
    }

    analysis['predictions'] = {
        'next_hour': {
            'price': sum(prices) / len(prices) * 1.002 if prices else 0,
            'confidence': 0.68,
            'trend': 'upward'
        }
    }

    # Generate recommendations
    if analysis['price_analysis'].get('aggregate', {}).get('spread_percentage', 0) > 0.5:
        analysis['recommendations'].append({
            'type': 'arbitrage',
            'message': 'Arbitrage opportunity detected',
            'confidence': 'high'
        })

    if analysis['sentiment']['score'] > 0.7:
        analysis['recommendations'].append({
            'type': 'momentum',
            'message': 'Strong bullish momentum',
            'confidence': 'medium'
        })

    analysis['timestamp'] = datetime.now().isoformat()

    return jsonify(analysis)

@app.route('/api/dashboard/predictions')
def get_prediction_dashboard():
    """Get prediction data for dashboard charts"""
    # Generate mock prediction chart data
    timestamps = []
    prices = []
    predictions = []
    confidence_upper = []
    confidence_lower = []

    base_time = datetime.now()
    base_price = 45000

    for i in range(24):  # 24 hours of data
        timestamp = (base_time + timedelta(hours=i)).isoformat()
        timestamps.append(timestamp)

        # Generate realistic price data with trend
        trend = 0.001 * i  # Slight upward trend
        noise = random.uniform(-0.02, 0.02)
        price = base_price * (1 + trend + noise)
        prices.append(round(price, 2))

        # Generate predictions with confidence intervals
        pred_noise = random.uniform(-0.015, 0.015)
        prediction = price * (1 + pred_noise)
        predictions.append(round(prediction, 2))

        # Confidence intervals
        conf_width = price * 0.05  # 5% confidence interval
        confidence_upper.append(round(prediction + conf_width, 2))
        confidence_lower.append(round(prediction - conf_width, 2))

    return jsonify({
        'price_chart': {
            'timestamps': timestamps,
            'actual_prices': prices,
            'predictions': predictions,
            'confidence_upper': confidence_upper,
            'confidence_lower': confidence_lower
        },
        'volatility_chart': {
            'timestamps': timestamps[-12:],  # Last 12 hours
            'volatility': [round(random.uniform(0.01, 0.08), 4) for _ in range(12)],
            'predicted_volatility': [round(random.uniform(0.01, 0.08), 4) for _ in range(12)]
        },
        'sentiment_chart': {
            'timestamps': timestamps[-24:],  # Last 24 hours
            'sentiment': [round(random.uniform(0.2, 0.8), 2) for _ in range(24)],
            'predicted_sentiment': [round(random.uniform(0.2, 0.8), 2) for _ in range(24)]
        },
        'model_performance': {
            'arima_accuracy': round(random.uniform(0.65, 0.85), 2),
            'lstm_accuracy': round(random.uniform(0.70, 0.90), 2),
            'ensemble_accuracy': round(random.uniform(0.75, 0.95), 2)
        },
        'last_update': datetime.now().isoformat()
    })

async def initialize_orchestrator():
    """Initialize the NeuroFlux orchestrator and CCXT manager"""
    global orchestrator, ccxt_manager
    if ORCHESTRATOR_AVAILABLE:
        try:
            orchestrator = NeuroFluxOrchestratorV32()
            await orchestrator.initialize()

            # Set up message forwarding from agents to dashboard
            await setup_agent_message_forwarding()

            print("✅ NeuroFlux orchestrator initialized for dashboard")
        except Exception as e:
            print(f"❌ Failed to initialize orchestrator: {e}")
            orchestrator = None

    # Initialize CCXT Exchange Manager with API credentials
    if CCXT_AVAILABLE:
        try:
            # Create config dict for CCXT manager
            ccxt_config = {
                'binance_api_key': config.BINANCE_API_KEY,
                'binance_secret': config.BINANCE_API_SECRET,
                'coinbase_api_key': config.COINBASE_API_KEY,
                'coinbase_secret': config.COINBASE_API_SECRET,
                'bybit_api_key': config.BYBIT_API_KEY,
                'bybit_secret': config.BYBIT_API_SECRET,
                'kucoin_api_key': config.KUCOIN_API_KEY,
                'kucoin_secret': config.KUCOIN_API_SECRET,
            }

            ccxt_manager = CCXTExchangeManager(ccxt_config)
            await ccxt_manager.start()
            print("✅ CCXT Exchange Manager initialized")
        except Exception as e:
            print(f"❌ Failed to initialize CCXT Manager: {e}")
            ccxt_manager = None

    # Initialize Real-Time Agent Bus
    global rt_agent_bus
    if RT_BUS_AVAILABLE:
        try:
            rt_agent_bus = RealTimeAgentBus(orchestrator=orchestrator, socketio_instance=socketio)
            await rt_agent_bus.start()
            print("✅ Real-Time Agent Bus initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Real-Time Agent Bus: {e}")
            rt_agent_bus = None

async def setup_agent_message_forwarding():
    """Set up forwarding of agent messages to dashboard WebSocket clients"""
    if not orchestrator:
        return

    # Start a background task to monitor agent activity
    asyncio.create_task(monitor_agent_activity())

async def monitor_agent_activity():
    """Monitor agent activity and forward status updates to dashboard"""
    if not orchestrator:
        return

    last_agent_states = {}

    while orchestrator and orchestrator.running:
        try:
            await asyncio.sleep(2)  # Check every 2 seconds

            current_agents = list(orchestrator.agent_registry.agents.values())

            for agent in current_agents:
                agent_key = agent.agent_id
                current_state = {
                    'status': agent.status.value,
                    'health_score': agent.health_score,
                    'load_factor': agent.load_factor,
                    'last_heartbeat': agent.last_heartbeat
                }

                # Check if agent state changed
                if agent_key not in last_agent_states or last_agent_states[agent_key] != current_state:
                    last_agent_states[agent_key] = current_state

                    dashboard_message = {
                        'type': 'agent_status_update',
                        'agent_id': agent.agent_id,
                        'agent_type': agent.agent_type,
                        'status': agent.status.value,
                        'health_score': agent.health_score,
                        'load_factor': agent.load_factor,
                        'capabilities': [cap.value for cap in agent.capabilities],
                        'timestamp': datetime.fromtimestamp(agent.last_heartbeat).isoformat()
                    }
                    socketio.emit('agent_message', dashboard_message)

        except Exception as e:
            print(f"Error monitoring agent activity: {e}")
            await asyncio.sleep(5)  # Wait before retrying

def generate_real_time_updates():
    """Generate real-time updates from orchestrator, agents, and market data"""
    market_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # Major trading pairs
    last_market_data = {}

    while True:
        time.sleep(3)  # Update every 3 seconds for more responsive data

        try:
            # Get real system status
            system_data = get_real_system_status()
            socketio.emit('system_update', system_data)

            # Get real agent data
            agent_data = get_real_agent_data()
            socketio.emit('agents_update', agent_data)

            # Stream market data from CCXT exchanges
            if CCXT_AVAILABLE and ccxt_manager:
                for symbol in market_symbols:
                    market_update = stream_market_data(symbol, last_market_data)
                    if market_update:
                        socketio.emit('market_data', market_update)

            # Agent status updates
            for agent in agent_data:
                if random.random() < 0.15:  # 15% chance for individual updates
                    agent_update = agent.copy()
                    agent_update['timestamp'] = datetime.now().isoformat()
                    socketio.emit('agent_update', agent_update)

            # Generate realistic trading signals and notifications
            if random.random() < 0.25:
                generate_trading_signals()

        except Exception as e:
            print(f"Error in real-time updates: {e}")
            # Fallback to mock updates
            socketio.emit('system_update', mock_data)

def stream_market_data(symbol, last_data):
    """Stream real market data from exchanges"""
    ccxt_mgr = get_ccxt_manager()
    if not CCXT_AVAILABLE or not ccxt_mgr:
        return None

    try:
        # Get data from multiple exchanges
        exchanges_to_check = ['binance', 'coinbase']
        market_data = {
            'symbol': symbol,
            'exchanges': {},
            'timestamp': datetime.now().isoformat()
        }

        # Use global event loop managers
        loop = event_loop_manager.get_loop()

        for exchange in exchanges_to_check:
            try:
                ticker = loop.run_until_complete(ccxt_mgr.get_ticker(exchange, symbol))
                if ticker:
                    market_data['exchanges'][exchange] = {
                        'price': ticker.get('last', ticker.get('close')),
                        'bid': ticker.get('bid'),
                        'ask': ticker.get('ask'),
                        'volume': ticker.get('quoteVolume', ticker.get('volume')),
                        'change_24h': ticker.get('percentage'),
                        'high_24h': ticker.get('high'),
                        'low_24h': ticker.get('low')
                    }
            except Exception as e:
                # Exchange might not be available or symbol not supported
                continue


        # Calculate price spread and arbitrage opportunities
        prices = [data['price'] for data in market_data['exchanges'].values() if data.get('price')]
        if len(prices) > 1:
            market_data['spread'] = max(prices) - min(prices)
            market_data['spread_percentage'] = (market_data['spread'] / min(prices)) * 100
            market_data['arbitrage_opportunity'] = market_data['spread_percentage'] > 0.5  # 0.5% threshold

        # Only emit if data changed significantly or it's been a while
        data_key = f"{symbol}"
        if data_key not in last_data or has_significant_change(last_data[data_key], market_data):
            last_data[data_key] = market_data.copy()
            return market_data

    except Exception as e:
        print(f"Error streaming market data for {symbol}: {e}")

    return None

def has_significant_change(old_data, new_data):
    """Check if market data has changed significantly"""
    if not old_data or 'exchanges' not in old_data:
        return True

    # Check for price changes > 0.1%
    for exchange, new_exchange_data in new_data.get('exchanges', {}).items():
        old_exchange_data = old_data.get('exchanges', {}).get(exchange)
        if old_exchange_data and new_exchange_data.get('price'):
            old_price = old_exchange_data.get('price')
            new_price = new_exchange_data.get('price')
            if old_price and abs((new_price - old_price) / old_price) > 0.001:  # 0.1% change
                return True

    return False

def generate_trading_signals():
    """Generate realistic trading signals and notifications"""
    signal_types = [
        {'type': 'BUY_SIGNAL', 'message': 'Bullish divergence detected', 'symbol': 'BTC/USDT', 'confidence': 0.78},
        {'type': 'SELL_SIGNAL', 'message': 'Bearish engulfing pattern', 'symbol': 'ETH/USDT', 'confidence': 0.82},
        {'type': 'ARBITRAGE', 'message': 'Price spread opportunity detected', 'symbol': 'SOL/USDT', 'confidence': 0.65},
        {'type': 'VOLATILITY', 'message': 'High volatility alert', 'symbol': 'BTC/USDT', 'confidence': 0.91},
        {'type': 'VOLUME_SPIKE', 'message': 'Unusual volume detected', 'symbol': 'ETH/USDT', 'confidence': 0.74}
    ]

    signal = random.choice(signal_types)
    signal['timestamp'] = datetime.now().isoformat()
    signal['agent'] = random.choice(['TechnicalAnalysis', 'ArbitrageAgent', 'VolumeAnalyzer'])

    socketio.emit('trading_signal', signal)

    # Also emit as notification
    notification = {
        'type': signal['type'].lower(),
        'message': signal['message'],
        'agent': signal['agent'],
        'symbol': signal.get('symbol'),
        'timestamp': signal['timestamp']
    }
    socketio.emit('notification', notification)

@socketio.on('connect')
def handle_connect():
    print('📊 Dashboard client connected')
    # Send initial system status
    initial_data = get_real_system_status()
    emit('system_update', initial_data)
    # Send initial agent data
    initial_agents = get_real_agent_data()
    emit('agents_update', initial_agents)

@socketio.on('disconnect')
def handle_disconnect():
    print('📊 Dashboard client disconnected')

@socketio.on('subscribe_agent_updates')
def handle_subscribe_agent_updates(data):
    """Subscribe to specific agent updates"""
    agent_ids = data.get('agent_ids', [])
    print(f'📊 Client subscribed to agent updates: {agent_ids}')
    emit('subscription_confirmed', {'type': 'agent_updates', 'agent_ids': agent_ids})

@socketio.on('subscribe_system_metrics')
def handle_subscribe_system_metrics():
    """Subscribe to system metrics updates"""
    print('📊 Client subscribed to system metrics')
    emit('subscription_confirmed', {'type': 'system_metrics'})

@socketio.on('request_agent_details')
def handle_request_agent_details(data):
    """Request detailed information about specific agents"""
    agent_ids = data.get('agent_ids', [])
    if ORCHESTRATOR_AVAILABLE and orchestrator:
        agent_details = []
        for agent_id in agent_ids:
            agent_info = orchestrator.agent_registry.get_agent_info(agent_id)
            if agent_info:
                agent_details.append({
                    'agent_id': agent_info.agent_id,
                    'agent_type': agent_info.agent_type,
                    'capabilities': [cap.value for cap in agent_info.capabilities],
                    'status': agent_info.status.value,
                    'health_score': agent_info.health_score,
                    'load_factor': agent_info.load_factor,
                    'performance_metrics': agent_info.performance_metrics,
                    'registered_at': agent_info.registered_at,
                    'last_heartbeat': agent_info.last_heartbeat,
                    'version': agent_info.version,
                    'tags': list(agent_info.tags)
                })
        emit('agent_details', agent_details)
    else:
        emit('agent_details', [])

@socketio.on('send_agent_command')
def handle_send_agent_command(data):
    """Send a command to an agent through the orchestrator"""
    agent_id = data.get('agent_id')
    command = data.get('command')
    parameters = data.get('parameters', {})

    if ORCHESTRATOR_AVAILABLE and orchestrator and agent_id:
        try:
            # Create a command message
            command_message = {
                'type': 'command',
                'command': command,
                'parameters': parameters,
                'timestamp': datetime.now().isoformat()
            }

            # Send through communication bus (this would need to be implemented)
            # For now, just acknowledge
            emit('command_acknowledged', {
                'agent_id': agent_id,
                'command': command,
                'status': 'queued'
            })
            print(f'📊 Command sent to agent {agent_id}: {command}')
        except Exception as e:
            emit('command_error', {
                'agent_id': agent_id,
                'command': command,
                'error': str(e)
            })
    else:
        emit('command_error', {
            'agent_id': agent_id,
            'command': command,
            'error': 'Orchestrator not available'
        })

async def start_orchestrator():
    """Start the orchestrator in a separate task"""
    await initialize_orchestrator()

if __name__ == '__main__':
    start_time = time.time()

    # Initialize orchestrator if available
    if ORCHESTRATOR_AVAILABLE:
        try:
            # Run orchestrator initialization in event loop
            loop = event_loop_manager.get_loop()
            loop.run_until_complete(start_orchestrator())
        except Exception as e:
            print(f"Failed to start orchestrator: {e}")

    # Start background update thread
    update_thread = threading.Thread(target=generate_real_time_updates, daemon=True)
    update_thread.start()

    print("🧠 Starting NeuroFlux Dashboard API Server...")
    print("📊 Dashboard: http://localhost:3000")
    print("🔌 API: http://localhost:5001")
    print("🌐 WebSocket: ws://localhost:5001")
    print(f"🤖 Orchestrator: {'✅ Connected' if orchestrator else '❌ Mock mode'}")
    print(f"🧠 ML Models: {'✅ Available' if ML_AVAILABLE else '❌ Unavailable'}")
    print(f"📈 CCXT Exchanges: {'✅ Available' if CCXT_AVAILABLE else '❌ Unavailable'}")
    print(f"🔄 Real-Time Bus: {'✅ Active' if rt_agent_bus else '❌ Unavailable'}")

    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)