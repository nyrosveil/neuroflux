"""
🧠 NeuroFlux Dashboard API Server
Flask + Socket.IO server for real-time dashboard data

Built with love by Nyros Veil 🚀
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import time
import random
from datetime import datetime
import threading
import os
import numpy as np
import pandas as pd

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

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

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

@app.route('/api/status')
def get_system_status():
    """Get current system status"""
    return jsonify(mock_data)

@app.route('/api/agents')
def get_agents():
    """Get current agent data"""
    return jsonify(mock_agents)

@app.route('/api/health')
def health_check():
    """System health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': time.time() - start_time,
        'ml_available': ML_AVAILABLE
    })

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

def generate_mock_updates():
    """Generate mock real-time updates"""
    while True:
        time.sleep(5)  # Update every 5 seconds

        # Update flux level slightly
        mock_data['flux_state']['level'] = max(0.1, min(1.0,
            mock_data['flux_state']['level'] + random.uniform(-0.1, 0.1)))
        mock_data['flux_state']['last_update'] = datetime.now().isoformat()

        # Update agent data
        for agent in mock_agents:
            if random.random() < 0.3:  # 30% chance of update
                agent['execution_time'] = round(random.uniform(0.5, 3.0), 1)
                agent['success'] = random.random() > 0.2  # 80% success rate

        # Emit socket updates
        socketio.emit('system_update', mock_data)

        # Random agent updates
        if random.random() < 0.4:
            agent_update = random.choice(mock_agents).copy()
            agent_update['timestamp'] = datetime.now().isoformat()
            socketio.emit('agent_update', agent_update)

        # Random notifications
        if random.random() < 0.2:
            notifications = [
                {'type': 'INFO', 'message': 'Agent cycle completed', 'agent': 'System'},
                {'type': 'SUCCESS', 'message': 'Trade executed successfully', 'agent': 'TradingAgent'},
                {'type': 'WARNING', 'message': 'High volatility detected', 'agent': 'RiskAgent'},
                {'type': 'ERROR', 'message': 'API rate limit reached', 'agent': 'ResearchAgent'}
            ]
            notification = random.choice(notifications)
            notification['timestamp'] = datetime.now().isoformat()
            socketio.emit('notification', notification)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('system_update', mock_data)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    start_time = time.time()

    # Start background update thread
    update_thread = threading.Thread(target=generate_mock_updates, daemon=True)
    update_thread.start()

    print("🧠 Starting NeuroFlux Dashboard API Server...")
    print("📊 Dashboard: http://localhost:3000")
    print("🔌 API: http://localhost:5001")
    print("🌐 WebSocket: ws://localhost:5001")

    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)