"""
🧠 NeuroFlux Dashboard API Server
Flask + Socket.IO server for real-time dashboard data

Built with love by Nyros Veil 🚀
"""

from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import time
import random
from datetime import datetime
import threading
import os

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
        'uptime': time.time() - start_time
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