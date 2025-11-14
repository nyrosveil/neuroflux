import React, { useState, useEffect } from 'react';
import './App.css';
import SystemStatus from './components/SystemStatus';
import AgentMetrics from './components/AgentMetrics';
import PerformanceCharts from './components/PerformanceCharts';
import RealTimeNotifications from './components/RealTimeNotifications';
import { io } from 'socket.io-client';

function App() {
  const [systemData, setSystemData] = useState(null);
  const [agentData, setAgentData] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    // Initialize WebSocket connection
    const newSocket = io('http://localhost:5001');
    setSocket(newSocket);

    // Fetch initial data
    fetchSystemData();
    fetchAgentData();

    // WebSocket event listeners
    newSocket.on('system_update', (data) => {
      setSystemData(data);
    });

    newSocket.on('agent_update', (data) => {
      setAgentData(prev => [...prev, data]);
    });

    newSocket.on('notification', (notification) => {
      setNotifications(prev => [notification, ...prev.slice(0, 9)]); // Keep last 10
    });

    return () => newSocket.close();
  }, []);

  const fetchSystemData = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/status');
      const data = await response.json();
      setSystemData(data);
    } catch (error) {
      console.error('Failed to fetch system data:', error);
    }
  };

  const fetchAgentData = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/agents');
      const data = await response.json();
      setAgentData(data);
    } catch (error) {
      console.error('Failed to fetch agent data:', error);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🧠 NeuroFlux Dashboard</h1>
        <p>Real-time Multi-Agent Trading System Monitor</p>
      </header>

      <div className="dashboard-container">
        <div className="dashboard-grid">
          <SystemStatus data={systemData} />
          <AgentMetrics data={agentData} />
          <PerformanceCharts data={agentData} />
          <RealTimeNotifications notifications={notifications} />
        </div>
      </div>

      <footer className="app-footer">
        <p>Built with love by Nyros Veil 🚀 | Phase 4.4 Dashboard</p>
      </footer>
    </div>
  );
}

export default App;
