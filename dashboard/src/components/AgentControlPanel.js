import React, { useState, useEffect } from 'react';
import { useTouchGestures, useTouchFeedback } from '../hooks/useTouchGestures';
import { useError } from '../contexts/ErrorContext';
import BottomSheet from './BottomSheet';
import './AgentControlPanel.css';

function AgentControlPanel({ agents, socket }) {
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [agentConfigs, setAgentConfigs] = useState({});
  const [bulkOperation, setBulkOperation] = useState(null);
  const { showSuccess, showError, handleWebSocketError } = useError();
  const [currentAgentIndex, setCurrentAgentIndex] = useState(0);

  // Touch gestures for agent navigation
  const { bindSwipe } = useTouchGestures({
    onSwipeLeft: () => {
      if (agents && agents.length > 0) {
        setCurrentAgentIndex(prev => (prev + 1) % agents.length);
      }
    },
    onSwipeRight: () => {
      if (agents && agents.length > 0) {
        setCurrentAgentIndex(prev => (prev - 1 + agents.length) % agents.length);
      }
    },
    swipeThreshold: 50,
  });

  // Touch feedback for buttons
  const { addTouchFeedback } = useTouchFeedback();

  // WebSocket error handling
  useEffect(() => {
    if (!socket) return;

    const handleSocketError = (error) => {
      handleWebSocketError(error, {
        context: 'Agent Control Panel',
        message: 'Lost connection to agent control system'
      });
    };

    const handleCommandResponse = (response) => {
      if (response.success === false) {
        showError(
          'Command Failed',
          `Agent command failed: ${response.message || 'Unknown error'}`
        );
      }
    };

    socket.on('error', handleSocketError);
    socket.on('command_response', handleCommandResponse);

    return () => {
      socket.off('error', handleSocketError);
      socket.off('command_response', handleCommandResponse);
    };
  }, [socket, handleWebSocketError, showError]);

  // Agent control actions
  const handleAgentAction = (agentId, action) => {
    if (!socket) {
      showError('Connection Error', 'No WebSocket connection available');
      return;
    }

    try {
      const command = {
        agent_id: agentId,
        command: action,
        timestamp: new Date().toISOString()
      };

      socket.emit('send_agent_command', command);

      // Show success notification
      showSuccess(
        `Agent ${action} command sent successfully`,
        `Agent ${agentId}`,
        { hideDelay: 2000 }
      );

      console.log(`📊 Sent ${action} command to agent ${agentId}`);
    } catch (error) {
      showError(
        'Command Failed',
        `Failed to send ${action} command to agent ${agentId}`
      );
      console.error(`❌ Failed to send ${action} command:`, error);
    }
  };

  const handleBulkAction = (action) => {
    if (!agents || !socket) return;

    const activeAgents = agents.filter(agent => agent.status === 'active');
    activeAgents.forEach(agent => {
      handleAgentAction(agent.id, action);
    });

    setBulkOperation(`${action} all active agents`);
    setTimeout(() => setBulkOperation(null), 3000);
  };

  const handleConfigUpdate = (agentId, config) => {
    if (!socket) {
      showError('Connection Error', 'No WebSocket connection available');
      return;
    }

    try {
      const command = {
        agent_id: agentId,
        command: 'update_config',
        parameters: config,
        timestamp: new Date().toISOString()
      };

      socket.emit('send_agent_command', command);
      setAgentConfigs(prev => ({
        ...prev,
        [agentId]: config
      }));

      showSuccess(
        'Configuration updated successfully',
        `Agent ${agentId}`,
        { hideDelay: 3000 }
      );

      setShowConfigModal(false);
      setSelectedAgent(null);
    } catch (error) {
      showError(
        'Configuration Failed',
        `Failed to update configuration for agent ${agentId}`
      );
      console.error(`❌ Failed to update config:`, error);
    }
  };

  const getAgentStatusColor = (status) => {
    switch (status) {
      case 'active': return '#6bcf7f';
      case 'inactive': return '#ff6b6b';
      case 'paused': return '#ffd93d';
      case 'error': return '#ff4757';
      default: return '#bdc3c7';
    }
  };

  const getAgentStatusIcon = (status) => {
    switch (status) {
      case 'active': return '🟢';
      case 'inactive': return '🔴';
      case 'paused': return '🟡';
      case 'error': return '❌';
      default: return '⚪';
    }
  };

  if (!agents || agents.length === 0) {
    return (
      <div className="agent-control-panel loading">
        <h3>🎮 Agent Control Panel</h3>
        <p>Loading agent data...</p>
      </div>
    );
  }

  return (
    <div className="agent-control-panel">
      <div className="panel-header">
        <h3>🎮 Agent Control Panel</h3>
        <div className="bulk-controls">
          <button
            className="bulk-btn start-all"
            onClick={() => handleBulkAction('start')}
            disabled={!agents.some(a => a.status !== 'active')}
          >
            ▶️ Start All
          </button>
          <button
            className="bulk-btn stop-all"
            onClick={() => handleBulkAction('stop')}
            disabled={!agents.some(a => a.status === 'active')}
          >
            ⏹️ Stop All
          </button>
          <button
            className="bulk-btn pause-all"
            onClick={() => handleBulkAction('pause')}
            disabled={!agents.some(a => a.status === 'active')}
          >
            ⏸️ Pause All
          </button>
        </div>
        {bulkOperation && (
          <div className="bulk-feedback">
            ✅ {bulkOperation}
          </div>
        )}
      </div>

      <div className="agent-grid" {...bindSwipe()}>
        {/* Mobile: Show current agent with navigation indicators */}
        <div className="mobile-agent-nav">
          <div className="agent-indicators">
            {agents.map((_, index) => (
              <div
                key={index}
                className={`agent-indicator ${index === currentAgentIndex ? 'active' : ''}`}
                onClick={() => setCurrentAgentIndex(index)}
              />
            ))}
          </div>
          <div className="agent-counter">
            {currentAgentIndex + 1} / {agents.length}
          </div>
        </div>

        {/* Desktop: Show all agents, Mobile: Show current agent */}
        <div className="agent-container">
          {agents.map((agent, index) => (
            <div
              key={agent.id}
              className={`agent-card ${index === currentAgentIndex ? 'active' : 'hidden-mobile'}`}
              ref={(el) => el && addTouchFeedback(el)}
            >
            <div className="agent-header">
              <div className="agent-info">
                <div className="agent-name">
                  {getAgentStatusIcon(agent.status)} {agent.name}
                </div>
                <div className="agent-type">{agent.type}</div>
              </div>
              <div
                className="status-indicator"
                style={{ backgroundColor: getAgentStatusColor(agent.status) }}
              ></div>
            </div>

            <div className="agent-metrics">
              <div className="metric">
                <span className="label">Health:</span>
                <span className="value">{agent.health_score || 0}%</span>
              </div>
              <div className="metric">
                <span className="label">Load:</span>
                <span className="value">{agent.load_factor || 0}%</span>
              </div>
              <div className="metric">
                <span className="label">Tasks:</span>
                <span className="value">{agent.task_count || 0}</span>
              </div>
            </div>

            <div className="agent-controls">
              <button
                className="control-btn start"
                onClick={() => handleAgentAction(agent.id, 'start')}
                disabled={agent.status === 'active'}
              >
                ▶️ Start
              </button>
              <button
                className="control-btn stop"
                onClick={() => handleAgentAction(agent.id, 'stop')}
                disabled={agent.status === 'inactive'}
              >
                ⏹️ Stop
              </button>
              <button
                className="control-btn pause"
                onClick={() => handleAgentAction(agent.id, 'pause')}
                disabled={agent.status !== 'active'}
              >
                ⏸️ Pause
              </button>
              <button
                className="control-btn config"
                onClick={() => {
                  setSelectedAgent(agent);
                  setShowConfigModal(true);
                }}
              >
                ⚙️ Config
              </button>
            </div>

            <div className="agent-capabilities">
              {agent.capabilities && agent.capabilities.slice(0, 3).map(cap => (
                <span key={cap} className="capability-tag">{cap}</span>
              ))}
              {agent.capabilities && agent.capabilities.length > 3 && (
                <span className="capability-tag more">+{agent.capabilities.length - 3}</span>
              )}
            </div>
          </div>
        ))}

        {/* Mobile swipe hint */}
        <div className="mobile-swipe-hint">
          👆 Swipe left/right to navigate agents
        </div>
      </div></div>

      {/* Configuration Bottom Sheet */}
      <BottomSheet
        isOpen={showConfigModal}
        onClose={() => {
          setShowConfigModal(false);
          setSelectedAgent(null);
        }}
        title={`Configure ${selectedAgent?.name || 'Agent'}`}
        height="70vh"
      >
        {selectedAgent && (
          <AgentConfigForm
            agent={selectedAgent}
            currentConfig={agentConfigs[selectedAgent.id] || {}}
            onSave={(config) => handleConfigUpdate(selectedAgent.id, config)}
          />
        )}
      </BottomSheet>
    </div>
  );
}

function AgentConfigForm({ agent, currentConfig, onSave }) {
  const [config, setConfig] = useState({
    flux_sensitivity: currentConfig.flux_sensitivity || 0.5,
    priority: currentConfig.priority || 'medium',
    max_concurrent_tasks: currentConfig.max_concurrent_tasks || 5,
    timeout_seconds: currentConfig.timeout_seconds || 300,
    retry_attempts: currentConfig.retry_attempts || 3,
    ...currentConfig
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(config);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="config-section">
        <h5>🤖 Agent Settings</h5>
        <div className="form-group">
          <label>Flux Sensitivity (0-1):</label>
          <div className="range-input">
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={config.flux_sensitivity}
              onChange={e => setConfig(prev => ({...prev, flux_sensitivity: parseFloat(e.target.value)}))}
            />
            <span className="range-value">{config.flux_sensitivity}</span>
          </div>
        </div>

        <div className="form-group">
          <label>Priority Level:</label>
          <select
            value={config.priority}
            onChange={e => setConfig(prev => ({...prev, priority: e.target.value}))}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>
      </div>

      <div className="config-section">
        <h5>⚙️ Performance Settings</h5>
        <div className="form-group">
          <label>Max Concurrent Tasks:</label>
          <input
            type="number"
            min="1"
            max="20"
            value={config.max_concurrent_tasks}
            onChange={e => setConfig(prev => ({...prev, max_concurrent_tasks: parseInt(e.target.value)}))}
          />
        </div>

        <div className="form-group">
          <label>Timeout (seconds):</label>
          <input
            type="number"
            min="30"
            max="3600"
            value={config.timeout_seconds}
            onChange={e => setConfig(prev => ({...prev, timeout_seconds: parseInt(e.target.value)}))}
          />
        </div>

        <div className="form-group">
          <label>Retry Attempts:</label>
          <input
            type="number"
            min="0"
            max="10"
            value={config.retry_attempts}
            onChange={e => setConfig(prev => ({...prev, retry_attempts: parseInt(e.target.value)}))}
          />
        </div>
      </div>

      <div className="form-actions">
        <button type="submit" className="save-btn">
          💾 Save Configuration
        </button>
      </div>
    </form>
  );
}

export default AgentControlPanel;