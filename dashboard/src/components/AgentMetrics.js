import React, { useState, useEffect } from 'react';
import './AgentMetrics.css';

function AgentMetrics({ data, socket }) {
  const [expandedAgent, setExpandedAgent] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [performanceHistory, setPerformanceHistory] = useState({});

  // Calculate comprehensive metrics
  const metrics = React.useMemo(() => {
    if (!data || data.length === 0) return null;

    const totalAgents = data.length;
    const activeAgents = data.filter(agent => agent.status === 'active').length;
    const healthyAgents = data.filter(agent => (agent.health_score || 0) > 70).length;
    const errorAgents = data.filter(agent => agent.status === 'error').length;

    // Performance metrics
    const avgHealthScore = data.reduce((sum, agent) => sum + (agent.health_score || 0), 0) / totalAgents;
    const avgLoadFactor = data.reduce((sum, agent) => sum + (agent.load_factor || 0), 0) / totalAgents;
    const totalTasks = data.reduce((sum, agent) => sum + (agent.task_count || 0), 0);
    const avgResponseTime = data.reduce((sum, agent) => sum + (agent.avg_response_time || 0), 0) / totalAgents;

    // Agent type distribution
    const agentTypes = {};
    data.forEach(agent => {
      agentTypes[agent.type] = (agentTypes[agent.type] || 0) + 1;
    });

    // Status distribution
    const statusDistribution = {};
    data.forEach(agent => {
      statusDistribution[agent.status] = (statusDistribution[agent.status] || 0) + 1;
    });

    return {
      totalAgents,
      activeAgents,
      healthyAgents,
      errorAgents,
      avgHealthScore,
      avgLoadFactor,
      totalTasks,
      avgResponseTime,
      agentTypes,
      statusDistribution,
      uptime: calculateUptime(data)
    };
  }, [data]);

  // Calculate system uptime
  const calculateUptime = (agents) => {
    if (!agents || agents.length === 0) return 0;
    const totalUptime = agents.reduce((sum, agent) => sum + (agent.uptime || 0), 0);
    return totalUptime / agents.length;
  };

  // Generate alerts based on agent health
  useEffect(() => {
    if (!data) return;

    const newAlerts = [];
    data.forEach(agent => {
      if ((agent.health_score || 0) < 50) {
        newAlerts.push({
          type: 'critical',
          message: `${agent.name} health critically low (${agent.health_score}%)`,
          agent: agent.name,
          timestamp: new Date().toISOString()
        });
      } else if ((agent.health_score || 0) < 70) {
        newAlerts.push({
          type: 'warning',
          message: `${agent.name} health degraded (${agent.health_score}%)`,
          agent: agent.name,
          timestamp: new Date().toISOString()
        });
      }

      if ((agent.load_factor || 0) > 90) {
        newAlerts.push({
          type: 'warning',
          message: `${agent.name} overloaded (${agent.load_factor}% load)`,
          agent: agent.name,
          timestamp: new Date().toISOString()
        });
      }
    });

    setAlerts(newAlerts.slice(0, 5)); // Keep only latest 5 alerts
  }, [data]);

  // WebSocket listeners for real-time updates
  useEffect(() => {
    if (!socket) return;

    const handleAgentUpdate = (updateData) => {
      // Update performance history for trending
      setPerformanceHistory(prev => ({
        ...prev,
        [updateData.agent_id]: [
          ...(prev[updateData.agent_id] || []).slice(-9), // Keep last 10 points
          {
            timestamp: new Date().toISOString(),
            health_score: updateData.health_score,
            load_factor: updateData.load_factor,
            response_time: updateData.response_time
          }
        ]
      }));
    };

    socket.on('agent_update', handleAgentUpdate);

    return () => {
      socket.off('agent_update', handleAgentUpdate);
    };
  }, [socket]);

  if (!metrics) {
    return (
      <div className="agent-metrics loading">
        <h3>📊 Enhanced Agent Metrics</h3>
        <p>Loading agent data...</p>
      </div>
    );
  }

  return (
    <div className="agent-metrics">
      <div className="metrics-header">
        <h3>📊 Enhanced Agent Metrics</h3>
        <div className="system-health">
          <div className="health-indicator">
            <span className={`health-dot ${getHealthStatus(metrics.avgHealthScore)}`}></span>
            <span className="health-text">
              System Health: {metrics.avgHealthScore.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Alerts Section */}
      {alerts.length > 0 && (
        <div className="alerts-section">
          <h4>🚨 Active Alerts</h4>
          <div className="alerts-list">
            {alerts.map((alert, index) => (
              <div key={index} className={`alert alert-${alert.type}`}>
                <span className="alert-icon">
                  {alert.type === 'critical' ? '🔴' : '🟡'}
                </span>
                <span className="alert-message">{alert.message}</span>
                <span className="alert-time">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Primary Metrics Grid */}
      <div className="metrics-grid">
        <div className="metric-card primary">
          <div className="metric-icon">🤖</div>
          <div className="metric-content">
            <span className="metric-label">Total Agents</span>
            <span className="metric-value">{metrics.totalAgents}</span>
          </div>
        </div>

        <div className="metric-card primary">
          <div className="metric-icon">🟢</div>
          <div className="metric-content">
            <span className="metric-label">Active Agents</span>
            <span className="metric-value active">{metrics.activeAgents}</span>
          </div>
        </div>

        <div className="metric-card primary">
          <div className="metric-icon">💚</div>
          <div className="metric-content">
            <span className="metric-label">Healthy Agents</span>
            <span className="metric-value healthy">{metrics.healthyAgents}</span>
          </div>
        </div>

        <div className="metric-card primary">
          <div className="metric-icon">⚠️</div>
          <div className="metric-content">
            <span className="metric-label">Error Agents</span>
            <span className="metric-value error">{metrics.errorAgents}</span>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="performance-metrics">
        <h4>⚡ Performance Overview</h4>
        <div className="performance-grid">
          <div className="perf-card">
            <span className="perf-label">Avg Health Score</span>
            <div className="perf-bar">
              <div
                className="perf-fill"
                style={{ width: `${metrics.avgHealthScore}%` }}
              ></div>
            </div>
            <span className="perf-value">{metrics.avgHealthScore.toFixed(1)}%</span>
          </div>

          <div className="perf-card">
            <span className="perf-label">Avg Load Factor</span>
            <div className="perf-bar">
              <div
                className="perf-fill load"
                style={{ width: `${metrics.avgLoadFactor}%` }}
              ></div>
            </div>
            <span className="perf-value">{metrics.avgLoadFactor.toFixed(1)}%</span>
          </div>

          <div className="perf-card">
            <span className="perf-label">Total Tasks</span>
            <span className="perf-value large">{metrics.totalTasks}</span>
          </div>

          <div className="perf-card">
            <span className="perf-label">Avg Response Time</span>
            <span className="perf-value">{metrics.avgResponseTime.toFixed(2)}ms</span>
          </div>
        </div>
      </div>

      {/* Agent Type Distribution */}
      <div className="distribution-section">
        <h4>📈 Agent Distribution</h4>
        <div className="distribution-grid">
          <div className="dist-card">
            <h5>By Type</h5>
            {Object.entries(metrics.agentTypes).map(([type, count]) => (
              <div key={type} className="dist-item">
                <span className="dist-label">{type}</span>
                <div className="dist-bar">
                  <div
                    className="dist-fill"
                    style={{ width: `${(count / metrics.totalAgents) * 100}%` }}
                  ></div>
                </div>
                <span className="dist-value">{count}</span>
              </div>
            ))}
          </div>

          <div className="dist-card">
            <h5>By Status</h5>
            {Object.entries(metrics.statusDistribution).map(([status, count]) => (
              <div key={status} className="dist-item">
                <span className="dist-label status">
                  <span className={`status-dot ${status}`}></span>
                  {status}
                </span>
                <div className="dist-bar">
                  <div
                    className={`dist-fill status-${status}`}
                    style={{ width: `${(count / metrics.totalAgents) * 100}%` }}
                  ></div>
                </div>
                <span className="dist-value">{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Detailed Agent Status */}
      <div className="agent-status-section">
        <h4>🔍 Agent Details</h4>
        <div className="agent-status-grid">
          {data.map(agent => (
            <AgentStatusCard
              key={agent.id}
              agent={agent}
              performanceHistory={performanceHistory[agent.id] || []}
              isExpanded={expandedAgent === agent.id}
              onToggle={() => setExpandedAgent(
                expandedAgent === agent.id ? null : agent.id
              )}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// Helper function for health status
const getHealthStatus = (score) => {
  if (score >= 80) return 'excellent';
  if (score >= 60) return 'good';
  if (score >= 40) return 'warning';
  return 'critical';
};

// Individual Agent Status Card Component
function AgentStatusCard({ agent, performanceHistory, isExpanded, onToggle }) {
  const [showPerformanceChart, setShowPerformanceChart] = useState(false);

  return (
    <div className={`agent-status-card ${isExpanded ? 'expanded' : ''}`}>
      <div className="agent-status-header" onClick={onToggle}>
        <div className="agent-basic-info">
          <div className="agent-name-type">
            <span className="agent-name">{agent.name}</span>
            <span className="agent-type">{agent.type}</span>
          </div>
          <div className="agent-status-indicator">
            <span className={`status-dot ${agent.status}`}></span>
            <span className="status-text">{agent.status}</span>
          </div>
        </div>

        <div className="agent-key-metrics">
          <div className="metric health">
            <span className="label">Health</span>
            <span className="value">{agent.health_score || 0}%</span>
          </div>
          <div className="metric load">
            <span className="label">Load</span>
            <span className="value">{agent.load_factor || 0}%</span>
          </div>
          <div className="metric tasks">
            <span className="label">Tasks</span>
            <span className="value">{agent.task_count || 0}</span>
          </div>
        </div>

        <div className="expand-toggle">
          {isExpanded ? '🔽' : '▶️'}
        </div>
      </div>

      {isExpanded && (
        <div className="agent-status-details">
          <div className="details-grid">
            <div className="detail-section">
              <h5>📊 Performance Metrics</h5>
              <div className="metrics-list">
                <div className="metric-item">
                  <span className="metric-label">Response Time:</span>
                  <span className="metric-value">{agent.avg_response_time || 0}ms</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Success Rate:</span>
                  <span className="metric-value">{agent.success_rate || 0}%</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Uptime:</span>
                  <span className="metric-value">{agent.uptime || 0}%</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Memory Usage:</span>
                  <span className="metric-value">{agent.memory_usage || 0}MB</span>
                </div>
              </div>
            </div>

            <div className="detail-section">
              <h5>⚙️ Configuration</h5>
              <div className="config-list">
                <div className="config-item">
                  <span className="config-label">Priority:</span>
                  <span className="config-value">{agent.priority || 'medium'}</span>
                </div>
                <div className="config-item">
                  <span className="config-label">Max Tasks:</span>
                  <span className="config-value">{agent.max_concurrent_tasks || 5}</span>
                </div>
                <div className="config-item">
                  <span className="config-label">Timeout:</span>
                  <span className="config-value">{agent.timeout_seconds || 300}s</span>
                </div>
              </div>
            </div>

            <div className="detail-section">
              <h5>🏷️ Capabilities</h5>
              <div className="capabilities-list">
                {agent.capabilities && agent.capabilities.map(cap => (
                  <span key={cap} className="capability-badge">{cap}</span>
                ))}
              </div>
            </div>
          </div>

          <div className="performance-toggle">
            <button
              className="chart-toggle-btn"
              onClick={() => setShowPerformanceChart(!showPerformanceChart)}
            >
              {showPerformanceChart ? '📈 Hide' : '📊 Show'} Performance Chart
            </button>
          </div>

          {showPerformanceChart && performanceHistory.length > 0 && (
            <div className="performance-chart">
              <h5>Performance History (Last 10 Updates)</h5>
              <div className="mini-chart">
                {performanceHistory.slice(-10).map((point, index) => (
                  <div key={index} className="chart-point">
                    <div className="point-time">
                      {new Date(point.timestamp).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </div>
                    <div className="point-metrics">
                      <div className="metric-bar health">
                        <div
                          className="bar-fill"
                          style={{ width: `${point.health_score || 0}%` }}
                        ></div>
                        <span className="bar-label">H:{point.health_score || 0}%</span>
                      </div>
                      <div className="metric-bar load">
                        <div
                          className="bar-fill"
                          style={{ width: `${point.load_factor || 0}%` }}
                        ></div>
                        <span className="bar-label">L:{point.load_factor || 0}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="agent-actions">
            <button className="action-btn logs">📋 View Logs</button>
            <button className="action-btn restart">🔄 Restart Agent</button>
            <button className="action-btn diagnostics">🔍 Run Diagnostics</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default AgentMetrics;