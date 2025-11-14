import React from 'react';
import './AgentMetrics.css';

function AgentMetrics({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="agent-metrics loading">
        <h3>🤖 Agent Metrics</h3>
        <p>Loading agent data...</p>
      </div>
    );
  }

  // Calculate metrics from agent data
  const totalAgents = data.length;
  const activeAgents = data.filter(agent => agent.status === 'active').length;
  const successfulAgents = data.filter(agent => agent.success).length;
  const avgExecutionTime = data.reduce((sum, agent) => sum + (agent.execution_time || 0), 0) / data.length;

  return (
    <div className="agent-metrics">
      <h3>🤖 Agent Metrics</h3>

      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-label">Total Agents</span>
          <span className="metric-value">{totalAgents}</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Active Agents</span>
          <span className="metric-value active">{activeAgents}</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Success Rate</span>
          <span className="metric-value success">
            {totalAgents > 0 ? ((successfulAgents / totalAgents) * 100).toFixed(1) : 0}%
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Avg Execution</span>
          <span className="metric-value">{avgExecutionTime.toFixed(2)}s</span>
        </div>
      </div>

      <div className="agent-list">
        <h4>Agent Status</h4>
        <div className="agent-items">
          {data.slice(0, 6).map((agent, index) => (
            <div key={index} className={`agent-item ${agent.success ? 'success' : 'error'}`}>
              <span className="agent-name">{agent.agent_name || `Agent ${index + 1}`}</span>
              <span className="agent-status">
                {agent.success ? '✅' : '❌'} {agent.execution_time?.toFixed(1)}s
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AgentMetrics;