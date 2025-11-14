import React from 'react';
import './SystemStatus.css';

function SystemStatus({ data }) {
  if (!data) {
    return (
      <div className="system-status loading">
        <h3>System Status</h3>
        <p>Loading system data...</p>
      </div>
    );
  }

  const fluxLevel = data.flux_state?.level || 0;
  const activeAgents = data.initialized_agents || 0;
  const totalAgents = data.total_agents || 0;
  const analyticsEnabled = data.analytics?.enabled || false;

  return (
    <div className="system-status">
      <h3>🖥️ System Status</h3>

      <div className="status-grid">
        <div className="status-item">
          <span className="label">Flux Level</span>
          <div className="flux-indicator">
            <div
              className="flux-bar"
              style={{ width: `${fluxLevel * 100}%` }}
            ></div>
            <span className="flux-value">{(fluxLevel * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="status-item">
          <span className="label">Active Agents</span>
          <span className="value">{activeAgents}/{totalAgents}</span>
        </div>

        <div className="status-item">
          <span className="label">Analytics</span>
          <span className={`status ${analyticsEnabled ? 'active' : 'inactive'}`}>
            {analyticsEnabled ? '✅ Active' : '❌ Inactive'}
          </span>
        </div>

        <div className="status-item">
          <span className="label">Market Stability</span>
          <span className="value">
            {data.flux_state?.market_stability ?
              `${(data.flux_state.market_stability * 100).toFixed(1)}%` :
              'N/A'
            }
          </span>
        </div>
      </div>

      <div className="last-update">
        Last updated: {data.flux_state?.last_update ?
          new Date(data.flux_state.last_update).toLocaleTimeString() :
          'Never'
        }
      </div>
    </div>
  );
}

export default SystemStatus;