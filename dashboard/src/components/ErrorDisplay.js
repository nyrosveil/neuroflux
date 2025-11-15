import React from 'react';
import { useError, ERROR_TYPES, ERROR_SEVERITY } from '../contexts/ErrorContext';
import './ErrorDisplay.css';

function ErrorDisplay() {
  const { errors, removeError, retryError } = useError();

  if (errors.length === 0) return null;

  return (
    <div className="error-display">
      {errors.map(error => (
        <ErrorItem
          key={error.id}
          error={error}
          onDismiss={() => removeError(error.id)}
          onRetry={() => retryError(error.id)}
        />
      ))}
    </div>
  );
}

function ErrorItem({ error, onDismiss, onRetry }) {
  const getErrorIcon = (type) => {
    switch (type) {
      case ERROR_TYPES.NETWORK:
        return '📡';
      case ERROR_TYPES.API:
        return '🔌';
      case ERROR_TYPES.WEBSOCKET:
        return '🔗';
      case ERROR_TYPES.VALIDATION:
        return '⚠️';
      case ERROR_TYPES.AUTHENTICATION:
        return '🔐';
      case ERROR_TYPES.PERMISSION:
        return '🚫';
      case ERROR_TYPES.TIMEOUT:
        return '⏱️';
      default:
        return '❌';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case ERROR_SEVERITY.CRITICAL:
        return '#e74c3c';
      case ERROR_SEVERITY.HIGH:
        return '#e67e22';
      case ERROR_SEVERITY.MEDIUM:
        return '#f39c12';
      case ERROR_SEVERITY.LOW:
        return '#27ae60';
      default:
        return '#95a5a6';
    }
  };

  return (
    <div
      className={`error-item error-${error.severity}`}
      style={{ borderLeftColor: getSeverityColor(error.severity) }}
    >
      <div className="error-header">
        <div className="error-icon">
          {getErrorIcon(error.type)}
        </div>
        <div className="error-content">
          <div className="error-title">{error.message}</div>
          {error.details && (
            <div className="error-details">{error.details}</div>
          )}
        </div>
        <button className="error-dismiss" onClick={onDismiss}>
          ✕
        </button>
      </div>

      <div className="error-footer">
        <div className="error-meta">
          <span className="error-type">{error.type.replace('_', ' ').toUpperCase()}</span>
          <span className="error-time">
            {new Date(error.timestamp).toLocaleTimeString()}
          </span>
        </div>

        <div className="error-actions">
          {error.retryable && error.retryAction && (
            <button className="error-retry" onClick={onRetry}>
              🔄 Retry
            </button>
          )}
          <button className="error-report">
            📋 Report
          </button>
        </div>
      </div>
    </div>
  );
}

export default ErrorDisplay;