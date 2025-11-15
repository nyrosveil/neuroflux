import React, { createContext, useContext, useState, useCallback } from 'react';

// Error types
export const ERROR_TYPES = {
  NETWORK: 'network',
  API: 'api',
  WEBSOCKET: 'websocket',
  VALIDATION: 'validation',
  AUTHENTICATION: 'authentication',
  PERMISSION: 'permission',
  TIMEOUT: 'timeout',
  UNKNOWN: 'unknown'
};

// Error severity levels
export const ERROR_SEVERITY = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical'
};

// Notification types
export const NOTIFICATION_TYPES = {
  SUCCESS: 'success',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info',
  LOADING: 'loading'
};

// Error context
const ErrorContext = createContext();

// Error provider component
export const ErrorProvider = ({ children }) => {
  const [errors, setErrors] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [loadingStates, setLoadingStates] = useState(new Map());

  // Add error to queue
  const addError = useCallback((error, options = {}) => {
    const errorObj = {
      id: Date.now() + Math.random(),
      type: ERROR_TYPES.UNKNOWN,
      severity: ERROR_SEVERITY.MEDIUM,
      message: 'An unexpected error occurred',
      details: null,
      timestamp: new Date().toISOString(),
      retryable: false,
      retryAction: null,
      autoHide: true,
      hideDelay: 5000,
      ...error,
      ...options
    };

    setErrors(prev => [...prev, errorObj]);

    // Auto-hide non-critical errors
    if (errorObj.autoHide && errorObj.severity !== ERROR_SEVERITY.CRITICAL) {
      setTimeout(() => {
        removeError(errorObj.id);
      }, errorObj.hideDelay);
    }

    // Log error for debugging
    console.error('Error added:', errorObj);

    return errorObj.id;
  }, []);

  // Remove error from queue
  const removeError = useCallback((errorId) => {
    setErrors(prev => prev.filter(error => error.id !== errorId));
  }, []);

  // Clear all errors
  const clearErrors = useCallback(() => {
    setErrors([]);
  }, []);

  // Add notification
  const addNotification = useCallback((notification, options = {}) => {
    const notificationObj = {
      id: Date.now() + Math.random(),
      type: NOTIFICATION_TYPES.INFO,
      title: '',
      message: '',
      timestamp: new Date().toISOString(),
      autoHide: true,
      hideDelay: 4000,
      action: null,
      ...notification,
      ...options
    };

    setNotifications(prev => [...prev, notificationObj]);

    // Auto-hide notifications
    if (notificationObj.autoHide) {
      setTimeout(() => {
        removeNotification(notificationObj.id);
      }, notificationObj.hideDelay);
    }

    return notificationObj.id;
  }, []);

  // Remove notification
  const removeNotification = useCallback((notificationId) => {
    setNotifications(prev => prev.filter(notification => notification.id !== notificationId));
  }, []);

  // Clear all notifications
  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  // Loading state management
  const setLoading = useCallback((key, loading, message = '') => {
    setLoadingStates(prev => {
      const newMap = new Map(prev);
      if (loading) {
        newMap.set(key, { loading: true, message, startTime: Date.now() });
      } else {
        newMap.delete(key);
      }
      return newMap;
    });
  }, []);

  const isLoading = useCallback((key) => {
    return loadingStates.has(key);
  }, [loadingStates]);

  const getLoadingMessage = useCallback((key) => {
    const state = loadingStates.get(key);
    return state ? state.message : '';
  }, [loadingStates]);

  // Retry error action
  const retryError = useCallback((errorId) => {
    const error = errors.find(e => e.id === errorId);
    if (error && error.retryable && error.retryAction) {
      error.retryAction();
      removeError(errorId);
    }
  }, [errors, removeError]);

  // Network error handler
  const handleNetworkError = useCallback((error, context = {}) => {
    addError({
      type: ERROR_TYPES.NETWORK,
      severity: ERROR_SEVERITY.HIGH,
      message: 'Network connection lost',
      details: 'Unable to connect to NeuroFlux servers. Please check your internet connection.',
      retryable: true,
      retryAction: () => window.location.reload(),
      ...context
    });
  }, [addError]);

  // API error handler
  const handleApiError = useCallback((error, endpoint = '', context = {}) => {
    const statusCode = error.status || error.statusCode || 500;
    let severity = ERROR_SEVERITY.MEDIUM;
    let message = 'API request failed';
    let details = 'An error occurred while communicating with the server.';
    let retryable = true;

    // Customize based on status code
    switch (statusCode) {
      case 400:
        severity = ERROR_SEVERITY.LOW;
        message = 'Invalid request';
        details = 'The request contained invalid data.';
        retryable = false;
        break;
      case 401:
        severity = ERROR_SEVERITY.HIGH;
        message = 'Authentication required';
        details = 'Please log in to continue.';
        retryable = false;
        break;
      case 403:
        severity = ERROR_SEVERITY.MEDIUM;
        message = 'Access denied';
        details = 'You do not have permission to perform this action.';
        retryable = false;
        break;
      case 404:
        severity = ERROR_SEVERITY.LOW;
        message = 'Resource not found';
        details = 'The requested resource could not be found.';
        retryable = false;
        break;
      case 429:
        severity = ERROR_SEVERITY.MEDIUM;
        message = 'Rate limit exceeded';
        details = 'Too many requests. Please wait before trying again.';
        retryable = true;
        break;
      case 500:
        severity = ERROR_SEVERITY.HIGH;
        message = 'Server error';
        details = 'A server error occurred. Please try again later.';
        retryable = true;
        break;
      default:
        if (statusCode >= 500) {
          severity = ERROR_SEVERITY.HIGH;
        }
    }

    addError({
      type: ERROR_TYPES.API,
      severity,
      message,
      details: `${details} (Endpoint: ${endpoint})`,
      retryable,
      retryAction: context.retryAction,
      ...context
    });
  }, [addError]);

  // WebSocket error handler
  const handleWebSocketError = useCallback((error, context = {}) => {
    addError({
      type: ERROR_TYPES.WEBSOCKET,
      severity: ERROR_SEVERITY.MEDIUM,
      message: 'Real-time connection lost',
      details: 'Lost connection to live data feed. Attempting to reconnect...',
      retryable: false, // Auto-reconnect handled by socket library
      ...context
    });
  }, [addError]);

  // Success notification helper
  const showSuccess = useCallback((message, title = 'Success', options = {}) => {
    addNotification({
      type: NOTIFICATION_TYPES.SUCCESS,
      title,
      message,
      ...options
    });
  }, [addNotification]);

  // Error notification helper
  const showError = useCallback((message, title = 'Error', options = {}) => {
    addNotification({
      type: NOTIFICATION_TYPES.ERROR,
      title,
      message,
      ...options
    });
  }, [addNotification]);

  // Warning notification helper
  const showWarning = useCallback((message, title = 'Warning', options = {}) => {
    addNotification({
      type: NOTIFICATION_TYPES.WARNING,
      title,
      message,
      ...options
    });
  }, [addNotification]);

  // Info notification helper
  const showInfo = useCallback((message, title = 'Info', options = {}) => {
    addNotification({
      type: NOTIFICATION_TYPES.INFO,
      title,
      message,
      ...options
    });
  }, [addNotification]);

  const contextValue = {
    // Errors
    errors,
    addError,
    removeError,
    clearErrors,
    retryError,

    // Notifications
    notifications,
    addNotification,
    removeNotification,
    clearNotifications,

    // Loading states
    setLoading,
    isLoading,
    getLoadingMessage,

    // Error handlers
    handleNetworkError,
    handleApiError,
    handleWebSocketError,

    // Notification helpers
    showSuccess,
    showError,
    showWarning,
    showInfo
  };

  return (
    <ErrorContext.Provider value={contextValue}>
      {children}
    </ErrorContext.Provider>
  );
};

// Custom hook to use error context
export const useError = () => {
  const context = useContext(ErrorContext);
  if (!context) {
    throw new Error('useError must be used within an ErrorProvider');
  }
  return context;
};

// Higher-order component for error boundary
export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error,
      errorInfo
    });

    // Log error
    console.error('Error boundary caught an error:', error, errorInfo);

    // Report error if error reporting is available
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-boundary-content">
            <h2>🚨 Something went wrong</h2>
            <p>An unexpected error occurred. Please refresh the page to continue.</p>
            <div className="error-actions">
              <button
                onClick={() => window.location.reload()}
                className="error-retry-btn"
              >
                🔄 Refresh Page
              </button>
              {process.env.NODE_ENV === 'development' && (
                <details className="error-details">
                  <summary>Error Details (Development)</summary>
                  <pre>{this.state.error && this.state.error.toString()}</pre>
                  <pre>{this.state.errorInfo.componentStack}</pre>
                </details>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}