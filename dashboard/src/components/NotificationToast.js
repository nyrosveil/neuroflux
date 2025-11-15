import React from 'react';
import { useError, NOTIFICATION_TYPES } from '../contexts/ErrorContext';
import './NotificationToast.css';

function NotificationToast() {
  const { notifications, removeNotification } = useError();

  if (notifications.length === 0) return null;

  return (
    <div className="notification-container">
      {notifications.map(notification => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onDismiss={() => removeNotification(notification.id)}
        />
      ))}
    </div>
  );
}

function NotificationItem({ notification, onDismiss }) {
  const getNotificationIcon = (type) => {
    switch (type) {
      case NOTIFICATION_TYPES.SUCCESS:
        return '✅';
      case NOTIFICATION_TYPES.ERROR:
        return '❌';
      case NOTIFICATION_TYPES.WARNING:
        return '⚠️';
      case NOTIFICATION_TYPES.INFO:
        return 'ℹ️';
      case NOTIFICATION_TYPES.LOADING:
        return '🔄';
      default:
        return '📢';
    }
  };

  const getNotificationClass = (type) => {
    switch (type) {
      case NOTIFICATION_TYPES.SUCCESS:
        return 'notification-success';
      case NOTIFICATION_TYPES.ERROR:
        return 'notification-error';
      case NOTIFICATION_TYPES.WARNING:
        return 'notification-warning';
      case NOTIFICATION_TYPES.INFO:
        return 'notification-info';
      case NOTIFICATION_TYPES.LOADING:
        return 'notification-loading';
      default:
        return 'notification-info';
    }
  };

  return (
    <div className={`notification-item ${getNotificationClass(notification.type)}`}>
      <div className="notification-icon">
        {getNotificationIcon(notification.type)}
      </div>

      <div className="notification-content">
        {notification.title && (
          <div className="notification-title">{notification.title}</div>
        )}
        <div className="notification-message">{notification.message}</div>
      </div>

      <button className="notification-dismiss" onClick={onDismiss}>
        ✕
      </button>

      {/* Progress bar for auto-hide */}
      {notification.autoHide && (
        <div className="notification-progress">
          <div
            className="progress-bar"
            style={{ animationDuration: `${notification.hideDelay}ms` }}
          />
        </div>
      )}

      {/* Action button if provided */}
      {notification.action && (
        <div className="notification-actions">
          <button
            className="notification-action-btn"
            onClick={notification.action.onClick}
          >
            {notification.action.label}
          </button>
        </div>
      )}
    </div>
  );
}

export default NotificationToast;