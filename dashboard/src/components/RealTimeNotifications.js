import React from 'react';
import './RealTimeNotifications.css';

function RealTimeNotifications({ notifications }) {
  if (!notifications || notifications.length === 0) {
    return (
      <div className="real-time-notifications">
        <h3>🔔 Real-time Notifications</h3>
        <p>No recent notifications</p>
      </div>
    );
  }

  return (
    <div className="real-time-notifications">
      <h3>🔔 Real-time Notifications</h3>

      <div className="notifications-list">
        {notifications.slice(0, 10).map((notification, index) => (
          <div key={index} className={`notification-item ${notification.type || 'info'}`}>
            <div className="notification-header">
              <span className="notification-type">{notification.type || 'INFO'}</span>
              <span className="notification-time">
                {notification.timestamp ?
                  new Date(notification.timestamp).toLocaleTimeString() :
                  'Now'
                }
              </span>
            </div>
            <div className="notification-message">
              {notification.message || notification.content || 'New notification'}
            </div>
            {notification.agent && (
              <div className="notification-agent">
                Agent: {notification.agent}
              </div>
            )}
          </div>
        ))}
      </div>

      {notifications.length > 10 && (
        <div className="notifications-footer">
          And {notifications.length - 10} more notifications...
        </div>
      )}
    </div>
  );
}

export default RealTimeNotifications;