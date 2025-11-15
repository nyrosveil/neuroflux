// Service Worker Registration for PWA
export const registerServiceWorker = async () => {
  if ('serviceWorker' in navigator) {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js', {
        scope: '/'
      });

      console.log('Service Worker registered successfully:', registration.scope);

      // Handle updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        if (newWorker) {
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              // New content is available, notify user
              showUpdateNotification();
            }
          });
        }
      });

      // Listen for messages from service worker
      navigator.serviceWorker.addEventListener('message', (event) => {
        console.log('Message from service worker:', event.data);
      });

      return registration;
    } catch (error) {
      console.error('Service Worker registration failed:', error);
      return null;
    }
  } else {
    console.log('Service Worker not supported');
    return null;
  }
};

export const unregisterServiceWorker = async () => {
  if ('serviceWorker' in navigator) {
    try {
      const registration = await navigator.serviceWorker.ready;
      await registration.unregister();
      console.log('Service Worker unregistered');
    } catch (error) {
      console.error('Service Worker unregistration failed:', error);
    }
  }
};

export const requestNotificationPermission = async () => {
  if ('Notification' in window) {
    try {
      const permission = await Notification.requestPermission();
      console.log('Notification permission:', permission);
      return permission === 'granted';
    } catch (error) {
      console.error('Error requesting notification permission:', error);
      return false;
    }
  }
  return false;
};

const showUpdateNotification = () => {
  // Create a custom update notification
  const notification = document.createElement('div');
  notification.className = 'pwa-update-notification';
  notification.innerHTML = `
    <div class="update-content">
      <span class="update-icon">🔄</span>
      <div class="update-text">
        <div class="update-title">Update Available</div>
        <div class="update-message">A new version of NeuroFlux is available</div>
      </div>
      <div class="update-actions">
        <button class="update-btn update-later">Later</button>
        <button class="update-btn update-now">Update Now</button>
      </div>
    </div>
  `;

  document.body.appendChild(notification);

  // Handle button clicks
  const laterBtn = notification.querySelector('.update-later');
  const nowBtn = notification.querySelector('.update-now');

  laterBtn.addEventListener('click', () => {
    document.body.removeChild(notification);
  });

  nowBtn.addEventListener('click', () => {
    window.location.reload();
  });

  // Auto-hide after 10 seconds
  setTimeout(() => {
    if (document.body.contains(notification)) {
      document.body.removeChild(notification);
    }
  }, 10000);
};

// Check if app is running in standalone mode (PWA)
export const isPWA = () => {
  return window.matchMedia('(display-mode: standalone)').matches ||
         window.navigator.standalone === true ||
         document.referrer.includes('android-app://');
};

// Get PWA install prompt
let deferredPrompt;
export const setupInstallPrompt = () => {
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;

    // Show install button or notification
    showInstallPrompt();
  });
};

const showInstallPrompt = () => {
  if (!deferredPrompt) return;

  const prompt = document.createElement('div');
  prompt.className = 'pwa-install-prompt';
  prompt.innerHTML = `
    <div class="install-content">
      <span class="install-icon">📱</span>
      <div class="install-text">
        <div class="install-title">Install NeuroFlux</div>
        <div class="install-message">Add to home screen for the best experience</div>
      </div>
      <div class="install-actions">
        <button class="install-btn install-later">Not Now</button>
        <button class="install-btn install-yes">Install</button>
      </div>
    </div>
  `;

  document.body.appendChild(prompt);

  const laterBtn = prompt.querySelector('.install-later');
  const yesBtn = prompt.querySelector('.install-yes');

  laterBtn.addEventListener('click', () => {
    document.body.removeChild(prompt);
  });

  yesBtn.addEventListener('click', async () => {
    document.body.removeChild(prompt);
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    console.log('Install outcome:', outcome);
    deferredPrompt = null;
  });
};