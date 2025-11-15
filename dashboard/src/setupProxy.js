const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Proxy API requests to Flask server on port 8000
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:8000',
      changeOrigin: true,
      ws: true, // Enable WebSocket proxying
      logLevel: 'debug'
    })
  );

  // Proxy WebSocket connections
  app.use(
    '/socket.io',
    createProxyMiddleware({
      target: 'http://localhost:8000',
      changeOrigin: true,
      ws: true,
      logLevel: 'debug'
    })
  );
};