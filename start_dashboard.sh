#!/bin/bash
# NeuroFlux Dashboard Startup Script
# Starts the React dashboard on port 3000

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() { echo -e "${BLUE}[DASHBOARD]${NC} $1"; }
log_success() { echo -e "${GREEN}[DASHBOARD]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[DASHBOARD]${NC} $1"; }
log_error() { echo -e "${RED}[DASHBOARD]${NC} $1"; }

echo "📊 NeuroFlux React Dashboard (Port 3000)"
echo "========================================="

# Check if we're in the right directory
if [ ! -d "dashboard" ]; then
    log_error "dashboard directory not found. Please run from NeuroFlux root directory."
    exit 1
fi

cd dashboard

# Check if package.json exists
if [ ! -f "package.json" ]; then
    log_error "package.json not found in dashboard directory."
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    log_error "npm not found. Please install Node.js and npm."
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    log_info "Installing dependencies..."
    npm install
fi

# Port cleanup before starting
log_info "Performing port cleanup before startup..."
if [ -f "../port_manager.sh" ]; then
    ../port_manager.sh cleanup
else
    log_warning "port_manager.sh not found, skipping port cleanup"
fi

# Trap for graceful shutdown
cleanup_and_exit() {
    log_warning "Received shutdown signal, stopping dashboard..."
    pkill -TERM -f "react-scripts start" 2>/dev/null || true
    pkill -TERM -f "npm start" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "react-scripts start" 2>/dev/null || true
    pkill -KILL -f "npm start" 2>/dev/null || true
    log_info "Dashboard stopped. Exiting."
    exit 0
}
trap cleanup_and_exit INT TERM HUP QUIT

# Start dashboard in development mode (required for API proxy)
log_info "Starting React dashboard in development mode on port 3000..."
log_info "API calls will be proxied to http://localhost:8000"
log_info "Press Ctrl+C to stop"

npm start