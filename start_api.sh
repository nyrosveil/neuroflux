#!/bin/bash
# NeuroFlux API Server Startup Script
# Starts the Flask API server on port 8000

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() { echo -e "${BLUE}[API]${NC} $1"; }
log_success() { echo -e "${GREEN}[API]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[API]${NC} $1"; }
log_error() { echo -e "${RED}[API]${NC} $1"; }

echo "🚀 NeuroFlux API Server (Port 8000)"
echo "==================================="

# Check if we're in the right directory
if [ ! -f "dashboard_api.py" ]; then
    log_error "dashboard_api.py not found. Please run from NeuroFlux root directory."
    exit 1
fi

# Port cleanup before starting
log_info "Performing port cleanup before startup..."
if [ -f "port_manager.sh" ]; then
    bash port_manager.sh cleanup
else
    log_warning "port_manager.sh not found, skipping port cleanup"
fi

# Environment setup
ENV_TYPE=$(bash env_manager.sh detect)

case $ENV_TYPE in
    "conda")
        log_info "Using conda environment"
        ;;
    "venv")
        log_info "Using virtual environment"
        ;;
    "none")
        log_warning "No environment detected - using system Python"
        ;;
esac

# Trap for graceful shutdown with port cleanup
cleanup_and_exit() {
    log_warning "Received shutdown signal, performing cleanup..."
    if [ -f "port_manager.sh" ]; then
        bash port_manager.sh kill-all
        bash port_manager.sh cleanup
    fi
    log_info "Cleanup completed. Exiting."
    exit 0
}
trap cleanup_and_exit INT TERM HUP QUIT

# Start server
log_info "Starting NeuroFlux API server on port 8000..."
log_info "Press Ctrl+C to stop"

if python dashboard_api.py; then
    log_success "API server exited normally"
else
    log_error "API server crashed"
    exit 1
fi