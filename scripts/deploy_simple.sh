#!/bin/bash
# NeuroFlux Simple Deployment Script (No Docker)
# Manages API and Dashboard services with proper port handling

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() { echo -e "${BLUE}[DEPLOY]${NC} $1"; }
log_success() { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[DEPLOY]${NC} $1"; }
log_error() { echo -e "${RED}[DEPLOY]${NC} $1"; }

echo "🚀 NeuroFlux Simple Deployment (No Docker)"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "dashboard_api.py" ]; then
    log_error "dashboard_api.py not found. Please run from NeuroFlux root directory."
    exit 1
fi

# Port cleanup before starting
log_info "Performing port cleanup..."
if [ -f "port_manager.sh" ]; then
    bash port_manager.sh cleanup
else
    log_warning "port_manager.sh not found, skipping port cleanup"
fi

# Start API server in background
log_info "Starting API server on port 8000..."
./start_api.sh &
API_PID=$!

# Wait for API to start
sleep 5

# Check if API is running
if curl -f -s --max-time 5 http://localhost:8000/api/health > /dev/null 2>&1; then
    log_success "API server started successfully"
else
    log_error "API server failed to start"
    kill $API_PID 2>/dev/null || true
    exit 1
fi

# Start Dashboard in background
log_info "Starting React dashboard on port 3000..."
./start_dashboard.sh &
DASHBOARD_PID=$!

# Wait for dashboard to start
sleep 10

# Check if dashboard is running
if curl -f -s --max-time 5 http://localhost:3000 > /dev/null 2>&1; then
    log_success "Dashboard started successfully"
else
    log_warning "Dashboard may still be compiling... (this can take 1-2 minutes)"
fi

# Save PIDs for cleanup
echo $API_PID > .api_pid
echo $DASHBOARD_PID > .dashboard_pid

log_success "NeuroFlux deployment completed!"
echo ""
echo "🌐 Access URLs:"
echo "  Dashboard: http://localhost:3000"
echo "  API:        http://localhost:8000"
echo ""
echo "🛑 To stop: ./scripts/stop.sh"
echo "📊 View logs: tail -f logs/*.log"

# Keep script running to maintain services
log_info "Services are running. Press Ctrl+C to stop..."

# Cleanup function
cleanup() {
    log_info "Stopping services..."
    if [ -f ".api_pid" ]; then
        kill $(cat .api_pid) 2>/dev/null || true
        rm .api_pid
    fi
    if [ -f ".dashboard_pid" ]; then
        kill $(cat .dashboard_pid) 2>/dev/null || true
        rm .dashboard_pid
    fi
    log_success "Services stopped"
}

# Set trap for cleanup
trap cleanup INT TERM HUP QUIT

# Wait for services
wait