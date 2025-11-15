#!/bin/bash
# NeuroFlux Stop Script
# Stops all running NeuroFlux services

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() { echo -e "${BLUE}[STOP]${NC} $1"; }
log_success() { echo -e "${GREEN}[STOP]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[STOP]${NC} $1"; }
log_error() { echo -e "${RED}[STOP]${NC} $1"; }

echo "🛑 Stopping NeuroFlux Services"
echo "============================="

# Stop by PID files
stopped_any=false

if [ -f ".api_pid" ]; then
    API_PID=$(cat .api_pid)
    log_info "Stopping API server (PID: $API_PID)..."
    kill $API_PID 2>/dev/null || true
    rm .api_pid
    stopped_any=true
fi

if [ -f ".dashboard_pid" ]; then
    DASHBOARD_PID=$(cat .dashboard_pid)
    log_info "Stopping dashboard (PID: $DASHBOARD_PID)..."
    kill $DASHBOARD_PID 2>/dev/null || true
    rm .dashboard_pid
    stopped_any=true
fi

# Fallback: Use port manager to kill any remaining processes
if [ -f "port_manager.sh" ]; then
    log_info "Performing port cleanup..."
    bash port_manager.sh kill-all
    bash port_manager.sh cleanup
fi

# Final cleanup
if [ "$stopped_any" = true ]; then
    log_success "NeuroFlux services stopped successfully"
else
    log_warning "No PID files found - services may not be running"
fi

echo ""
echo "✅ All services stopped"