#!/bin/bash
# NeuroFlux Process Monitor
# Simple process management for hybrid deployment

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() { echo -e "${BLUE}[MONITOR]${NC} $1"; }
log_success() { echo -e "${GREEN}[MONITOR]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[MONITOR]${NC} $1"; }
log_error() { echo -e "${RED}[MONITOR]${NC} $1"; }

# Check if NeuroFlux process is running
check_process() {
    if pgrep -f "python dashboard_api.py" > /dev/null 2>&1; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

# Get process information
get_process_info() {
    local pid=$(pgrep -f "python dashboard_api.py" 2>/dev/null || echo "")
    if [ -n "$pid" ]; then
        echo "PID: $pid"
        echo "Command: $(ps -p $pid -o cmd= 2>/dev/null || echo 'N/A')"
        echo "CPU: $(ps -p $pid -o pcpu= 2>/dev/null || echo 'N/A')%"
        echo "Memory: $(ps -p $pid -o pmem= 2>/dev/null || echo 'N/A')%"
        echo "Started: $(ps -p $pid -o lstart= 2>/dev/null || echo 'N/A')"
    else
        echo "No NeuroFlux process found"
    fi
}

# Start NeuroFlux process
start_process() {
    log_info "Starting NeuroFlux..."

    if check_process; then
        log_warning "NeuroFlux is already running"
        return 1
    fi

    # Check if we're in the right directory
    if [ ! -f "dashboard_api.py" ]; then
        log_error "dashboard_api.py not found. Please run from NeuroFlux root directory."
        return 1
    fi

    # Start in background
    nohup bash start_hybrid.sh > neuroflux.log 2>&1 &
    local pid=$!

    # Wait for startup
    log_info "Waiting for NeuroFlux to start (PID: $pid)..."
    local count=0
    while [ $count -lt 30 ]; do
        if check_process; then
            log_success "NeuroFlux started successfully"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    log_error "NeuroFlux failed to start within 30 seconds"
    return 1
}

# Stop NeuroFlux process
stop_process() {
    log_info "Stopping NeuroFlux..."

    if ! check_process; then
        log_warning "NeuroFlux is not running"
        return 0
    fi

    # Try graceful shutdown first
    pkill -TERM -f "python dashboard_api.py" 2>/dev/null || true

    # Wait for graceful shutdown
    local count=0
    while [ $count -lt 10 ]; do
        if ! check_process; then
            log_success "NeuroFlux stopped gracefully"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    # Force kill if still running
    log_warning "Graceful shutdown failed, force killing..."
    pkill -KILL -f "python dashboard_api.py" 2>/dev/null || true

    sleep 2
    if ! check_process; then
        log_success "NeuroFlux force killed"
        return 0
    else
        log_error "Failed to stop NeuroFlux"
        return 1
    fi
}

# Restart NeuroFlux process
restart_process() {
    log_info "Restarting NeuroFlux..."
    stop_process
    sleep 2
    start_process
}

# Show status
show_status() {
    echo "NeuroFlux Status"
    echo "================"

    if check_process; then
        log_success "Status: Running"
        echo ""
        get_process_info
    else
        log_error "Status: Not Running"
        echo ""
        echo "To start: $0 start"
    fi

    echo ""
    echo "Environment Info:"
    bash env_manager.sh info 2>/dev/null | head -10

    echo ""
    echo "Recent Logs:"
    if [ -f "neuroflux.log" ]; then
        tail -5 neuroflux.log 2>/dev/null || echo "No recent logs"
    else
        echo "No log file found"
    fi
}

# Health check
health_check() {
    log_info "Running health check..."

    if ! check_process; then
        echo "❌ Service not running"
        return 1
    fi

    # Try to connect to health endpoint
    if command -v curl &> /dev/null; then
        if curl -f -s --max-time 5 http://localhost:5001/api/health > /dev/null 2>&1; then
            echo "✅ Health endpoint responding"
            return 0
        else
            echo "❌ Health endpoint not responding"
            return 1
        fi
    else
        echo "⚠️ curl not available, skipping health endpoint check"
        echo "✅ Process is running"
        return 0
    fi
}

# Main command handling
case "$1" in
    start)
        start_process
        ;;
    stop)
        stop_process
        ;;
    restart)
        restart_process
        ;;
    status)
        show_status
        ;;
    info)
        get_process_info
        ;;
    health)
        health_check
        ;;
    *)
        echo "NeuroFlux Process Monitor"
        echo "========================"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|info|health}"
        echo ""
        echo "Commands:"
        echo "  start   - Start NeuroFlux server"
        echo "  stop    - Stop NeuroFlux server"
        echo "  restart - Restart NeuroFlux server"
        echo "  status  - Show detailed status"
        echo "  info    - Show process information"
        echo "  health  - Run health check"
        echo ""
        echo "Examples:"
        echo "  $0 start    # Start the server"
        echo "  $0 status   # Check if running"
        echo "  $0 restart  # Restart the server"
        exit 1
        ;;
esac