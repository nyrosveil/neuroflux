#!/bin/bash
# NeuroFlux Port Management Utility
# Handles port checking, killing, and cleanup for multi-service architecture

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Port Configuration - More Reasonable Ports
API_PORT=8000
WEBSOCKET_PORT=8001  # Separate WebSocket port
DASHBOARD_PORT=3000
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001

# Functions
log_info() { echo -e "${BLUE}[PORT-MGR]${NC} $1"; }
log_success() { echo -e "${GREEN}[PORT-MGR]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[PORT-MGR]${NC} $1"; }
log_error() { echo -e "${RED}[PORT-MGR]${NC} $1"; }

# Check if a port is in use
check_port_in_use() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Get process using a port
get_port_process() {
    local port=$1
    lsof -Pi :$port -sTCP:LISTEN -t 2>/dev/null || echo ""
}

# Kill process using a specific port
kill_port_process() {
    local port=$1
    local signal=${2:-TERM}
    local process_pid=$(get_port_process $port)

    if [ -n "$process_pid" ]; then
        log_info "Killing process $process_pid using port $port with $signal..."
        kill -$signal $process_pid 2>/dev/null || true

        # Wait up to 5 seconds for graceful shutdown
        local count=0
        while [ $count -lt 5 ] && check_port_in_use $port; do
            sleep 1
            count=$((count + 1))
        done

        if check_port_in_use $port; then
            log_warning "Process didn't stop gracefully, force killing..."
            kill -KILL $process_pid 2>/dev/null || true
            sleep 1
        fi

        if ! check_port_in_use $port; then
            log_success "Port $port freed successfully"
            return 0
        else
            log_error "Failed to free port $port"
            return 1
        fi
    else
        log_info "Port $port is already free"
        return 0
    fi
}

# Kill all NeuroFlux-related processes
kill_neuroflux_processes() {
    log_info "Killing all NeuroFlux processes..."

    # Kill by process names
    pkill -TERM -f "python dashboard_api.py" 2>/dev/null || true
    pkill -TERM -f "react-scripts start" 2>/dev/null || true
    pkill -TERM -f "npm start" 2>/dev/null || true
    pkill -TERM -f "gunicorn.*dashboard_api" 2>/dev/null || true

    # Wait a moment for graceful shutdown
    sleep 2

    # Force kill if still running
    pkill -KILL -f "python dashboard_api.py" 2>/dev/null || true
    pkill -KILL -f "react-scripts start" 2>/dev/null || true
    pkill -KILL -f "npm start" 2>/dev/null || true
    pkill -KILL -f "gunicorn.*dashboard_api" 2>/dev/null || true

    log_success "NeuroFlux processes killed"
}

# Clean up all NeuroFlux ports
cleanup_neuroflux_ports() {
    log_info "Cleaning up NeuroFlux ports..."

    local ports=($API_PORT $WEBSOCKET_PORT $DASHBOARD_PORT $PROMETHEUS_PORT $GRAFANA_PORT)

    for port in "${ports[@]}"; do
        if check_port_in_use $port; then
            log_warning "Port $port is in use, attempting to free..."
            kill_port_process $port
        else
            log_info "Port $port is free"
        fi
    done

    log_success "Port cleanup completed"
}

# Show port status
show_port_status() {
    log_info "NeuroFlux Port Status:"

    local ports=(
        "API:$API_PORT"
        "WebSocket:$WEBSOCKET_PORT"
        "Dashboard:$DASHBOARD_PORT"
        "Prometheus:$PROMETHEUS_PORT"
        "Grafana:$GRAFANA_PORT"
    )

    for port_info in "${ports[@]}"; do
        local name=$(echo $port_info | cut -d: -f1)
        local port=$(echo $port_info | cut -d: -f2)

        if check_port_in_use $port; then
            local process=$(get_port_process $port)
            echo -e "  $name (Port $port): ${RED}IN USE${NC} (PID: $process)"
        else
            echo -e "  $name (Port $port): ${GREEN}FREE${NC}"
        fi
    done
}

# Setup signal traps for cleanup
setup_cleanup_traps() {
    log_info "Setting up cleanup traps for signals..."

    # Cleanup function
    cleanup_on_exit() {
        log_warning "Received termination signal, cleaning up..."
        kill_neuroflux_processes
        cleanup_neuroflux_ports
        log_info "Cleanup completed. Exiting."
        exit 0
    }

    # Trap common termination signals
    trap cleanup_on_exit INT TERM HUP QUIT
}

# Main command handling
case "${1:-status}" in
    "status")
        show_port_status
        ;;
    "cleanup")
        cleanup_neuroflux_ports
        ;;
    "kill-all")
        kill_neuroflux_processes
        ;;
    "kill-port")
        if [ -z "$2" ]; then
            log_error "Usage: $0 kill-port <port>"
            exit 1
        fi
        kill_port_process "$2"
        ;;
    "check-port")
        if [ -z "$2" ]; then
            log_error "Usage: $0 check-port <port>"
            exit 1
        fi
        if check_port_in_use "$2"; then
            local process=$(get_port_process "$2")
            log_info "Port $2 is in use by process: $process"
            exit 0
        else
            log_info "Port $2 is free"
            exit 1
        fi
        ;;
    "setup-traps")
        setup_cleanup_traps
        log_info "Cleanup traps set. Press Ctrl+C to test."
        # Keep script running to test traps
        while true; do
            sleep 1
        done
        ;;
    "help"|"-h"|"--help")
        echo "NeuroFlux Port Management Utility"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  status              Show status of all NeuroFlux ports"
        echo "  cleanup             Free all NeuroFlux ports by killing processes"
        echo "  kill-all            Kill all NeuroFlux-related processes"
        echo "  kill-port <port>    Kill process using specific port"
        echo "  check-port <port>   Check if port is in use"
        echo "  setup-traps         Setup signal traps for cleanup (testing)"
        echo "  help                Show this help message"
        echo ""
        echo "Port Configuration:"
        echo "  API: $API_PORT"
        echo "  WebSocket: $WEBSOCKET_PORT"
        echo "  Dashboard: $DASHBOARD_PORT"
        echo "  Prometheus: $PROMETHEUS_PORT"
        echo "  Grafana: $GRAFANA_PORT"
        ;;
    *)
        log_error "Unknown command: $1"
        log_info "Run '$0 help' for usage information"
        exit 1
        ;;
esac