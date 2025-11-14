#!/bin/bash
# NeuroFlux Monitoring Script
# This script provides real-time monitoring of the NeuroFlux system

# Configuration
API_URL="http://localhost:5001"
SERVICE_NAME="neuroflux"
LOG_FILE="/var/log/neuroflux/monitor.log"
INTERVAL=${1:-60}  # Default monitoring interval in seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

print_status() {
    echo -e "$(date '+%H:%M:%S') - $1"
}

get_system_metrics() {
    # CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')

    # Memory usage
    MEM_TOTAL=$(free -m | grep '^Mem:' | awk '{print $2}')
    MEM_USED=$(free -m | grep '^Mem:' | awk '{print $3}')
    MEM_USAGE=$(( MEM_USED * 100 / MEM_TOTAL ))

    # Disk usage
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

    # Network connections
    NET_CONNECTIONS=$(netstat -tun | grep ESTABLISHED | wc -l)

    echo "$CPU_USAGE:$MEM_USAGE:$DISK_USAGE:$NET_CONNECTIONS"
}

get_neuroflux_metrics() {
    # Get API status
    if curl -f -s --max-time 5 "$API_URL/api/status" > /dev/null 2>&1; then
        API_STATUS="UP"

        # Get detailed metrics
        STATUS_JSON=$(curl -s "$API_URL/api/status" 2>/dev/null)
        if echo "$STATUS_JSON" | jq -e '.initialized_agents' > /dev/null 2>&1; then
            AGENTS=$(echo "$STATUS_JSON" | jq -r '.initialized_agents')
            TOTAL_AGENTS=$(echo "$STATUS_JSON" | jq -r '.total_agents // 12')
        else
            AGENTS="N/A"
            TOTAL_AGENTS="12"
        fi
    else
        API_STATUS="DOWN"
        AGENTS="0"
        TOTAL_AGENTS="12"
    fi

    # Process count
    PROCESS_COUNT=$(pgrep -f "neuroflux\|gunicorn" | wc -l)

    echo "$API_STATUS:$AGENTS:$TOTAL_AGENTS:$PROCESS_COUNT"
}

display_header() {
    echo "=================================================================================="
    echo " NeuroFlux Production Monitoring Dashboard"
    echo "=================================================================================="
    printf "%-12s %-8s %-8s %-8s %-12s %-8s %-8s %-8s %-8s\n" \
           "TIME" "CPU%" "MEM%" "DISK%" "NET_CONN" "API" "AGENTS" "PROCS" "STATUS"
    echo "----------------------------------------------------------------------------------"
}

display_metrics() {
    local timestamp=$1
    local cpu=$2
    local mem=$3
    local disk=$4
    local net=$5
    local api_status=$6
    local agents=$7
    local processes=$8

    # Determine overall status
    local status="OK"
    local status_color=$GREEN

    if [ "$api_status" = "DOWN" ]; then
        status="CRIT"
        status_color=$RED
    elif [ "$cpu" -gt 90 ] || [ "$mem" -gt 90 ] || [ "$disk" -gt 90 ]; then
        status="WARN"
        status_color=$YELLOW
    fi

    printf "%-12s %-8s %-8s %-8s %-12s %-8s %-8s %-8s %s\n" \
           "$timestamp" "${cpu}%" "${mem}%" "${disk}%" "$net" \
           "$api_status" "$agents" "$processes" "${status_color}${status}${NC}"
}

monitor_loop() {
    display_header

    while true; do
        # Get system metrics
        SYS_METRICS=$(get_system_metrics)
        CPU=$(echo "$SYS_METRICS" | cut -d: -f1)
        MEM=$(echo "$SYS_METRICS" | cut -d: -f2)
        DISK=$(echo "$SYS_METRICS" | cut -d: -f3)
        NET=$(echo "$SYS_METRICS" | cut -d: -f4)

        # Get NeuroFlux metrics
        NF_METRICS=$(get_neuroflux_metrics)
        API_STATUS=$(echo "$NF_METRICS" | cut -d: -f1)
        AGENTS=$(echo "$NF_METRICS" | cut -d: -f2)
        PROCESSES=$(echo "$NF_METRICS" | cut -d: -f3)

        # Display metrics
        TIMESTAMP=$(date '+%H:%M:%S')
        display_metrics "$TIMESTAMP" "$CPU" "$MEM" "$DISK" "$NET" "$API_STATUS" "$AGENTS" "$PROCESSES"

        # Log metrics
        log "METRICS: CPU=${CPU}% MEM=${MEM}% DISK=${DISK}% NET=${NET} API=${API_STATUS} AGENTS=${AGENTS} PROCESSES=${PROCESSES}"

        # Alert on critical conditions
        if [ "$API_STATUS" = "DOWN" ]; then
            print_status "${RED}🚨 ALERT: NeuroFlux API is DOWN!${NC}"
            # Could send notification here
        elif [ "$CPU" -gt 95 ] || [ "$MEM" -gt 95 ]; then
            print_status "${YELLOW}⚠️  WARNING: High system resource usage${NC}"
        fi

        sleep "$INTERVAL"
    done
}

show_help() {
    echo "NeuroFlux Monitoring Script"
    echo "Usage: $0 [interval_seconds]"
    echo ""
    echo "Arguments:"
    echo "  interval_seconds  Monitoring interval in seconds (default: 60)"
    echo ""
    echo "Examples:"
    echo "  $0              # Monitor every 60 seconds"
    echo "  $0 30          # Monitor every 30 seconds"
    echo "  $0 300         # Monitor every 5 minutes"
    echo ""
    echo "Metrics displayed:"
    echo "  CPU%    - CPU usage percentage"
    echo "  MEM%    - Memory usage percentage"
    echo "  DISK%   - Disk usage percentage"
    echo "  NET_CONN- Number of network connections"
    echo "  API     - API status (UP/DOWN)"
    echo "  AGENTS  - Number of active agents"
    echo "  PROCS   - Number of running processes"
    echo "  STATUS  - Overall system status"
}

main() {
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            INTERVAL=${1:-60}
            if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [ "$INTERVAL" -lt 5 ]; then
                echo "Error: Interval must be a number >= 5 seconds"
                exit 1
            fi
            log "Starting NeuroFlux monitoring (interval: ${INTERVAL}s)"
            monitor_loop
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${BLUE}Monitoring stopped${NC}"; exit 0' INT

# Run main function
main "$@"