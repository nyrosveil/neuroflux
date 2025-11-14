#!/bin/bash
# NeuroFlux Health Check Script
# This script performs comprehensive health checks on the NeuroFlux system

# Configuration
API_URL="http://localhost:5001"
SERVICE_NAME="neuroflux"
LOG_FILE="/var/log/neuroflux/health_check.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    echo -e "$1"
}

check_service() {
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log "${GREEN}✓ Service $SERVICE_NAME is running${NC}"
        return 0
    else
        log "${RED}✗ Service $SERVICE_NAME is not running${NC}"
        return 1
    fi
}

check_api() {
    local endpoint=$1
    local expected_status=${2:-200}

    if curl -f -s --max-time 10 "$API_URL$endpoint" > /dev/null 2>&1; then
        log "${GREEN}✓ API endpoint $endpoint is responding${NC}"
        return 0
    else
        log "${RED}✗ API endpoint $endpoint is not responding${NC}"
        return 1
    fi
}

check_disk_space() {
    local threshold=${1:-90}
    local usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

    if [ "$usage" -lt "$threshold" ]; then
        log "${GREEN}✓ Disk usage is $usage% (below ${threshold}%)${NC}"
        return 0
    else
        log "${RED}✗ Disk usage is $usage% (above ${threshold}%)${NC}"
        return 1
    fi
}

check_memory() {
    local threshold=${1:-90}
    local usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')

    if [ "$usage" -lt "$threshold" ]; then
        log "${GREEN}✓ Memory usage is $usage% (below ${threshold}%)${NC}"
        return 0
    else
        log "${RED}✗ Memory usage is $usage% (above ${threshold}%)${NC}"
        return 1
    fi
}

check_processes() {
    local process_count=$(pgrep -f "neuroflux\|gunicorn" | wc -l)

    if [ "$process_count" -ge 4 ]; then  # At least gunicorn master + 3 workers
        log "${GREEN}✓ $process_count NeuroFlux processes running${NC}"
        return 0
    else
        log "${RED}✗ Only $process_count NeuroFlux processes running${NC}"
        return 1
    fi
}

check_logs() {
    local log_age_hours=${1:-24}
    local recent_errors=$(find /var/log/neuroflux -name "*.log" -mtime -1 -exec grep -l "ERROR\|CRITICAL" {} \; 2>/dev/null | wc -l)

    if [ "$recent_errors" -eq 0 ]; then
        log "${GREEN}✓ No recent errors in logs${NC}"
        return 0
    else
        log "${YELLOW}⚠ $recent_errors log files contain recent errors${NC}"
        return 1
    fi
}

send_alert() {
    local message=$1
    local severity=${2:-warning}

    # Here you could integrate with monitoring systems like:
    # - Send email
    # - Send Slack notification
    # - Send to monitoring dashboard
    # - Trigger PagerDuty alert

    log "${RED}ALERT [$severity]: $message${NC}"

    # Example: Send email alert (requires mailutils)
    # echo "$message" | mail -s "NeuroFlux Alert [$severity]" admin@example.com
}

main() {
    log "=== NeuroFlux Health Check Started ==="

    local failed_checks=0
    local total_checks=0

    # Service check
    ((total_checks++))
    if ! check_service; then
        ((failed_checks++))
        send_alert "NeuroFlux service is not running" "critical"
    fi

    # API checks
    ((total_checks++))
    if ! check_api "/api/status"; then
        ((failed_checks++))
        send_alert "API status endpoint not responding" "critical"
    fi

    ((total_checks++))
    if ! check_api "/api/dashboard/predictions"; then
        ((failed_checks++))
        send_alert "API predictions endpoint not responding" "warning"
    fi

    # System resource checks
    ((total_checks++))
    if ! check_disk_space 90; then
        ((failed_checks++))
        send_alert "Disk space usage is high" "warning"
    fi

    ((total_checks++))
    if ! check_memory 90; then
        ((failed_checks++))
        send_alert "Memory usage is high" "warning"
    fi

    # Process check
    ((total_checks++))
    if ! check_processes; then
        ((failed_checks++))
        send_alert "Insufficient NeuroFlux processes running" "critical"
    fi

    # Log check
    ((total_checks++))
    if ! check_logs; then
        ((failed_checks++))
        send_alert "Recent errors found in logs" "warning"
    fi

    # Summary
    local success_rate=$(( (total_checks - failed_checks) * 100 / total_checks ))

    if [ $failed_checks -eq 0 ]; then
        log "${GREEN}🎉 All health checks passed!${NC}"
        exit 0
    elif [ $success_rate -ge 75 ]; then
        log "${YELLOW}⚠️ Health check completed with $failed_checks failed checks ($success_rate% success rate)${NC}"
        exit 1
    else
        log "${RED}🚨 Critical health issues detected! $failed_checks failed checks ($success_rate% success rate)${NC}"
        exit 2
    fi
}

# Run health check
main "$@"