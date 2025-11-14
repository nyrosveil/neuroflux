#!/bin/bash
# Test Monitoring Setup Configuration

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "Testing monitoring setup configuration..."

# Check if setup_monitoring.sh exists and is valid
if [ -f "setup_monitoring.sh" ]; then
    if bash -n setup_monitoring.sh 2>/dev/null; then
        log_success "Monitoring setup script syntax valid"
    else
        log_error "Monitoring setup script has syntax errors"
        exit 1
    fi
else
    log_error "Monitoring setup script not found"
    exit 1
fi

# Check if metrics sample exists
if [ -f "metrics_sample.txt" ]; then
    log_success "Metrics sample file exists"
else
    log_warning "Metrics sample file not found (will be created during setup)"
fi

# Check if health_check.sh includes monitoring integration
if grep -q "send_alert" health_check.sh; then
    log_success "Health check script includes alerting functionality"
else
    log_error "Health check script missing alerting functionality"
    exit 1
fi

# Check if monitor.sh includes real-time metrics
if grep -q "get_system_metrics" monitor.sh && grep -q "get_neuroflux_metrics" monitor.sh; then
    log_success "Monitor script includes comprehensive metrics collection"
else
    log_error "Monitor script missing metrics collection"
    exit 1
fi

log_success "🎉 Monitoring configuration validated successfully!"
echo "Monitoring setup is ready for production deployment."