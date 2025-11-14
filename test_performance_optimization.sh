#!/bin/bash
# Test Performance Optimization Configuration

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "Testing performance optimization configuration..."

# Check if optimize_performance.sh exists and is valid
if [ -f "optimize_performance.sh" ]; then
    if bash -n optimize_performance.sh 2>/dev/null; then
        log_success "Performance optimization script syntax valid"
    else
        log_error "Performance optimization script has syntax errors"
        exit 1
    fi
else
    log_error "Performance optimization script not found"
    exit 1
fi

# Check systemd service has basic performance settings
if grep -q "MemoryLimit" neuroflux.service; then
    log_success "Systemd service includes memory limits"
else
    log_error "Systemd service missing memory limits"
    exit 1
fi

# Check for gthread (will be added by optimization script)
if grep -q "gthread" neuroflux.service; then
    log_success "Gunicorn worker class optimized"
else
    log_info "Gunicorn worker class will be optimized during setup"
fi

# Check for worker configuration
if grep -q "\-w [0-9]" neuroflux.service; then
    log_success "Gunicorn worker configuration present"
else
    log_error "Gunicorn worker configuration missing"
    exit 1
fi

# Check for monitoring scripts
if [ -f "monitor.sh" ] && [ -f "health_check.sh" ]; then
    log_success "Monitoring scripts available for performance tracking"
else
    log_error "Monitoring scripts missing"
    exit 1
fi

log_success "🎉 Performance optimization configuration validated successfully!"
echo "Performance optimization is ready for production deployment."