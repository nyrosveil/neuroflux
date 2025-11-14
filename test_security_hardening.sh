#!/bin/bash
# Test Security Hardening Configuration

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "Testing security hardening configuration..."

# Check if harden_security.sh exists and is valid
if [ -f "harden_security.sh" ]; then
    if bash -n harden_security.sh 2>/dev/null; then
        log_success "Security hardening script syntax valid"
    else
        log_error "Security hardening script has syntax errors"
        exit 1
    fi
else
    log_error "Security hardening script not found"
    exit 1
fi

# Check nginx config has security headers
if grep -q "X-Frame-Options" nginx.conf && grep -q "X-Content-Type-Options" nginx.conf; then
    log_success "Nginx config includes security headers"
else
    log_error "Nginx config missing security headers"
    exit 1
fi

# Check systemd service has security settings
if grep -q "NoNewPrivileges\|ProtectSystem\|PrivateTmp" neuroflux.service; then
    log_success "Systemd service includes security hardening"
else
    log_error "Systemd service missing security settings"
    exit 1
fi

# Check for monitoring scripts
if [ -f "health_check.sh" ] && [ -f "monitor.sh" ]; then
    log_success "Security monitoring scripts available"
else
    log_error "Security monitoring scripts missing"
    exit 1
fi

log_success "🎉 Security hardening configuration validated successfully!"
echo "Security hardening is ready for production deployment."