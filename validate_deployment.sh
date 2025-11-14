#!/bin/bash
# Quick NeuroFlux Production Deployment Validation

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "Validating NeuroFlux production deployment files..."

# Check required files exist
files=("requirements.txt" "neuroflux.service" "nginx.conf" "health_check.sh" "monitor.sh" "deploy_production.sh")

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        log_success "$file exists"
    else
        log_error "$file missing"
        exit 1
    fi
done

# Validate service file structure
if grep -q "\[Unit\]" neuroflux.service && grep -q "\[Service\]" neuroflux.service && grep -q "\[Install\]" neuroflux.service; then
    log_success "Service file structure valid"
else
    log_error "Service file structure invalid"
    exit 1
fi

# Validate nginx config structure
if grep -q "server {" nginx.conf && grep -q "location /api/" nginx.conf; then
    log_success "Nginx config structure valid"
else
    log_error "Nginx config structure invalid"
    exit 1
fi

# Validate script syntax
for script in health_check.sh monitor.sh deploy_production.sh; do
    if bash -n "$script" 2>/dev/null; then
        log_success "$script syntax valid"
    else
        log_error "$script syntax invalid"
        exit 1
    fi
done

# Check requirements.txt has production dependencies
if grep -q "gunicorn" requirements.txt && grep -q "flask" requirements.txt; then
    log_success "Requirements.txt includes production dependencies"
else
    log_error "Requirements.txt missing production dependencies"
    exit 1
fi

log_success "🎉 All production deployment files validated successfully!"
echo "Ready for production deployment testing."