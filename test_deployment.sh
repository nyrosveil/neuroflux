#!/bin/bash
# NeuroFlux Production Deployment Test Script

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "Testing NeuroFlux production deployment configuration..."

# Test Python
if command -v python3 &> /dev/null; then
    log_success "Python3 found"
else
    log_error "Python3 not found"
    exit 1
fi

# Test virtual environment
if python3 -m venv test_venv 2>/dev/null; then
    log_success "Virtual environment creation works"
else
    log_error "Virtual environment creation failed"
    exit 1
fi

# Test dependencies
if source test_venv/bin/activate && pip install --quiet -r requirements.txt; then
    log_success "Dependencies install successfully"
    deactivate
else
    log_error "Dependency installation failed"
    deactivate
    exit 1
fi

# Test service file
if [ -f "neuroflux.service" ] && grep -q "\[Unit\]" neuroflux.service; then
    log_success "Service file syntax valid"
else
    log_error "Service file invalid"
    exit 1
fi

# Test nginx config
if [ -f "nginx.conf" ] && grep -q "server {" nginx.conf; then
    log_success "Nginx config syntax valid"
else
    log_error "Nginx config invalid"
    exit 1
fi

# Test scripts
for script in health_check.sh monitor.sh; do
    if [ -f "$script" ] && bash -n "$script" 2>/dev/null; then
        log_success "$script syntax valid"
    else
        log_error "$script has syntax errors"
        exit 1
    fi
done

# Test API import
if source test_venv/bin/activate && python3 -c "import sys; sys.path.insert(0, 'src'); import dashboard_api" 2>/dev/null; then
    log_success "API module imports successfully"
    deactivate
else
    log_error "API module import failed"
    deactivate
    exit 1
fi

# Cleanup
rm -rf test_venv

log_success "🎉 All deployment tests passed!"
echo "Production deployment configuration is ready!"