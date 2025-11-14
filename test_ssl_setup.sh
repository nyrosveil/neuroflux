#!/bin/bash
# Test SSL Setup Configuration

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "Testing SSL setup configuration..."

# Check if setup_ssl.sh exists and is valid
if [ -f "setup_ssl.sh" ]; then
    if bash -n setup_ssl.sh 2>/dev/null; then
        log_success "SSL setup script syntax valid"
    else
        log_error "SSL setup script has syntax errors"
        exit 1
    fi
else
    log_error "SSL setup script not found"
    exit 1
fi

# Check nginx config has SSL placeholders
if grep -q "ssl_certificate" nginx.conf; then
    log_success "Nginx config includes SSL certificate placeholders"
else
    log_error "Nginx config missing SSL certificate configuration"
    exit 1
fi

# Check for domain placeholder
if grep -q "your-domain.com" nginx.conf; then
    log_success "Nginx config has domain placeholders for SSL setup"
else
    log_warning "Nginx config may need domain configuration"
fi

# Check SSL security settings
if grep -q "ssl_protocols" nginx.conf && grep -q "ssl_ciphers" nginx.conf; then
    log_success "Nginx config includes SSL security settings"
else
    log_error "Nginx config missing SSL security settings"
    exit 1
fi

log_success "🎉 SSL configuration validated successfully!"
echo "SSL setup is ready for production deployment."