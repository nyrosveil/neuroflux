#!/bin/bash
# NeuroFlux SSL Certificate Setup Script
# This script sets up SSL certificates using Let's Encrypt

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (sudo)"
    exit 1
fi

# Get domain from nginx config or ask user
get_domain() {
    # Try to extract domain from nginx.conf
    if [ -f "/etc/nginx/sites-available/neuroflux" ]; then
        DOMAIN=$(grep "server_name" /etc/nginx/sites-available/neuroflux | head -1 | awk '{print $2}' | sed 's/;//')
        if [[ $DOMAIN == "your-domain.com" ]] || [[ $DOMAIN == "_" ]]; then
            DOMAIN=""
        fi
    fi

    if [ -z "$DOMAIN" ]; then
        echo "Please enter your domain name (e.g., neuroflux.example.com):"
        read -r DOMAIN
        if [ -z "$DOMAIN" ]; then
            log_error "Domain name is required"
            exit 1
        fi
    fi

    echo "$DOMAIN"
}

# Install certbot if not present
install_certbot() {
    log_info "Installing certbot..."

    # Update package list
    apt-get update

    # Install certbot and nginx plugin
    apt-get install -y certbot python3-certbot-nginx

    log_success "Certbot installed"
}

# Configure nginx for SSL setup
configure_nginx_for_ssl() {
    local domain=$1

    log_info "Configuring nginx for SSL setup..."

    # Backup original config
    cp /etc/nginx/sites-available/neuroflux /etc/nginx/sites-available/neuroflux.backup

    # Update nginx config with domain
    sed -i "s/your-domain\.com/$domain/g" /etc/nginx/sites-available/neuroflux

    # Test nginx configuration
    if nginx -t; then
        systemctl reload nginx
        log_success "Nginx configured for domain $domain"
    else
        log_error "Nginx configuration test failed"
        # Restore backup
        cp /etc/nginx/sites-available/neuroflux.backup /etc/nginx/sites-available/neuroflux
        exit 1
    fi
}

# Obtain SSL certificate
obtain_certificate() {
    local domain=$1

    log_info "Obtaining SSL certificate for $domain..."

    # Run certbot
    if certbot --nginx -d "$domain" --non-interactive --agree-tos --email "admin@$domain"; then
        log_success "SSL certificate obtained successfully"

        # Verify certificate
        if [ -f "/etc/letsencrypt/live/$domain/fullchain.pem" ]; then
            log_success "Certificate files created"
        else
            log_error "Certificate files not found"
            exit 1
        fi
    else
        log_error "Failed to obtain SSL certificate"
        exit 1
    fi
}

# Setup automatic renewal
setup_renewal() {
    log_info "Setting up automatic certificate renewal..."

    # Add renewal cron job if not exists
    CRON_JOB="0 12 * * * /usr/bin/certbot renew --quiet"
    if ! crontab -l | grep -q "certbot renew"; then
        (crontab -l ; echo "$CRON_JOB") | crontab -
        log_success "Certificate renewal cron job added"
    else
        log_success "Certificate renewal cron job already exists"
    fi

    # Test renewal
    certbot renew --dry-run
    log_success "Certificate renewal configured"
}

# Update nginx config with SSL paths
update_nginx_ssl_paths() {
    local domain=$1

    log_info "Updating nginx configuration with SSL certificate paths..."

    # The nginx.conf should already have the correct paths after certbot runs
    # But let's verify
    if grep -q "/etc/letsencrypt/live/$domain/fullchain.pem" /etc/nginx/sites-available/neuroflux; then
        log_success "SSL paths configured in nginx"
    else
        log_warning "SSL paths may not be properly configured"
    fi
}

# Test SSL setup
test_ssl() {
    local domain=$1

    log_info "Testing SSL setup..."

    # Test HTTPS connection
    if curl -f -s --max-time 10 "https://$domain/health" > /dev/null; then
        log_success "HTTPS connection successful"
    else
        log_warning "HTTPS connection test failed - this may be normal if nginx isn't reloaded yet"
    fi

    # Check certificate validity
    if openssl s_client -connect "$domain:443" -servername "$domain" < /dev/null 2>/dev/null | openssl x509 -noout -dates > /dev/null 2>&1; then
        log_success "SSL certificate is valid"
    else
        log_error "SSL certificate validation failed"
        exit 1
    fi
}

# Main function
main() {
    log_info "NeuroFlux SSL Certificate Setup"
    log_info "================================"

    # Get domain
    DOMAIN=$(get_domain)
    log_info "Setting up SSL for domain: $DOMAIN"

    # Install certbot
    install_certbot

    # Configure nginx
    configure_nginx_for_ssl "$DOMAIN"

    # Obtain certificate
    obtain_certificate "$DOMAIN"

    # Setup renewal
    setup_renewal

    # Update nginx SSL paths
    update_nginx_ssl_paths "$DOMAIN"

    # Test SSL
    test_ssl "$DOMAIN"

    log_success "🎉 SSL setup completed successfully!"
    echo ""
    echo "SSL Certificate Summary:"
    echo "- Domain: $DOMAIN"
    echo "- Certificate: /etc/letsencrypt/live/$DOMAIN/fullchain.pem"
    echo "- Private Key: /etc/letsencrypt/live/$DOMAIN/privkey.pem"
    echo "- Auto-renewal: Configured (runs daily at 12:00)"
    echo ""
    echo "Next steps:"
    echo "1. Update your DNS to point to this server"
    echo "2. Test HTTPS access: https://$DOMAIN"
    echo "3. Consider setting up HTTP Strict Transport Security (HSTS)"
}

# Run main function
main "$@"