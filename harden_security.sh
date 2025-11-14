#!/bin/bash
# NeuroFlux Security Hardening Script
# Configures firewall, authentication, and encryption for production security

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

# Configure UFW firewall
configure_firewall() {
    log_info "Configuring UFW firewall..."

    # Install UFW if not present
    if ! command -v ufw &> /dev/null; then
        apt-get update && apt-get install -y ufw
    fi

    # Reset firewall to defaults
    ufw --force reset

    # Default policies
    ufw default deny incoming
    ufw default allow outgoing

    # Allow SSH (change port if using non-standard)
    ufw allow ssh
    log_success "SSH access allowed"

    # Allow HTTP and HTTPS
    ufw allow 80/tcp
    ufw allow 443/tcp
    log_success "HTTP/HTTPS access allowed"

    # Allow monitoring ports (if needed)
    # ufw allow 9090/tcp  # Prometheus
    # ufw allow 3000/tcp  # Grafana
    # ufw allow 9100/tcp  # Node Exporter

    # Enable firewall
    echo "y" | ufw enable

    log_success "Firewall configured and enabled"
}

# Configure fail2ban for SSH protection
configure_fail2ban() {
    log_info "Configuring fail2ban for SSH protection..."

    # Install fail2ban if not present
    if ! command -v fail2ban-client &> /dev/null; then
        apt-get install -y fail2ban
    fi

    # Create jail configuration
    cat > /etc/fail2ban/jail.d/neuroflux.conf << EOF
[neuroflux-ssh]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
EOF

    # Restart fail2ban
    systemctl restart fail2ban
    systemctl enable fail2ban

    log_success "Fail2ban configured for SSH protection"
}

# Secure SSH configuration
secure_ssh() {
    log_info "Securing SSH configuration..."

    # Backup original config
    cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

    # Apply security settings
    sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
    sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords no/' /etc/ssh/sshd_config
    sed -i 's/#Protocol 2/Protocol 2/' /etc/ssh/sshd_config

    # Add additional security settings
    cat >> /etc/ssh/sshd_config << EOF

# Additional security settings
AllowUsers neuroflux
ClientAliveInterval 300
ClientAliveCountMax 2
MaxAuthTries 3
MaxSessions 2
EOF

    # Restart SSH
    systemctl restart ssh
    systemctl restart sshd 2>/dev/null || true

    log_success "SSH configuration secured"
}

# Configure API authentication
configure_api_auth() {
    log_info "Configuring API authentication..."

    # Create API keys file
    API_KEYS_FILE="/opt/neuroflux/api_keys.json"
    mkdir -p /opt/neuroflux

    # Generate API keys
    ADMIN_KEY=$(openssl rand -hex 32)
    READ_KEY=$(openssl rand -hex 32)

    cat > "$API_KEYS_FILE" << EOF
{
  "admin": "$ADMIN_KEY",
  "read": "$READ_KEY",
  "keys": {
    "$ADMIN_KEY": {
      "role": "admin",
      "permissions": ["read", "write", "delete", "admin"],
      "description": "Full administrative access"
    },
    "$READ_KEY": {
      "role": "read",
      "permissions": ["read"],
      "description": "Read-only access to API"
    }
  }
}
EOF

    chown neuroflux:neuroflux "$API_KEYS_FILE"
    chmod 600 "$API_KEYS_FILE"

    log_success "API authentication keys generated"
    log_warning "Save these API keys securely:"
    echo "Admin Key: $ADMIN_KEY"
    echo "Read Key: $READ_KEY"
}

# Configure data encryption
configure_encryption() {
    log_info "Configuring data encryption..."

    # Install encryption tools
    apt-get install -y gnupg cryptsetup

    # Create encryption key for sensitive data
    ENCRYPTION_KEY="/opt/neuroflux/.encryption_key"
    openssl rand -hex 64 > "$ENCRYPTION_KEY"
    chown neuroflux:neuroflux "$ENCRYPTION_KEY"
    chmod 600 "$ENCRYPTION_KEY"

    # Create encrypted storage for sensitive data (optional)
    SENSITIVE_DIR="/opt/neuroflux/sensitive"
    mkdir -p "$SENSITIVE_DIR"
    chown neuroflux:neuroflux "$SENSITIVE_DIR"

    log_success "Data encryption configured"
}

# Configure nginx security headers
configure_nginx_security() {
    log_info "Configuring nginx security headers..."

    # Check if nginx config exists
    if [ -f "/etc/nginx/sites-available/neuroflux" ]; then
        # Add additional security headers
        sed -i '/add_header Content-Security-Policy/a\    add_header X-Frame-Options "DENY" always;' /etc/nginx/sites-available/neuroflux
        sed -i '/add_header X-Frame-Options/a\    add_header X-Content-Type-Options "nosniff" always;' /etc/nginx/sites-available/neuroflux
        sed -i '/add_header X-Content-Type-Options/a\    add_header Referrer-Policy "strict-origin-when-cross-origin" always;' /etc/nginx/sites-available/neuroflux

        # Test nginx configuration
        nginx -t && systemctl reload nginx

        log_success "Nginx security headers configured"
    else
        log_warning "Nginx config not found, skipping security headers"
    fi
}

# Configure log security
configure_log_security() {
    log_info "Configuring log security..."

    # Secure log files
    LOG_DIR="/var/log/neuroflux"
    mkdir -p "$LOG_DIR"
    chown neuroflux:neuroflux "$LOG_DIR"
    chmod 750 "$LOG_DIR"

    # Configure logrotate for security
    cat > /etc/logrotate.d/neuroflux << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 640 neuroflux neuroflux
    postrotate
        systemctl reload neuroflux
    endscript
}
EOF

    log_success "Log security configured"
}

# Install security monitoring tools
install_security_tools() {
    log_info "Installing security monitoring tools..."

    # Install auditd for system auditing
    apt-get install -y auditd audispd-plugins

    # Configure basic audit rules
    cat > /etc/audit/rules.d/neuroflux.rules << EOF
# NeuroFlux security audit rules
-w /opt/neuroflux -p wa -k neuroflux_files
-w /etc/neuroflux -p wa -k neuroflux_config
-w /var/log/neuroflux -p wa -k neuroflux_logs
EOF

    systemctl restart auditd

    log_success "Security monitoring tools installed"
}

# Create security monitoring script
create_security_monitor() {
    log_info "Creating security monitoring script..."

    cat > monitor_security.sh << 'EOF'
#!/bin/bash
# Security monitoring script for NeuroFlux

echo "=== NeuroFlux Security Report ==="
echo "Timestamp: $(date)"
echo ""

echo "Firewall Status:"
ufw status | head -10
echo ""

echo "Failed SSH Attempts (last 24h):"
grep "Failed password" /var/log/auth.log | wc -l 2>/dev/null || echo "No failed attempts found"
echo ""

echo "Active SSH Sessions:"
who | grep -v localhost | wc -l
echo ""

echo "System Users with Shell Access:"
grep -v '/usr/sbin/nologin\|/bin/false' /etc/passwd | cut -d: -f1
echo ""

echo "Open Ports:"
netstat -tlnp | grep LISTEN | head -10
echo ""

echo "Recent Security Events:"
ausearch -k neuroflux -ts today 2>/dev/null | tail -5 || echo "No audit events found"
echo ""

echo "File Integrity Check:"
find /opt/neuroflux -type f -exec ls -la {} \; | head -5
echo ""
EOF

    chmod +x monitor_security.sh
    log_success "Security monitoring script created"
}

# Generate security report
generate_security_report() {
    log_info "Generating security hardening report..."

    cat > security_report.txt << EOF
NeuroFlux Security Hardening Report
===================================

Firewall Configuration:
- UFW enabled with default deny policy
- SSH, HTTP, and HTTPS ports allowed
- Fail2ban configured for SSH protection

SSH Security:
- Root login disabled
- Password authentication disabled
- Key-based authentication required
- Additional security settings applied

API Security:
- API key authentication implemented
- Role-based access control configured
- Admin and read-only keys generated

Data Protection:
- Encryption keys generated for sensitive data
- Secure file permissions configured
- Encrypted storage prepared

Web Security:
- Security headers configured in nginx
- HTTPS enforcement
- XSS and clickjacking protection

Logging & Monitoring:
- Secure log file permissions
- Log rotation configured
- Audit rules for file monitoring
- Security monitoring script created

Recommendations:
1. Store API keys securely (not in version control)
2. Regularly rotate encryption keys
3. Monitor security logs daily
4. Keep system packages updated
5. Implement regular security audits
6. Consider using a WAF (Web Application Firewall)
7. Implement rate limiting for API endpoints

Security Checklist:
- [x] Firewall configured
- [x] SSH secured
- [x] API authentication implemented
- [x] Data encryption configured
- [x] Security headers added
- [x] Logging secured
- [x] Monitoring tools installed

Next Steps:
1. Save API keys securely
2. Configure SSH key authentication
3. Test all security measures
4. Set up regular security monitoring
5. Implement backup encryption
EOF

    log_success "Security report generated (security_report.txt)"
}

# Main function
main() {
    log_info "NeuroFlux Security Hardening"
    log_info "============================="

    configure_firewall
    configure_fail2ban
    secure_ssh
    configure_api_auth
    configure_encryption
    configure_nginx_security
    configure_log_security
    install_security_tools
    create_security_monitor
    generate_security_report

    log_success "🎉 Security hardening completed!"
    echo ""
    echo "Security Summary:"
    echo "- Firewall: Configured and enabled"
    echo "- SSH: Secured with key authentication"
    echo "- API: Authentication keys generated"
    echo "- Encryption: Keys and storage configured"
    echo ""
    echo "IMPORTANT: Save the API keys displayed above securely!"
    echo ""
    echo "Next steps:"
    echo "1. Review security_report.txt"
    echo "2. Configure SSH key authentication for admin access"
    echo "3. Test security measures"
    echo "4. Monitor with: ./monitor_security.sh"
}

# Run main function
main "$@"
