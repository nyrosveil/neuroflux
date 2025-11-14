#!/bin/bash
# NeuroFlux Production Deployment Script
# This script sets up NeuroFlux for production deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="neuroflux"
APP_DIR="/opt/$APP_NAME"
USER_NAME="$APP_NAME"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="/var/log/$APP_NAME"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

create_user() {
    log_info "Creating system user..."
    if ! id "$USER_NAME" &>/dev/null; then
        useradd --system --shell /bin/bash --home "$APP_DIR" --create-home "$USER_NAME"
        log_success "User $USER_NAME created"
    else
        log_warning "User $USER_NAME already exists"
    fi
}

setup_directories() {
    log_info "Setting up directories..."

    # Create application directory
    mkdir -p "$APP_DIR"
    mkdir -p "$APP_DIR/src"
    mkdir -p "$APP_DIR/static"
    mkdir -p "$APP_DIR/logs"

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Set permissions
    chown -R "$USER_NAME:$USER_NAME" "$APP_DIR"
    chown -R "$USER_NAME:$USER_NAME" "$LOG_DIR"

    log_success "Directories created and permissions set"
}

setup_virtualenv() {
    log_info "Setting up Python virtual environment..."

    # Create virtual environment
    sudo -u "$USER_NAME" python3 -m venv "$VENV_DIR"

    # Activate and install dependencies
    sudo -u "$USER_NAME" bash -c "source $VENV_DIR/bin/activate && pip install --upgrade pip"
    sudo -u "$USER_NAME" bash -c "source $VENV_DIR/bin/activate && pip install -r $APP_DIR/requirements.txt"

    log_success "Virtual environment created and dependencies installed"
}

copy_application() {
    log_info "Copying application files..."

    # Copy all files from current directory to app directory
    cp -r . "$APP_DIR/"

    # Remove any unnecessary files
    rm -rf "$APP_DIR/.git"
    rm -rf "$APP_DIR/__pycache__"
    rm -rf "$APP_DIR/*.pyc"

    # Set permissions
    chown -R "$USER_NAME:$USER_NAME" "$APP_DIR"

    log_success "Application files copied"
}

setup_environment() {
    log_info "Setting up environment configuration..."

    # Create .env file if it doesn't exist
    ENV_FILE="$APP_DIR/.env"
    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" << EOF
# NeuroFlux Production Environment Configuration

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=$(openssl rand -hex 32)

# Server Configuration
HOST=127.0.0.1
PORT=5001
WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_FILE=$LOG_DIR/neuroflux.log

# Database (if needed in future)
# DATABASE_URL=postgresql://user:password@localhost/neuroflux

# API Keys (configure these for your exchanges)
# BINANCE_API_KEY=your_api_key
# BINANCE_API_SECRET=your_api_secret
# COINGECKO_API_KEY=your_api_key

# ML Configuration
ML_ENABLED=True
ML_MODEL_CACHE_DIR=$APP_DIR/models

# System Configuration
MAX_MEMORY=2G
CPU_QUOTA=200%
EOF
        chown "$USER_NAME:$USER_NAME" "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        log_success "Environment file created at $ENV_FILE"
        log_warning "Please edit $ENV_FILE with your API keys and configuration"
    else
        log_warning "Environment file already exists"
    fi
}

setup_systemd() {
    log_info "Setting up systemd service..."

    # Copy service file
    cp "$APP_DIR/neuroflux.service" "/etc/systemd/system/"

    # Reload systemd
    systemctl daemon-reload

    # Enable service
    systemctl enable neuroflux

    log_success "Systemd service configured"
}

setup_nginx() {
    log_info "Setting up nginx configuration..."

    # Check if nginx is installed
    if ! command -v nginx &> /dev/null; then
        log_warning "nginx not found. Installing..."
        apt-get update && apt-get install -y nginx
    fi

    # Copy nginx configuration
    cp "$APP_DIR/nginx.conf" "/etc/nginx/sites-available/neuroflux"

    # Create symlink
    ln -sf "/etc/nginx/sites-available/neuroflux" "/etc/nginx/sites-enabled/"

    # Remove default nginx site
    rm -f "/etc/nginx/sites-enabled/default"

    # Test nginx configuration
    nginx -t

    # Reload nginx
    systemctl reload nginx

    log_success "nginx configured"
}

setup_ssl() {
    log_info "Setting up SSL certificates..."

    if command -v certbot &> /dev/null; then
        log_info "SSL certificate setup (manual step required):"
        echo "Run the following command after updating nginx.conf with your domain:"
        echo "certbot --nginx -d your-domain.com"
        log_warning "SSL setup requires manual domain configuration"
    else
        log_warning "certbot not found. Install with: apt-get install certbot python3-certbot-nginx"
    fi
}

setup_monitoring() {
    log_info "Setting up monitoring..."

    # Create logrotate configuration
    cat > "/etc/logrotate.d/neuroflux" << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $USER_NAME $USER_NAME
    postrotate
        systemctl reload neuroflux
    endscript
}
EOF

    log_success "Log rotation configured"
}

test_deployment() {
    log_info "Testing deployment..."

    # Start service
    systemctl start neuroflux

    # Wait for service to start
    sleep 5

    # Check service status
    if systemctl is-active --quiet neuroflux; then
        log_success "NeuroFlux service started successfully"
    else
        log_error "NeuroFlux service failed to start"
        journalctl -u neuroflux --no-pager -n 20
        exit 1
    fi

    # Test API endpoint
    if curl -f -s "http://localhost:5001/api/status" > /dev/null; then
        log_success "API endpoint responding"
    else
        log_error "API endpoint not responding"
        exit 1
    fi

    log_success "Deployment test completed successfully"
}

show_post_installation() {
    log_info "Post-installation instructions:"
    echo ""
    echo "1. Edit environment configuration:"
    echo "   sudo nano $APP_DIR/.env"
    echo ""
    echo "2. Configure nginx with your domain:"
    echo "   sudo nano /etc/nginx/sites-available/neuroflux"
    echo ""
    echo "3. Setup SSL certificates:"
    echo "   sudo certbot --nginx -d your-domain.com"
    echo ""
    echo "4. Start the service:"
    echo "   sudo systemctl start neuroflux"
    echo ""
    echo "5. Check service status:"
    echo "   sudo systemctl status neuroflux"
    echo ""
    echo "6. View logs:"
    echo "   sudo journalctl -u neuroflux -f"
    echo ""
    echo "7. Test API:"
    echo "   curl http://localhost:5001/api/status"
    echo ""
    echo "8. Access web interface:"
    echo "   https://your-domain.com"
}

main() {
    log_info "NeuroFlux Production Deployment Script"
    log_info "====================================="

    check_root

    create_user
    setup_directories
    copy_application
    setup_virtualenv
    setup_environment
    setup_systemd
    setup_nginx
    setup_ssl
    setup_monitoring
    test_deployment

    log_success "NeuroFlux deployment completed!"
    echo ""
    show_post_installation
}

# Run main function
main "$@"