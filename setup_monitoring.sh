#!/bin/bash
# NeuroFlux Monitoring Setup Script
# Installs and configures Prometheus and Grafana for system monitoring

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

# Install Prometheus
install_prometheus() {
    log_info "Installing Prometheus..."

    # Create prometheus user
    useradd --no-create-home --shell /bin/false prometheus

    # Create directories
    mkdir -p /etc/prometheus
    mkdir -p /var/lib/prometheus

    # Download and install Prometheus
    cd /tmp
    PROMETHEUS_VERSION="2.45.0"
    wget https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz
    tar xvf prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz
    cd prometheus-${PROMETHEUS_VERSION}.linux-amd64

    # Copy binaries
    cp prometheus /usr/local/bin/
    cp promtool /usr/local/bin/

    # Copy config templates
    cp -r consoles /etc/prometheus
    cp -r console_libraries /etc/prometheus

    # Set ownership
    chown -R prometheus:prometheus /etc/prometheus
    chown -R prometheus:prometheus /var/lib/prometheus
    chown prometheus:prometheus /usr/local/bin/prometheus
    chown prometheus:prometheus /usr/local/bin/promtool

    # Cleanup
    cd /
    rm -rf /tmp/prometheus*

    log_success "Prometheus installed"
}

# Configure Prometheus
configure_prometheus() {
    log_info "Configuring Prometheus..."

    # Create prometheus.yml
    cat > /etc/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'neuroflux'
    static_configs:
      - targets: ['localhost:5001']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
EOF

    chown prometheus:prometheus /etc/prometheus/prometheus.yml

    log_success "Prometheus configured"
}

# Create Prometheus systemd service
create_prometheus_service() {
    log_info "Creating Prometheus systemd service..."

    cat > /etc/systemd/system/prometheus.service << EOF
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \
    --config.file /etc/prometheus/prometheus.yml \
    --storage.tsdb.path /var/lib/prometheus/ \
    --web.console.templates=/etc/prometheus/consoles \
    --web.console.libraries=/etc/prometheus/console_libraries

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable prometheus

    log_success "Prometheus service created"
}

# Install Node Exporter
install_node_exporter() {
    log_info "Installing Node Exporter..."

    # Create node_exporter user
    useradd --no-create-home --shell /bin/false node_exporter

    # Download and install
    cd /tmp
    NODE_EXPORTER_VERSION="1.6.1"
    wget https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
    tar xvf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
    cd node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64

    cp node_exporter /usr/local/bin/
    chown node_exporter:node_exporter /usr/local/bin/node_exporter

    # Cleanup
    cd /
    rm -rf /tmp/node_exporter*

    log_success "Node Exporter installed"
}

# Create Node Exporter service
create_node_exporter_service() {
    log_info "Creating Node Exporter systemd service..."

    cat > /etc/systemd/system/node_exporter.service << EOF
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable node_exporter

    log_success "Node Exporter service created"
}

# Install Grafana
install_grafana() {
    log_info "Installing Grafana..."

    # Add Grafana repository
    apt-get install -y apt-transport-https software-properties-common wget
    wget -q -O - https://packages.grafana.com/gpg.key | apt-key add -
    echo "deb https://packages.grafana.com/oss/deb stable main" | tee -a /etc/apt/sources.list.d/grafana.list

    # Update and install
    apt-get update
    apt-get install -y grafana

    log_success "Grafana installed"
}

# Configure Grafana
configure_grafana() {
    log_info "Configuring Grafana..."

    # Enable and start service
    systemctl daemon-reload
    systemctl enable grafana-server

    log_success "Grafana configured"
}

# Create NeuroFlux metrics endpoint
create_metrics_endpoint() {
    log_info "Creating NeuroFlux metrics endpoint..."

    # This would be added to dashboard_api.py
    # For now, create a sample metrics file
    cat > /opt/neuroflux/metrics_sample.txt << EOF
# NeuroFlux Metrics
neuroflux_agents_total 12
neuroflux_agents_active 8
neuroflux_trading_cycles_total 150
neuroflux_predictions_accuracy 0.85
neuroflux_api_requests_total 2500
neuroflux_websocket_connections 5
EOF

    log_success "Metrics endpoint sample created"
}

# Start services
start_services() {
    log_info "Starting monitoring services..."

    systemctl start prometheus
    systemctl start node_exporter
    systemctl start grafana-server

    # Wait for services to start
    sleep 5

    # Check status
    if systemctl is-active --quiet prometheus; then
        log_success "Prometheus started"
    else
        log_error "Prometheus failed to start"
    fi

    if systemctl is-active --quiet node_exporter; then
        log_success "Node Exporter started"
    else
        log_error "Node Exporter failed to start"
    fi

    if systemctl is-active --quiet grafana-server; then
        log_success "Grafana started"
    else
        log_error "Grafana failed to start"
    fi
}

# Main function
main() {
    log_info "NeuroFlux Monitoring Setup"
    log_info "=========================="

    install_prometheus
    configure_prometheus
    create_prometheus_service

    install_node_exporter
    create_node_exporter_service

    install_grafana
    configure_grafana

    create_metrics_endpoint

    start_services

    log_success "🎉 Monitoring setup completed successfully!"
    echo ""
    echo "Monitoring URLs:"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3000 (admin/admin)"
    echo "- Node Exporter: http://localhost:9100"
    echo ""
    echo "Next steps:"
    echo "1. Access Grafana and add Prometheus as data source"
    echo "2. Import NeuroFlux dashboard (dashboard JSON will be provided)"
    echo "3. Configure alerts in Prometheus"
    echo "4. Set up backup and retention policies"
}

# Run main function
main "$@"