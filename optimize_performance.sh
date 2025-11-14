#!/bin/bash
# NeuroFlux Performance Optimization Script
# Analyzes system resources and optimizes Gunicorn and system settings

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

# Get system information
get_system_info() {
    log_info "Analyzing system resources..."

    # CPU cores
    CPU_CORES=$(nproc)
    echo "CPU_CORES=$CPU_CORES"

    # Total memory in GB
    TOTAL_MEM_GB=$(free -g | grep '^Mem:' | awk '{print $2}')
    echo "TOTAL_MEM_GB=$TOTAL_MEM_GB"

    # Memory per core
    MEM_PER_CORE=$(( TOTAL_MEM_GB * 1024 / CPU_CORES ))  # MB per core
    echo "MEM_PER_CORE=$MEM_PER_CORE"
}

# Calculate optimal Gunicorn settings
calculate_gunicorn_settings() {
    local cpu_cores=$1
    local total_mem_gb=$2
    local mem_per_core=$3

    log_info "Calculating optimal Gunicorn settings..."

    # Workers: (2 * CPU_CORES) + 1
    GUNICORN_WORKERS=$(( 2 * cpu_cores + 1 ))
    echo "GUNICORN_WORKERS=$GUNICORN_WORKERS"

    # Threads: 2-4 per worker (start with 2)
    GUNICORN_THREADS=2
    echo "GUNICORN_THREADS=$GUNICORN_THREADS"

    # Worker class: gthread for better performance
    GUNICORN_WORKER_CLASS="gthread"
    echo "GUNICORN_WORKER_CLASS=$GUNICORN_WORKER_CLASS"

    # Max requests per worker before restart
    GUNICORN_MAX_REQUESTS=1000
    echo "GUNICORN_MAX_REQUESTS=$GUNICORN_MAX_REQUESTS"

    # Max requests jitter
    GUNICORN_MAX_REQUESTS_JITTER=50
    echo "GUNICORN_MAX_REQUESTS_JITTER=$GUNICORN_MAX_REQUESTS_JITTER"

    # Timeout
    GUNICORN_TIMEOUT=30
    echo "GUNICORN_TIMEOUT=$GUNICORN_TIMEOUT"

    # Keep alive
    GUNICORN_KEEP_ALIVE=10
    echo "GUNICORN_KEEP_ALIVE=$GUNICORN_KEEP_ALIVE"
}

# Calculate memory limits
calculate_memory_limits() {
    local total_mem_gb=$1
    local gunicorn_workers=$2

    log_info "Calculating memory limits..."

    # Reserve 2GB for system
    SYSTEM_RESERVED_GB=2

    # Available memory for application
    APP_MEM_GB=$(( total_mem_gb - SYSTEM_RESERVED_GB ))
    if [ $APP_MEM_GB -lt 2 ]; then
        APP_MEM_GB=2
        log_warning "Low memory system detected, using minimum allocation"
    fi

    # Memory limit per worker (MB)
    MEM_LIMIT_PER_WORKER=$(( APP_MEM_GB * 1024 / gunicorn_workers ))
    echo "MEM_LIMIT_PER_WORKER=${MEM_LIMIT_PER_WORKER}MB"

    # Total memory limit
    TOTAL_MEM_LIMIT=$(( APP_MEM_GB * 1024 ))
    echo "TOTAL_MEM_LIMIT=${TOTAL_MEM_LIMIT}MB"
}

# Update systemd service with optimized settings
update_systemd_service() {
    local gunicorn_workers=$1
    local gunicorn_threads=$2
    local gunicorn_worker_class=$3
    local gunicorn_max_requests=$4
    local gunicorn_max_requests_jitter=$5
    local gunicorn_timeout=$6
    local gunicorn_keep_alive=$7
    local mem_limit_per_worker=$8

    log_info "Updating systemd service with optimized settings..."

    # Backup original service file
    cp neuroflux.service neuroflux.service.backup

    # Update ExecStart with optimized settings
    sed -i "s|-w [0-9]*|-w $gunicorn_workers|g" neuroflux.service
    sed -i "s|ExecStart=/opt/neuroflux/venv/bin/gunicorn.*|ExecStart=/opt/neuroflux/venv/bin/gunicorn -w $gunicorn_workers --threads $gunicorn_threads -k $gunicorn_worker_class --max-requests $gunicorn_max_requests --max-requests-jitter $gunicorn_max_requests_jitter --timeout $gunicorn_timeout --keep-alive $gunicorn_keep_alive -b 127.0.0.1:5001 --access-logfile /var/log/neuroflux/access.log --error-logfile /var/log/neuroflux/error.log --log-level info dashboard_api:app|g" neuroflux.service

    # Update memory limit
    sed -i "s|MemoryLimit=.*|MemoryLimit=${mem_limit_per_worker}M|g" neuroflux.service

    log_success "Systemd service updated"
}

# Optimize system settings
optimize_system_settings() {
    log_info "Optimizing system settings..."

    # Increase file descriptors limit
    if ! grep -q "neuroflux.*nofile" /etc/security/limits.conf; then
        echo "neuroflux soft nofile 65536" >> /etc/security/limits.conf
        echo "neuroflux hard nofile 65536" >> /etc/security/limits.conf
        log_success "File descriptor limits increased"
    fi

    # Optimize kernel parameters for network performance
    cat > /etc/sysctl.d/99-neuroflux.conf << EOF
# NeuroFlux network optimizations
net.core.somaxconn = 65536
net.ipv4.tcp_max_syn_backlog = 65536
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
EOF

    sysctl -p /etc/sysctl.d/99-neuroflux.conf
    log_success "System network parameters optimized"
}

# Create performance monitoring script
create_performance_monitor() {
    log_info "Creating performance monitoring script..."

    cat > monitor_performance.sh << 'EOF'
#!/bin/bash
# Performance monitoring script for NeuroFlux

echo "=== NeuroFlux Performance Report ==="
echo "Timestamp: $(date)"
echo ""

echo "System Resources:"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df / | tail -1 | awk '{print $5}')"
echo ""

echo "NeuroFlux Processes:"
ps aux | head -1
ps aux | grep -E "(neuroflux|gunicorn)" | grep -v grep | head -10
echo ""

echo "Network Connections:"
netstat -tun | grep ESTABLISHED | wc -l
echo ""

echo "Gunicorn Workers:"
ps aux | grep gunicorn | grep -v grep | wc -l
echo ""

echo "Recent Logs:"
tail -20 /var/log/neuroflux/error.log 2>/dev/null || echo "No error logs found"
echo ""
EOF

    chmod +x monitor_performance.sh
    log_success "Performance monitoring script created"
}

# Generate performance report
generate_report() {
    local cpu_cores=$1
    local total_mem_gb=$2
    local gunicorn_workers=$3
    local mem_limit_per_worker=$4

    log_info "Generating performance optimization report..."

    cat > performance_report.txt << EOF
NeuroFlux Performance Optimization Report
========================================

System Analysis:
- CPU Cores: $cpu_cores
- Total Memory: ${total_mem_gb}GB
- Memory per Core: $(( total_mem_gb * 1024 / cpu_cores ))MB

Recommended Gunicorn Configuration:
- Workers: $gunicorn_workers (2 * CPU_CORES + 1)
- Worker Class: gthread
- Threads per Worker: 2
- Max Requests per Worker: 1000
- Memory Limit per Worker: ${mem_limit_per_worker}MB

System Optimizations Applied:
- File descriptor limits increased to 65536
- Network parameters optimized for high concurrency
- Memory limits configured in systemd service

Monitoring:
- Performance monitoring script created (monitor_performance.sh)
- Real-time monitoring available via monitor.sh
- Health checks configured via health_check.sh

Recommendations:
1. Monitor memory usage under load
2. Adjust worker count based on actual usage patterns
3. Consider using more threads if CPU-bound
4. Monitor for memory leaks and restart workers periodically
5. Use connection pooling for database connections (if added later)

To apply these changes:
1. Run: sudo systemctl daemon-reload
2. Run: sudo systemctl restart neuroflux
3. Monitor: ./monitor_performance.sh
EOF

    log_success "Performance report generated (performance_report.txt)"
}

# Main function
main() {
    log_info "NeuroFlux Performance Optimization"
    log_info "==================================="

    # Get system information
    SYS_INFO=$(get_system_info)
    eval "$SYS_INFO"

    # Calculate optimal settings
    GUNICORN_SETTINGS=$(calculate_gunicorn_settings "$CPU_CORES" "$TOTAL_MEM_GB" "$MEM_PER_CORE")
    eval "$GUNICORN_SETTINGS"

    # Calculate memory limits
    MEM_LIMITS=$(calculate_memory_limits "$TOTAL_MEM_GB" "$GUNICORN_WORKERS")
    eval "$MEM_LIMITS"

    # Update systemd service
    update_systemd_service "$GUNICORN_WORKERS" "$GUNICORN_THREADS" "$GUNICORN_WORKER_CLASS" \
                          "$GUNICORN_MAX_REQUESTS" "$GUNICORN_MAX_REQUESTS_JITTER" \
                          "$GUNICORN_TIMEOUT" "$GUNICORN_KEEP_ALIVE" "$MEM_LIMIT_PER_WORKER"

    # Optimize system settings
    optimize_system_settings

    # Create performance monitor
    create_performance_monitor

    # Generate report
    generate_report "$CPU_CORES" "$TOTAL_MEM_GB" "$GUNICORN_WORKERS" "$MEM_LIMIT_PER_WORKER"

    log_success "🎉 Performance optimization completed!"
    echo ""
    echo "Optimization Summary:"
    echo "- Gunicorn Workers: $GUNICORN_WORKERS"
    echo "- Memory Limit per Worker: ${MEM_LIMIT_PER_WORKER}MB"
    echo "- System optimizations applied"
    echo ""
    echo "Next steps:"
    echo "1. Review performance_report.txt"
    echo "2. Test with: sudo systemctl restart neuroflux"
    echo "3. Monitor with: ./monitor_performance.sh"
}

# Run main function
main "$@"
