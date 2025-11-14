# NeuroFlux Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying NeuroFlux in a production environment. NeuroFlux is a multi-agent trading system that requires careful setup for security, performance, and reliability.

## Prerequisites

### System Requirements
- Ubuntu 20.04 LTS or later (or equivalent Debian-based system)
- Minimum 4GB RAM, recommended 8GB+
- Minimum 2 CPU cores, recommended 4+
- 20GB+ free disk space
- Root or sudo access

### Network Requirements
- Domain name pointing to server IP
- Ability to obtain SSL certificates (Let's Encrypt)
- Open ports: 22 (SSH), 80 (HTTP), 443 (HTTPS)

### Knowledge Requirements
- Basic Linux system administration
- Command line usage
- Basic networking concepts
- Security best practices

## Quick Start Deployment

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y git curl wget python3 python3-pip nginx certbot python3-certbot-nginx

# Clone repository
git clone https://github.com/your-org/neuroflux.git
cd neuroflux
```

### 2. Automated Deployment

```bash
# Make deployment script executable
chmod +x deploy_production.sh

# Run deployment script (requires root)
sudo ./deploy_production.sh
```

### 3. SSL Certificate Setup

```bash
# Make SSL setup script executable
chmod +x setup_ssl.sh

# Run SSL setup (replace with your domain)
sudo ./setup_ssl.sh
# Follow prompts to enter your domain name
```

### 4. Security Hardening

```bash
# Make security script executable
chmod +x harden_security.sh

# Run security hardening
sudo ./harden_security.sh
# IMPORTANT: Save the API keys displayed during setup
```

### 5. Performance Optimization

```bash
# Make optimization script executable
chmod +x optimize_performance.sh

# Run performance optimization
sudo ./optimize_performance.sh
```

### 6. Monitoring Setup (Optional)

```bash
# Make monitoring script executable
chmod +x setup_monitoring.sh

# Run monitoring setup
sudo ./setup_monitoring.sh
```

## Manual Configuration

### Environment Configuration

After deployment, configure your environment variables:

```bash
sudo nano /opt/neuroflux/.env
```

Required settings:
```bash
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-generated-secret-key

# Server Configuration
HOST=127.0.0.1
PORT=5001
WORKERS=4

# API Keys (configure these for your exchanges)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
COINGECKO_API_KEY=your_coingecko_api_key

# ML Configuration
ML_ENABLED=True
ML_MODEL_CACHE_DIR=/opt/neuroflux/models

# System Configuration
MAX_MEMORY=2G
CPU_QUOTA=200%
```

### API Authentication

The security hardening script generates API keys. Store them securely:

- **Admin Key**: Full access to all API endpoints
- **Read Key**: Read-only access to status and metrics

Use API keys in requests:
```bash
curl -H "X-API-Key: your-admin-key" https://your-domain.com/api/status
```

## Service Management

### Starting the Service

```bash
sudo systemctl start neuroflux
```

### Checking Status

```bash
sudo systemctl status neuroflux
```

### Viewing Logs

```bash
# System logs
sudo journalctl -u neuroflux -f

# Application logs
sudo tail -f /var/log/neuroflux/error.log
sudo tail -f /var/log/neuroflux/access.log
```

### Restarting the Service

```bash
sudo systemctl restart neuroflux
```

### Stopping the Service

```bash
sudo systemctl stop neuroflux
```

## Monitoring and Health Checks

### Health Check Script

```bash
./health_check.sh
```

### Performance Monitoring

```bash
./monitor_performance.sh
```

### Security Monitoring

```bash
./monitor_security.sh
```

### Real-time Monitoring

```bash
./monitor.sh
```

## API Endpoints

### Status Endpoints
- `GET /api/status` - System status and agent information
- `GET /health` - Basic health check
- `GET /metrics` - Prometheus metrics (when monitoring is enabled)

### Trading Endpoints
- `GET /api/dashboard/predictions` - ML predictions
- `POST /api/trade` - Execute trades (admin key required)
- `GET /api/balance` - Account balance

### Agent Endpoints
- `GET /api/agents` - List active agents
- `POST /api/agents/{agent_id}/start` - Start specific agent
- `POST /api/agents/{agent_id}/stop` - Stop specific agent

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check service status
sudo systemctl status neuroflux

# Check logs
sudo journalctl -u neuroflux --no-pager -n 50

# Check if port is available
sudo netstat -tlnp | grep :5001
```

#### API Returns 500 Error
```bash
# Check application logs
sudo tail -f /var/log/neuroflux/error.log

# Check dependencies
sudo -u neuroflux bash -c "cd /opt/neuroflux && source venv/bin/activate && python -c 'import dashboard_api'"
```

#### High Memory Usage
```bash
# Check memory usage
ps aux --sort=-%mem | head

# Restart service
sudo systemctl restart neuroflux
```

#### SSL Certificate Issues
```bash
# Check certificate status
sudo certbot certificates

# Renew certificates
sudo certbot renew

# Reload nginx
sudo systemctl reload nginx
```

### Log Analysis

#### Finding Errors
```bash
grep "ERROR" /var/log/neuroflux/error.log | tail -10
```

#### Performance Issues
```bash
grep "timeout\|slow" /var/log/neuroflux/error.log
```

#### Security Events
```bash
grep "Failed\|unauthorized" /var/log/neuroflux/error.log
```

## Backup and Recovery

### Configuration Backup
```bash
# Backup configuration
tar -czf neuroflux_config_backup_$(date +%Y%m%d).tar.gz /opt/neuroflux/.env /etc/nginx/sites-available/neuroflux /etc/systemd/system/neuroflux.service
```

### Log Backup
```bash
# Backup logs
tar -czf neuroflux_logs_backup_$(date +%Y%m%d).tar.gz /var/log/neuroflux/
```

### Full System Backup
```bash
# Create full backup (excluding virtual environment)
tar -czf neuroflux_full_backup_$(date +%Y%m%d).tar.gz --exclude='/opt/neuroflux/venv' /opt/neuroflux/
```

### Recovery Procedure
```bash
# Stop service
sudo systemctl stop neuroflux

# Restore from backup
tar -xzf neuroflux_backup.tar.gz -C /

# Restart services
sudo systemctl daemon-reload
sudo systemctl restart neuroflux
sudo systemctl reload nginx
```

## Maintenance Tasks

### Daily Tasks
- Monitor system health: `./health_check.sh`
- Check disk space: `df -h`
- Review logs for errors: `grep ERROR /var/log/neuroflux/error.log`

### Weekly Tasks
- Update system packages: `sudo apt update && sudo apt upgrade`
- Rotate logs manually if needed: `sudo logrotate /etc/logrotate.d/neuroflux`
- Check SSL certificate expiry: `sudo certbot certificates`

### Monthly Tasks
- Full system backup
- Review and rotate API keys
- Performance analysis and optimization
- Security audit

### Quarterly Tasks
- Major version updates
- Security hardening review
- Capacity planning
- Disaster recovery testing

## Security Best Practices

### Access Control
- Use SSH key authentication only
- Disable root login
- Limit sudo access
- Use strong passwords

### Network Security
- Keep firewall enabled
- Use HTTPS only
- Implement rate limiting
- Regular security updates

### Data Protection
- Encrypt sensitive data
- Secure API keys
- Regular backups
- Log monitoring

### Monitoring
- Enable all monitoring tools
- Set up alerts
- Regular log review
- Performance monitoring

## Performance Tuning

### Memory Optimization
- Monitor memory usage with `./monitor_performance.sh`
- Adjust Gunicorn workers based on load
- Consider increasing server memory if needed

### CPU Optimization
- Monitor CPU usage
- Adjust worker threads if CPU-bound
- Consider more CPU cores for high load

### Network Optimization
- Monitor network connections
- Adjust timeouts based on usage patterns
- Consider CDN for static assets (if added)

## Scaling Considerations

### Vertical Scaling
- Increase CPU cores
- Add more memory
- Use faster storage

### Horizontal Scaling
- Load balancer setup
- Database clustering (future)
- Redis for session storage (future)

### Microservices Architecture
- Separate API from agents
- Use message queues
- Implement service mesh

## Support and Resources

### Documentation
- This deployment guide
- API documentation (in dashboard)
- Troubleshooting guides

### Community Resources
- GitHub issues
- Documentation wiki
- Community forums

### Professional Services
- System administration consulting
- Security audits
- Performance optimization
- Custom development

## Changelog

### Version 1.0.0
- Initial production deployment guide
- Automated deployment scripts
- Security hardening
- Monitoring setup
- Performance optimization

---

For additional support, please refer to the project documentation or create an issue on GitHub.