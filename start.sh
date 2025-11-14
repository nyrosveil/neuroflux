#!/bin/bash
# NeuroFlux Single-Command Deployment Script
# Build React và chạy Flask server trong 1 lệnh

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "🚀 NeuroFlux Single-Command Deployment"
echo "======================================"

# Check if React build exists
if [ ! -d "dashboard/build" ]; then
    log_info "Building React dashboard..."
    cd dashboard
    npm run build
    cd ..
    log_success "React dashboard built"
else
    log_info "React build exists, skipping build..."
fi

# Check if conda environment exists
if ! conda info --envs | grep -q neuroflux-env; then
    log_error "Conda environment 'neuroflux-env' not found!"
    log_info "Please run: conda create -n neuroflux-env python=3.11 -y"
    exit 1
fi

# Activate conda environment and start server
log_info "Starting NeuroFlux server..."
conda run -n neuroflux-env python dashboard_api.py