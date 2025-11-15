#!/bin/bash
# NeuroFlux Build Script
# Ensures all dependencies are installed and ready

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() { echo -e "${BLUE}[BUILD]${NC} $1"; }
log_success() { echo -e "${GREEN}[BUILD]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[BUILD]${NC} $1"; }
log_error() { echo -e "${RED}[BUILD]${NC} $1"; }

echo "🏗️ NeuroFlux Build Check"
echo "======================="

# Check if we're in the right directory
if [ ! -f "dashboard_api.py" ]; then
    log_error "dashboard_api.py not found. Please run from NeuroFlux root directory."
    exit 1
fi

# Check Python environment
log_info "Checking Python environment..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    log_success "Python available: $PYTHON_VERSION"
else
    log_error "Python not found. Please install Python 3.8+"
    exit 1
fi

# Check Node.js environment
log_info "Checking Node.js environment..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version 2>&1)
    log_success "Node.js available: $NODE_VERSION"
else
    log_warning "Node.js not found. Dashboard will not be available."
fi

if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version 2>&1)
    log_success "npm available: $NPM_VERSION"
else
    log_warning "npm not found. Dashboard dependencies cannot be installed."
fi

# Check dashboard dependencies
if [ -d "dashboard" ] && [ -f "dashboard/package.json" ]; then
    log_info "Checking dashboard dependencies..."
    cd dashboard

    if [ ! -d "node_modules" ]; then
        log_warning "Dashboard dependencies not installed."
        if command -v npm &> /dev/null; then
            log_info "Installing dashboard dependencies..."
            npm install
            log_success "Dashboard dependencies installed"
        else
            log_error "Cannot install dashboard dependencies - npm not available"
        fi
    else
        log_success "Dashboard dependencies already installed"
    fi

    cd ..
fi

# Check Python dependencies
log_info "Checking Python dependencies..."
if [ -f "requirements.txt" ]; then
    log_info "Python requirements file found"
    # Note: Actual installation should be done via environment manager
else
    log_warning "requirements.txt not found"
fi

log_success "Build check completed!"
echo ""
echo "🎯 Ready to deploy with: ./scripts/deploy_simple.sh"
echo "🛑 Stop services with: ./scripts/stop.sh"