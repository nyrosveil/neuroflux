#!/bin/bash
# NeuroFlux Hybrid Deployment Script
# Combines conda (for complex deps) + venv (for isolation)

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() { echo -e "${BLUE}[HYBRID]${NC} $1"; }
log_success() { echo -e "${GREEN}[HYBRID]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[HYBRID]${NC} $1"; }
log_error() { echo -e "${RED}[HYBRID]${NC} $1"; }

echo "🚀 NeuroFlux Hybrid Deployment (Conda + Venv)"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "dashboard_api.py" ]; then
    log_error "dashboard_api.py not found. Please run from NeuroFlux root directory."
    exit 1
fi

# Check if React build exists and build if needed
if [ ! -d "dashboard/build" ]; then
    log_info "Building React dashboard..."
    if [ -d "dashboard" ] && [ -f "dashboard/package.json" ]; then
        cd dashboard
        if command -v npm &> /dev/null; then
            npm install
            npm run build
            log_success "React dashboard built"
        else
            log_warning "npm not found, skipping React build"
        fi
        cd ..
    else
        log_warning "dashboard directory not found, skipping React build"
    fi
fi

# Environment setup and validation
ENV_TYPE=$(bash env_manager.sh detect)

case $ENV_TYPE in
    "conda")
        log_info "Using existing conda environment"
        # Validate conda environment is properly set up
        CONDA_STATUS=$(bash env_manager.sh get_conda_status)
        if [ "$CONDA_STATUS" = "not_initialized" ]; then
            log_warning "Conda environment exists but not initialized"
            log_info "Attempting to initialize conda..."
            if ! bash env_manager.sh ensure_conda_initialized; then
                log_error "Failed to initialize conda. Falling back to venv-only mode."
                ENV_TYPE="venv"
            fi
        fi
        ;;
    "venv")
        log_info "Using existing virtual environment"
        ;;
    "none")
        log_info "Setting up hybrid environment..."

        # Check if conda is available and can be initialized
        if command -v conda &> /dev/null; then
            log_info "Step 1: Setting up conda base environment"
            if bash env_manager.sh setup_conda; then
                log_success "Conda environment created"
            else
                log_warning "Conda setup failed, falling back to venv-only"
                ENV_TYPE="venv_fallback"
            fi
        else
            log_info "Conda not available, using venv-only mode"
            ENV_TYPE="venv_fallback"
        fi

        if [ "$ENV_TYPE" != "venv_fallback" ]; then
            log_info "Step 2: Setting up virtual environment"
            bash env_manager.sh setup_venv
            log_success "Hybrid environment setup complete"
        fi
        ;;
    *)
        log_error "Unknown environment type: $ENV_TYPE"
        exit 1
        ;;
esac

# Handle venv fallback mode
if [ "$ENV_TYPE" = "venv_fallback" ]; then
    log_info "Setting up virtual environment only..."
    bash env_manager.sh setup_venv
    ENV_TYPE="venv"
    log_success "Virtual environment setup complete"
fi

# Activate environment with recovery suggestions
log_info "Activating environment..."
if ! bash env_manager.sh activate; then
    log_error "Failed to activate environment"
    echo ""
    echo "🔧 Troubleshooting suggestions:"
    echo "1. Run diagnostics: bash env_manager.sh get_conda_status"
    echo "2. Reset environment: bash env_manager.sh cleanup && bash start_hybrid.sh"
    echo "3. Manual conda init: conda init bash && source ~/.bashrc"
    echo "4. Force venv-only: export FORCE_VENV_ONLY=true && bash start_hybrid.sh"
    echo ""
    exit 1
fi

# Verify environment and dependencies
log_info "Verifying environment and dependencies..."
python -c "
import sys
print(f'Python: {sys.version}')

# Test core dependencies
try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except ImportError as e:
    print(f'❌ NumPy: {e}')

try:
    import pandas as pd
    print(f'✅ Pandas: {pd.__version__}')
except ImportError as e:
    print(f'❌ Pandas: {e}')

try:
    import flask
    print(f'✅ Flask: {flask.__version__}')
except ImportError as e:
    print(f'❌ Flask: {e}')

try:
    import ccxt
    print(f'✅ CCXT: {ccxt.__version__}')
except ImportError as e:
    print(f'❌ CCXT: {e}')

print('Environment verification complete')
"

# Set environment variables
export FLASK_ENV=${FLASK_ENV:-development}
export FLASK_DEBUG=${FLASK_DEBUG:-true}
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

log_info "Environment variables set:"
echo "  FLASK_ENV=$FLASK_ENV"
echo "  FLASK_DEBUG=$FLASK_DEBUG"
echo "  PYTHONPATH includes: $(pwd)/src"

# Start server with monitoring and auto-restart
log_info "Starting NeuroFlux server..."
log_info "Press Ctrl+C to stop"

# Trap for graceful shutdown
trap 'log_info "Received shutdown signal, stopping server..."; exit 0' INT TERM

# Auto-restart loop
RESTART_COUNT=0
MAX_RESTARTS=5

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    log_info "Starting server (attempt $((RESTART_COUNT + 1))/$MAX_RESTARTS)..."

    if python dashboard_api.py; then
        log_success "Server exited normally"
        break
    else
        RESTART_COUNT=$((RESTART_COUNT + 1))
        if [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
            log_warning "Server crashed, restarting in 5 seconds... ($RESTART_COUNT/$MAX_RESTARTS)"
            sleep 5
        else
            log_error "Server crashed $MAX_RESTARTS times, giving up"
            exit 1
        fi
    fi
done

log_success "NeuroFlux hybrid deployment completed"