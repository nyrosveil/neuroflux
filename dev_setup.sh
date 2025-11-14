#!/bin/bash
# NeuroFlux Development Environment Setup
# One-time setup for hybrid conda + venv environment

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() { echo -e "${BLUE}[SETUP]${NC} $1"; }
log_success() { echo -e "${GREEN}[SETUP]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[SETUP]${NC} $1"; }
log_error() { echo -e "${RED}[SETUP]${NC} $1"; }

echo "🚀 NeuroFlux Development Environment Setup"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "dashboard_api.py" ]; then
    log_error "dashboard_api.py not found. Please run from NeuroFlux root directory."
    exit 1
fi

# Check for conda
log_info "Checking for conda..."
if ! command -v conda &> /dev/null; then
    log_error "Conda not found!"
    log_info "Please install Miniconda or Anaconda:"
    log_info "  - Download: https://docs.conda.io/en/latest/miniconda.html"
    log_info "  - Or install with: brew install miniconda"
    exit 1
fi
log_success "Conda found: $(conda --version)"

# Check for npm (for React build)
log_info "Checking for npm..."
if ! command -v npm &> /dev/null; then
    log_warning "npm not found - React dashboard will not be built automatically"
    log_info "Install Node.js from: https://nodejs.org/"
else
    log_success "npm found: $(npm --version)"
fi

# Check current environment status
ENV_TYPE=$(bash env_manager.sh detect)
log_info "Current environment: $ENV_TYPE"

if [ "$ENV_TYPE" = "conda" ]; then
    log_info "Conda environment already exists"
    CONDA_EXISTS=true
else
    CONDA_EXISTS=false
fi

if [ "$ENV_TYPE" = "venv" ]; then
    log_info "Virtual environment already exists"
    VENV_EXISTS=true
else
    VENV_EXISTS=false
fi

# Setup conda base environment
if [ "$CONDA_EXISTS" = false ]; then
    log_info "Setting up conda base environment..."
    bash env_manager.sh setup_conda
else
    log_info "Using existing conda environment"
fi

# Setup virtual environment
if [ "$VENV_EXISTS" = false ]; then
    log_info "Setting up virtual environment..."
    bash env_manager.sh setup_venv
else
    log_info "Using existing virtual environment"
fi

# Activate environment for testing
log_info "Activating environment for testing..."
if ! bash env_manager.sh activate; then
    log_error "Failed to activate environment"
    exit 1
fi

# Test core dependencies
log_info "Testing core dependencies..."
python -c "
import sys
print('Python version:', sys.version)

# Test conda-installed packages
try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
except ImportError as e:
    print('❌ NumPy:', e)

try:
    import pandas as pd
    print('✅ Pandas:', pd.__version__)
except ImportError as e:
    print('❌ Pandas:', e)

try:
    import flask
    print('✅ Flask:', flask.__version__)
except ImportError as e:
    print('❌ Flask:', e)

try:
    import ccxt
    print('✅ CCXT:', ccxt.__version__)
except ImportError as e:
    print('❌ CCXT:', e)

print('Core dependency test complete')
"

# Test configuration loading
log_info "Testing configuration loading..."
python -c "
try:
    from config import config
    print('✅ Config loaded successfully')
    print('Environment:', config.ENV)
    print('Debug:', config.DEBUG)
    print('Host:', f'{config.HOST}:{config.PORT}')
    print('Conda env:', config.CONDA_ENV_NAME or 'None')
    print('Venv path:', config.VENV_PATH or 'None')
except Exception as e:
    print('❌ Config loading failed:', e)
"

# Create environment-specific config files if they don't exist
log_info "Setting up environment configuration files..."

if [ ! -f ".env.development" ]; then
    cat > .env.development << 'EOF'
# NeuroFlux Development Environment
FLASK_ENV=development
FLASK_DEBUG=true
HOST=127.0.0.1
PORT=5001
SECRET_KEY=dev-secret-key-change-in-production

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=./logs/neuroflux_dev.log

# Performance (development settings)
GUNICORN_WORKERS=2
GUNICORN_THREADS=2

# AI APIs (add your keys here)
# OPENAI_API_KEY=your_key_here
# COINGECKO_API_KEY=your_key_here

# NeuroFlux settings
FLUX_SENSITIVITY=0.8
SWARM_SIZE=3
ML_ENABLED=true
EOF
    log_success "Created .env.development"
else
    log_info ".env.development already exists"
fi

if [ ! -f ".env.production" ]; then
    cp .env.production.example .env.production 2>/dev/null || log_warning ".env.production.example not found"
fi

# Create logs directory
mkdir -p logs

# Test basic import
log_info "Testing basic application import..."
python -c "
try:
    import dashboard_api
    print('✅ Application import successful')
except Exception as e:
    print('❌ Application import failed:', e)
    import traceback
    traceback.print_exc()
"

# Create convenience scripts
log_info "Creating convenience scripts..."

# Create run_dev.sh for quick development start
cat > run_dev.sh << 'EOF'
#!/bin/bash
# Quick development server start
export FLASK_ENV=development
export FLASK_DEBUG=true
bash start_hybrid.sh
EOF
chmod +x run_dev.sh

# Create run_prod.sh for production-like testing
cat > run_prod.sh << 'EOF'
#!/bin/bash
# Production-like server start
export FLASK_ENV=production
export FLASK_DEBUG=false
bash start_hybrid.sh
EOF
chmod +x run_prod.sh

log_success "Created convenience scripts: run_dev.sh, run_prod.sh"

# Final instructions
echo ""
log_success "NeuroFlux development environment setup complete!"
echo ""
echo "📋 Next Steps:"
echo "  1. Configure your API keys in .env.development"
echo "  2. Run: bash run_dev.sh (for development)"
echo "  3. Or:  bash monitor.sh start (with process management)"
echo ""
echo "🔧 Available Commands:"
echo "  bash start_hybrid.sh     # Start with auto environment setup"
echo "  bash monitor.sh status   # Check server status"
echo "  bash monitor.sh start    # Start server in background"
echo "  bash monitor.sh stop     # Stop server"
echo "  bash env_manager.sh info # Show environment info"
echo ""
echo "📊 Dashboard Access:"
echo "  Web UI: http://localhost:3000"
echo "  API:    http://localhost:5001"
echo "  Health: http://localhost:5001/api/health"
echo ""
echo "⚠️  Remember to configure your API keys for full functionality!"