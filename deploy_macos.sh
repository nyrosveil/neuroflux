#!/bin/bash
# NeuroFlux macOS Development Deployment Script

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "NeuroFlux macOS Development Deployment"
echo "======================================"

# Check dependencies
log_info "Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    log_error "Python3 not found. Please install Python3."
    exit 1
fi
log_success "Python3 found"

# Setup directories
log_info "Setting up directories..."
APP_DIR="$HOME/Development/neuroflux"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="$HOME/Library/Logs/neuroflux"

mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/logs"
mkdir -p "$APP_DIR/static"
mkdir -p "$LOG_DIR"
log_success "Directories created"

# Setup virtual environment
log_info "Setting up virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements_minimal.txt
log_success "Virtual environment created"

# Copy application files
log_info "Copying application files..."
cp -r . "$APP_DIR/"
rm -rf "$APP_DIR/.git" "$APP_DIR/__pycache__" "$APP_DIR"/*.pyc 2>/dev/null || true
log_success "Application files copied"

# Setup environment
log_info "Setting up environment..."
ENV_FILE="$APP_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    cat > "$ENV_FILE" << EOF
# NeuroFlux Development Environment Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=$(openssl rand -hex 32)
HOST=127.0.0.1
PORT=5001
WORKERS=2
LOG_LEVEL=INFO
LOG_FILE=$LOG_DIR/neuroflux.log
ML_ENABLED=True
ML_MODEL_CACHE_DIR=$APP_DIR/models
MAX_MEMORY=1G
CPU_QUOTA=100%
EOF
    log_success "Environment file created"
else
    log_warning "Environment file already exists"
fi

# Create start script
log_info "Creating start script..."
cat > "$APP_DIR/start_neuroflux.sh" << 'EOF'
#!/bin/bash
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="$HOME/Library/Logs/neuroflux"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$APP_DIR/src"
export FLASK_ENV=development
export FLASK_DEBUG=True

mkdir -p "$LOG_DIR"
cd "$APP_DIR"

echo "Starting NeuroFlux development server..."
python -m gunicorn \
    --bind 127.0.0.1:5001 \
    --workers 2 \
    --threads 2 \
    --access-logfile "$LOG_DIR/access.log" \
    --error-logfile "$LOG_DIR/error.log" \
    --log-level info \
    dashboard_api:app
EOF

chmod +x "$APP_DIR/start_neuroflux.sh"
log_success "Start script created"

# Create stop script
log_info "Creating stop script..."
cat > "$APP_DIR/stop_neuroflux.sh" << 'EOF'
#!/bin/bash
echo "Stopping NeuroFlux processes..."
pkill -f "gunicorn.*dashboard_api" || true
pkill -f "neuroflux" || true
echo "NeuroFlux processes stopped"
EOF

chmod +x "$APP_DIR/stop_neuroflux.sh"
log_success "Stop script created"

# Test deployment
log_info "Testing deployment..."
source "$VENV_DIR/bin/activate"
if python3 -c "import sys; sys.path.insert(0, '$APP_DIR/src'); import dashboard_api; print('API import successful')" 2>/dev/null; then
    log_success "API module imports successfully"
else
    log_error "API module import failed"
    exit 1
fi

log_success "🎉 NeuroFlux macOS deployment completed!"
echo ""
echo "To start NeuroFlux:"
echo "  $APP_DIR/start_neuroflux.sh"
echo ""
echo "To stop NeuroFlux:"
echo "  $APP_DIR/stop_neuroflux.sh"
echo ""
echo "Access at: http://localhost:5001"