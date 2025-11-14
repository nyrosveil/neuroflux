#!/bin/bash
# NeuroFlux macOS Development Setup (Minimal)

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

echo "NeuroFlux macOS Development Setup (Minimal)"
echo "==========================================="

# Check Python
log_info "Checking Python..."
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

# Create virtual environment (but don't install packages)
log_info "Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
log_success "Virtual environment created"

# Copy application files (excluding .git)
log_info "Copying application files..."
mkdir -p "$APP_DIR"
# Copy all files except .git directory
find . -maxdepth 1 -not -name '.git' -not -name '.' -exec cp -r {} "$APP_DIR/" \;
# Clean up any pyc files
find "$APP_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$APP_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
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
echo "Note: You may need to install dependencies first:"
echo "  source venv/bin/activate && pip install -r requirements_minimal.txt"
echo ""

# Try to start, but it may fail if dependencies aren't installed
python3 -c "import flask; print('Flask available')" 2>/dev/null || {
    echo "Flask not installed. Please run:"
    echo "  source venv/bin/activate"
    echo "  pip install flask flask-cors gunicorn"
    exit 1
}

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

# Create install script
log_info "Creating dependency installation script..."
cat > "$APP_DIR/install_deps.sh" << 'EOF'
#!/bin/bash
echo "Installing NeuroFlux dependencies..."
echo "This may take several minutes..."
source venv/bin/activate
pip install -r requirements_minimal.txt
echo "Dependencies installed!"
echo "You can now run: ./start_neuroflux.sh"
EOF

chmod +x "$APP_DIR/install_deps.sh"
log_success "Dependency installation script created"

log_success "🎉 NeuroFlux macOS setup completed!"
echo ""
echo "Next steps:"
echo "1. Install dependencies: $APP_DIR/install_deps.sh"
echo "2. Start NeuroFlux: $APP_DIR/start_neuroflux.sh"
echo "3. Stop NeuroFlux: $APP_DIR/stop_neuroflux.sh"
echo ""
echo "Access at: http://localhost:5001"
echo ""
echo "Note: The full production requirements.txt contains heavy ML packages"
echo "that may not be compatible with your Python version. Use requirements_minimal.txt for basic functionality."