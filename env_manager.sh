#!/bin/bash
# NeuroFlux Environment Manager
# Handles conda + venv hybrid environment setup

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[ENV]${NC} $1"; }
log_success() { echo -e "${GREEN}[ENV]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[ENV]${NC} $1"; }
log_error() { echo -e "${RED}[ENV]${NC} $1"; }

# Detect current shell type
detect_shell() {
    local shell_path="$SHELL"
    local shell_name=$(basename "$shell_path")

    # Handle common shell variants
    case "$shell_name" in
        "zsh")
            echo "zsh"
            ;;
        "bash")
            echo "bash"
            ;;
        "fish")
            echo "fish"
            ;;
        "ash"|"dash"|"sh")
            echo "posix"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Detect current environment type
detect_environment() {
    # Check for neuroflux-base first (preferred)
    if command -v conda &> /dev/null && conda info --envs 2>/dev/null | grep -q neuroflux-base; then
        echo "conda"
    # Also check for neuroflux (legacy/alternative name)
    elif command -v conda &> /dev/null && conda info --envs 2>/dev/null | grep -q '^neuroflux '; then
        echo "conda"
    elif [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
        echo "venv"
    elif [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
        echo "venv"
    else
        echo "none"
    fi
}

# Detect conda initialization and environment status
get_conda_status() {
    if ! command -v conda &> /dev/null; then
        echo "not_installed"
        return
    fi

    # Test if conda commands work (indicates initialization)
    if ! conda info --envs >/dev/null 2>&1; then
        echo "not_initialized"
        return
    fi

    # Check if neuroflux-base or neuroflux environment exists
    if conda info --envs | grep -q neuroflux-base || conda info --envs | grep -q '^neuroflux '; then
        echo "environment_ready"
    else
        echo "environment_missing"
    fi
}

# Check and initialize conda if needed
ensure_conda_initialized() {
    local status=$(get_conda_status)

    case $status in
        "not_installed")
            log_error "Conda not found. Please install Miniconda or Anaconda first."
            log_info "Download: https://docs.conda.io/en/latest/miniconda.html"
            return 1
            ;;
        "not_initialized")
            log_info "Initializing conda..."
            if ! initialize_conda; then
                log_error "Failed to initialize conda"
                return 1
            fi
            ;;
        "environment_missing")
            log_info "Conda initialized but environment missing"
            return 2  # Signal to create environment
            ;;
        "environment_ready")
            log_info "Conda environment ready"
            ;;
    esac

    return 0
}

# Initialize conda for current shell
initialize_conda() {
    local shell_type=$(detect_shell)
    log_info "Initializing conda for $shell_type shell..."

    case "$shell_type" in
        "zsh")
            init_conda_zsh
            ;;
        "bash")
            init_conda_bash
            ;;
        "fish")
            init_conda_fish
            ;;
        *)
            init_conda_fallback
            ;;
    esac
}

# Zsh-specific conda initialization
init_conda_zsh() {
    log_info "Setting up conda for Zsh..."

    # Try conda init zsh
    if command -v conda &> /dev/null; then
        conda init zsh >/dev/null 2>&1 || true
    fi

    # Source zsh configuration files
    local zsh_configs=("$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.zshenv")
    for config in "${zsh_configs[@]}"; do
        if [ -f "$config" ]; then
            source "$config" >/dev/null 2>&1 || true
        fi
    done

    # Check if conda is now available
    if conda info --envs >/dev/null 2>&1; then
        log_success "Conda initialized for Zsh"
        return 0
    else
        log_warning "Conda init failed, trying manual setup"
        setup_conda_path_manual
        return $?
    fi
}

# Bash-specific conda initialization
init_conda_bash() {
    log_info "Setting up conda for Bash..."

    # Try conda init bash
    if command -v conda &> /dev/null; then
        conda init bash >/dev/null 2>&1 || true
    fi

    # Source bash configuration files
    local bash_configs=("$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile")
    for config in "${bash_configs[@]}"; do
        if [ -f "$config" ]; then
            source "$config" >/dev/null 2>&1 || true
        fi
    done

    # Check if conda is now available
    if conda info --envs >/dev/null 2>&1; then
        log_success "Conda initialized for Bash"
        return 0
    else
        log_warning "Conda init failed, trying manual setup"
        setup_conda_path_manual
        return $?
    fi
}

# Fish shell conda initialization
init_conda_fish() {
    log_warning "Fish shell detected - conda initialization may not work properly"
    log_info "Consider using bash or zsh for NeuroFlux"
    init_conda_fallback
}

# Fallback conda initialization for unknown shells
init_conda_fallback() {
    log_info "Using fallback conda initialization..."

    # Try bash init as default
    if command -v conda &> /dev/null; then
        conda init bash >/dev/null 2>&1 || true
    fi

    # Source common profile files
    local profiles=("$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile" "$HOME/.zshrc")
    for profile in "${profiles[@]}"; do
        if [ -f "$profile" ]; then
            source "$profile" >/dev/null 2>&1 || true
        fi
    done

    # Manual path setup if still not working
    if ! conda info --envs >/dev/null 2>&1; then
        setup_conda_path_manual
    fi
}

# Manual conda path setup (works for any shell)
setup_conda_path_manual() {
    local conda_paths=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "/opt/miniconda3"
        "/usr/local/miniconda3"
        "/opt/conda"
    )

    for conda_path in "${conda_paths[@]}"; do
        if [ -d "$conda_path" ] && [ -f "$conda_path/bin/conda" ]; then
            # Add to PATH
            export PATH="$conda_path/bin:$PATH"

            # Set conda environment variables
            export CONDA_EXE="$conda_path/bin/conda"
            export CONDA_ROOT="$conda_path"
            export _CONDA_EXE="$conda_path/bin/conda"
            export _CONDA_ROOT="$conda_path"

            # Source conda setup if available
            if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
                source "$conda_path/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
            fi

            # Test if it works
            if conda info --envs >/dev/null 2>&1; then
                log_success "Manual conda setup successful: $conda_path"
                return 0
            fi
        fi
    done

    log_error "Manual conda setup failed"
    return 1
}

# Setup conda path directly (for systems where init fails)
setup_conda_path() {
    local conda_paths=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "/opt/miniconda3"
        "/usr/local/miniconda3"
        "/opt/conda"
    )

    for conda_path in "${conda_paths[@]}"; do
        if [ -d "$conda_path" ] && [ -f "$conda_path/bin/conda" ]; then
            export PATH="$conda_path/bin:$PATH"
            export CONDA_AUTO_UPDATE_CONDA=false

            # Source conda script if it exists
            if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
                source "$conda_path/etc/profile.d/conda.sh"
            fi

            # Test if it works
            if conda info --envs >/dev/null 2>&1; then
                log_success "Conda path setup successful: $conda_path"
                return 0
            fi
        fi
    done

    return 1
}

# Direct conda environment activation (bypasses conda init issues)
activate_conda_env_direct() {
    local env_name="$1"
    local conda_base=""

    # Find conda installation
    if [ -d "/opt/homebrew/Caskroom/miniforge/base" ]; then
        conda_base="/opt/homebrew/Caskroom/miniforge/base"
    elif [ -d "$HOME/miniconda3" ]; then
        conda_base="$HOME/miniconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        conda_base="$HOME/anaconda3"
    elif [ -d "/opt/conda" ]; then
        conda_base="/opt/conda"
    else
        # Try to find conda base from which conda
        local conda_path=$(which conda 2>/dev/null)
        if [ -n "$conda_path" ]; then
            conda_base=$(dirname $(dirname "$conda_path"))
        fi
    fi

    if [ -z "$conda_base" ] || [ ! -d "$conda_base/envs/$env_name" ]; then
        log_warning "Could not find conda environment $env_name, falling back to conda activate"
        conda activate "$env_name" 2>/dev/null || return 1
        return 0
    fi

    # Direct environment activation
    local env_path="$conda_base/envs/$env_name"

    # Set environment variables directly
    export CONDA_DEFAULT_ENV="$env_name"
    export CONDA_PREFIX="$env_path"
    export CONDA_PROMPT_MODIFIER="($env_name) "

    # Update PATH
    export PATH="$env_path/bin:$PATH"

    # Source environment activation script if it exists
    if [ -f "$env_path/bin/activate" ]; then
        source "$env_path/bin/activate" >/dev/null 2>&1 || true
    fi

    # Verify activation worked
    if [ "$CONDA_PREFIX" = "$env_path" ]; then
        log_success "Direct conda environment activation successful"
        return 0
    else
        log_warning "Direct activation failed, trying conda activate"
        conda activate "$env_name" 2>/dev/null || return 1
        return 0
    fi
}

# Setup conda base environment with optimized packages
setup_conda_base() {
    log_info "Setting up conda base environment (neuroflux-base)..."

    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install Miniconda or Anaconda first."
        log_info "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    # Create conda environment with optimized scientific packages
    log_info "Creating conda environment with scientific packages..."
    conda create -n neuroflux-base python=3.11 \
        numpy scipy pandas scikit-learn numba \
        matplotlib seaborn jupyter \
        -c conda-forge -y

    # Activate and install additional packages
    log_info "Activating conda environment and installing additional packages..."
    conda activate neuroflux-base

    # Install conda-specific packages that are hard to install with pip
    conda install -c conda-forge \
        ta-lib \
        python-dotenv \
        flask flask-cors flask-socketio \
        gunicorn eventlet gevent \
        requests aiohttp \
        -y

    log_success "Conda base environment created successfully"
}

# Fallback: Pure virtual environment mode (no conda)
setup_pure_venv() {
    log_info "Setting up pure virtual environment (no conda)..."

    # Create virtual environment
    python3 -m venv .venv

    # Activate and upgrade pip
    source .venv/bin/activate
    pip install --upgrade pip

    # Install all dependencies via pip
    if [ -f "requirements.txt" ]; then
        log_info "Installing all dependencies via pip..."
        pip install -r requirements.txt
    fi

    # Try to install conda-specific packages via pip
    log_info "Attempting to install scientific packages via pip..."
    pip install numpy scipy pandas scikit-learn numba \
               matplotlib seaborn jupyter \
               flask flask-cors flask-socketio \
               gunicorn eventlet gevent \
               ccxt python-dotenv requests aiohttp || true

    log_success "Pure virtual environment created"
}

# Detect if we should use pure venv mode
should_use_pure_venv() {
    # Use pure venv if:
    # 1. FORCE_VENV_ONLY environment variable is set
    # 2. Conda is not available
    # 3. Conda initialization consistently fails

    if [ "${FORCE_VENV_ONLY:-false}" = "true" ]; then
        return 0
    fi

    if ! command -v conda &> /dev/null; then
        return 0
    fi

    # Check if conda has failed before
    if [ -f ".conda_failed" ]; then
        return 0
    fi

    return 1
}

# Setup virtual environment for project isolation
setup_venv() {
    log_info "Setting up virtual environment (.venv)..."

    # Check if we should use pure venv mode
    if should_use_pure_venv; then
        setup_pure_venv
        return $?
    fi

    # Try conda-based setup first
    local conda_available=false
    if command -v conda &> /dev/null && conda info --envs 2>/dev/null | grep -q neuroflux-base; then
        if conda activate neuroflux-base 2>/dev/null; then
            conda_available=true
            log_info "Using conda base environment for venv creation"
        fi
    fi

    # Create virtual environment
    python -m venv .venv

    # Activate and upgrade pip
    source .venv/bin/activate
    pip install --upgrade pip

    # Install project dependencies
    if [ -f "requirements.txt" ]; then
        log_info "Installing project dependencies..."
        pip install -r requirements.txt
    else
        log_warning "requirements.txt not found, installing minimal dependencies..."
        pip install flask flask-cors flask-socketio python-dotenv
    fi

    # Mark conda as failed if we couldn't use it
    if [ "$conda_available" = false ] && command -v conda &> /dev/null; then
        touch .conda_failed
        log_warning "Conda setup failed, marked for future venv-only mode"
    fi

    log_success "Virtual environment created successfully"
}

# Activate appropriate environment with initialization checks
activate_environment() {
    env_type=$(detect_environment)

    case $env_type in
        "conda")
            # Ensure conda is ready
            if ! ensure_conda_initialized; then
                log_error "Failed to initialize conda environment"
                return 1
            fi

            # Try direct environment activation (bypass conda init issues)
            if conda info --envs | grep -q neuroflux-base; then
                log_info "Activating conda environment (neuroflux-base)..."
                activate_conda_env_direct "neuroflux-base"
            elif conda info --envs | grep -q '^neuroflux '; then
                log_info "Activating conda environment (neuroflux)..."
                activate_conda_env_direct "neuroflux"
            else
                log_error "No suitable conda environment found"
                return 1
            fi

            # Activate venv layer if it exists
            if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
                log_info "Activating virtual environment (.venv)..."
                source .venv/bin/activate
            fi
            ;;
        "venv")
            log_info "Activating virtual environment..."
            if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
                source .venv/bin/activate
            elif [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
                source venv/bin/activate
            fi
            ;;
        "none")
            log_warning "No environment found. Run setup first."
            return 1
            ;;
        *)
            log_error "Unknown environment type: $env_type"
            return 1
            ;;
    esac

    return 0
}

# Clean up environments with safe conda operations
cleanup_environments() {
    log_info "Cleaning up environments..."

    # Safe conda cleanup - only if conda is initialized
    if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
        if conda info --envs >/dev/null 2>&1 2>&1; then
            # Conda is initialized, safe to deactivate
            conda deactivate >/dev/null 2>&1 || true
            log_success "Conda environment deactivated"
        else
            # Conda not initialized, manual cleanup
            log_info "Conda not initialized, performing manual environment cleanup"
            cleanup_conda_env_vars
        fi
    fi

    # Remove virtual environments
    for venv_dir in ".venv" "venv"; do
        if [ -d "$venv_dir" ]; then
            rm -rf "$venv_dir"
            log_success "Removed $venv_dir directory"
        fi
    done

    # Safe conda environment removal
    if command -v conda &> /dev/null; then
        if conda info --envs >/dev/null 2>&1 2>&1; then
            # Conda is initialized, safe to remove environment
            conda env remove -n neuroflux-base -y >/dev/null 2>&1 || true
            log_success "Removed conda environment neuroflux-base"
        else
            log_info "Conda not initialized, skipping environment removal"
        fi
    fi

    # Clean up marker files
    rm -f .conda_failed 2>/dev/null || true

    log_success "Environment cleanup complete"
}

# Manual conda environment variable cleanup
cleanup_conda_env_vars() {
    # List of conda-related environment variables to clean up
    local conda_vars=(
        CONDA_DEFAULT_ENV
        CONDA_PREFIX
        CONDA_PROMPT_MODIFIER
        CONDA_SHLVL
        CONDA_EXE
        CONDA_PYTHON_EXE
        CONDA_ROOT
        _CONDA_EXE
        _CONDA_ROOT
    )

    for var in "${conda_vars[@]}"; do
        unset "$var" 2>/dev/null || true
    done

    # Clean conda paths from PATH
    local conda_paths=(
        "$HOME/miniconda3/bin"
        "$HOME/anaconda3/bin"
        "/opt/miniconda3/bin"
        "/usr/local/miniconda3/bin"
        "/opt/conda/bin"
    )

    local new_path=""
    local path_array=("${PATH//:/ }")  # Split PATH by colon
    for path_element in "${path_array[@]}"; do
        local keep=true
        for conda_path in "${conda_paths[@]}"; do
            if [[ "$path_element" == "$conda_path" ]]; then
                keep=false
                break
            fi
        done
        if [ "$keep" = true ]; then
            if [ -z "$new_path" ]; then
                new_path="$path_element"
            else
                new_path="$new_path:$path_element"
            fi
        fi
    done

    export PATH="$new_path"
    log_success "Manual conda environment cleanup completed"
}

# Show environment information
show_info() {
    echo "NeuroFlux Environment Information"
    echo "=================================="

    env_type=$(detect_environment)
    echo "Environment Type: $env_type"

    if command -v conda &> /dev/null; then
        echo "Conda Available: Yes"
        if conda info --envs 2>/dev/null | grep -q neuroflux-base; then
            echo "Conda Environment: neuroflux-base (exists)"
        else
            echo "Conda Environment: neuroflux-base (not created)"
        fi
    else
        echo "Conda Available: No"
    fi

    if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
        echo "Virtual Environment: .venv (exists)"
    elif [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
        echo "Virtual Environment: venv (exists)"
    else
        echo "Virtual Environment: Not created"
    fi

    # Check Python version
    if command -v python &> /dev/null; then
        python_version=$(python --version 2>&1)
        echo "Python Version: $python_version"
    fi

    # Check if in activated environment
    if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
        echo "Active Conda Env: $CONDA_DEFAULT_ENV"
    fi

    if [ ! -z "$VIRTUAL_ENV" ]; then
        echo "Active Virtual Env: $(basename $VIRTUAL_ENV)"
    fi
}

# Run comprehensive diagnostics
run_diagnostics() {
    echo "NeuroFlux Environment Diagnostics"
    echo "=================================="

    echo "1. System Information:"
    echo "   OS: $(uname -s)"
    echo "   Shell: $SHELL ($(detect_shell))"
    echo "   Python: $(python3 --version 2>&1 || echo 'Not found')"

    echo ""
    echo "2. Conda Status:"
    if command -v conda &> /dev/null; then
        echo "   ✅ Conda installed: $(conda --version)"
        if conda info --envs >/dev/null 2>&1; then
            echo "   ✅ Conda initialized"
            if conda info --envs | grep -q neuroflux-base; then
                echo "   ✅ neuroflux-base environment exists"
            else
                echo "   ❌ neuroflux-base environment missing"
            fi
        else
            echo "   ❌ Conda not initialized"
        fi
    else
        echo "   ❌ Conda not installed"
    fi

    echo ""
    echo "3. Virtual Environment:"
    if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
        echo "   ✅ .venv exists"
        if [ -f ".venv/bin/python" ]; then
            echo "   ✅ Python available in .venv"
        fi
    else
        echo "   ❌ .venv missing or incomplete"
    fi

    echo ""
    echo "4. Configuration Files:"
    for env_file in .env .env.development .env.production; do
        if [ -f "$env_file" ]; then
            echo "   ✅ $env_file exists"
        else
            echo "   ❌ $env_file missing"
        fi
    done

    echo ""
    echo "5. Recommendations:"
    local issues=0

    if ! command -v conda &> /dev/null; then
        echo "   - Install Miniconda for optimal performance"
        ((issues++))
    elif ! conda info --envs >/dev/null 2>&1; then
        echo "   - Run: conda init bash"
        ((issues++))
    fi

    if [ ! -d ".venv" ]; then
        echo "   - Run: bash env_manager.sh setup_venv"
        ((issues++))
    fi

    if [ $issues -eq 0 ]; then
        echo "   ✅ All systems go!"
    else
        echo "   ⚠️  $issues issues need attention"
    fi
}

# Main command handling
case "$1" in
    detect)
        detect_environment
        ;;
    detect_shell)
        detect_shell
        ;;
    get_conda_status)
        get_conda_status
        ;;
    setup_conda)
        setup_conda_base
        ;;
    setup_venv)
        setup_venv
        ;;
    activate)
        activate_environment
        ;;
    cleanup)
        cleanup_environments
        ;;
    info)
        show_info
        ;;
    doctor)
        run_diagnostics
        ;;
    *)
        echo "Usage: $0 {detect|detect_shell|get_conda_status|setup_conda|setup_venv|activate|cleanup|info|doctor}"
        echo ""
        echo "Commands:"
        echo "  detect           - Detect current environment type"
        echo "  detect_shell     - Detect current shell type"
        echo "  get_conda_status - Check conda initialization status"
        echo "  setup_conda      - Setup conda base environment"
        echo "  setup_venv       - Setup virtual environment"
        echo "  activate         - Activate appropriate environment"
        echo "  cleanup          - Remove all environments"
        echo "  info             - Show environment information"
        echo "  doctor           - Run comprehensive diagnostics"
        exit 1
        ;;
esac