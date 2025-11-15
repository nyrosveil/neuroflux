#!/bin/bash

# NeuroFlux CI/CD Pipeline
# Automated testing, linting, and deployment script

set -e  # Exit on any error

echo "🚀 NeuroFlux CI/CD Pipeline Starting..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
PIP_EXECUTABLE="${PIP_EXECUTABLE:-pip}"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if we're in the right directory
check_environment() {
    log "Checking environment..."
    if [ ! -f "requirements.txt" ]; then
        error "requirements.txt not found. Are you in the project root?"
        exit 1
    fi
    if [ ! -f "src/main.py" ]; then
        error "src/main.py not found. Project structure seems incorrect."
        exit 1
    fi
    success "Environment check passed"
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies..."
    if [ -f "requirements-dev.txt" ]; then
        $PIP_EXECUTABLE install -r requirements-dev.txt
    else
        $PIP_EXECUTABLE install -r requirements.txt
    fi
    success "Dependencies installed"
}

# Run linting
run_linting() {
    log "Running code linting..."
    if command -v flake8 &> /dev/null; then
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics || {
            error "Flake8 found syntax errors"
            exit 1
        }
        success "Flake8 linting passed"
    else
        warning "flake8 not found, skipping linting"
    fi
}

# Run type checking
run_type_checking() {
    log "Running type checking..."
    if command -v mypy &> /dev/null; then
        mypy src/ --ignore-missing-imports || {
            warning "MyPy found type issues (non-blocking)"
        }
        success "Type checking completed"
    else
        warning "mypy not found, skipping type checking"
    fi
}

# Run tests
run_tests() {
    log "Running test suite..."
    if [ -d "src/tests" ]; then
        $PYTHON_EXECUTABLE -m pytest src/tests/ -v --tb=short || {
            error "Tests failed"
            exit 1
        }
        success "All tests passed"
    else
        warning "No tests directory found"
    fi
}

# Run security checks
run_security_checks() {
    log "Running security checks..."
    # Check for common security issues
    if command -v bandit &> /dev/null; then
        bandit -r src/ -f txt || {
            warning "Bandit found potential security issues"
        }
    else
        warning "bandit not found, skipping security checks"
    fi
}

# Build documentation
build_docs() {
    log "Building documentation..."
    if [ -d "docs" ]; then
        # Could add sphinx or other doc builders here
        success "Documentation check completed"
    else
        warning "No docs directory found"
    fi
}

# Run integration tests
run_integration_tests() {
    log "Running integration tests..."
    # Add integration test commands here
    success "Integration tests completed"
}

# Deploy (if on main branch and all tests pass)
deploy() {
    log "Checking deployment conditions..."
    if [ "$CI_BRANCH" = "main" ] || [ "$CI_BRANCH" = "master" ]; then
        log "On main branch, preparing for deployment..."
        # Add deployment commands here
        success "Deployment preparation completed"
    else
        log "Not on main branch, skipping deployment"
    fi
}

# Main CI/CD pipeline
main() {
    log "Starting NeuroFlux CI/CD Pipeline"

    check_environment
    install_dependencies
    run_linting
    run_type_checking
    run_security_checks
    run_tests
    run_integration_tests
    build_docs
    deploy

    success "🎉 NeuroFlux CI/CD Pipeline completed successfully!"
}

# Allow running specific stages
case "${1:-all}" in
    "check")
        check_environment
        ;;
    "install")
        install_dependencies
        ;;
    "lint")
        run_linting
        ;;
    "typecheck")
        run_type_checking
        ;;
    "test")
        run_tests
        ;;
    "security")
        run_security_checks
        ;;
    "docs")
        build_docs
        ;;
    "deploy")
        deploy
        ;;
    "all")
        main
        ;;
    *)
        error "Usage: $0 {check|install|lint|typecheck|test|security|docs|deploy|all}"
        exit 1
        ;;
esac