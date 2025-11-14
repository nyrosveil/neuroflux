#!/bin/bash
# Test Documentation Completeness

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "Testing documentation completeness..."

# Check if documentation files exist
files=("PRODUCTION_DEPLOYMENT_GUIDE.md" "OPERATIONS_GUIDE.md")

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        log_success "$file exists"
    else
        log_error "$file missing"
        exit 1
    fi
done

# Check deployment guide completeness
if grep -q "## Prerequisites" PRODUCTION_DEPLOYMENT_GUIDE.md && \
   grep -q "## Quick Start Deployment" PRODUCTION_DEPLOYMENT_GUIDE.md && \
   grep -q "## Troubleshooting" PRODUCTION_DEPLOYMENT_GUIDE.md; then
    log_success "Deployment guide has required sections"
else
    log_error "Deployment guide missing required sections"
    exit 1
fi

# Check operations guide completeness
if grep -q "## Daily Operations" OPERATIONS_GUIDE.md && \
   grep -q "## Incident Response" OPERATIONS_GUIDE.md && \
   grep -q "## Monitoring Alerts" OPERATIONS_GUIDE.md; then
    log_success "Operations guide has required sections"
else
    log_error "Operations guide missing required sections"
    exit 1
fi

# Check for key topics in deployment guide
key_topics=("SSL" "security" "monitoring" "performance" "backup")
for topic in "${key_topics[@]}"; do
    if grep -i "$topic" PRODUCTION_DEPLOYMENT_GUIDE.md > /dev/null; then
        log_success "Deployment guide covers $topic"
    else
        log_error "Deployment guide missing $topic coverage"
        exit 1
    fi
done

# Check for key topics in operations guide
ops_topics=("monitoring" "backup" "security" "performance" "incident")
for topic in "${ops_topics[@]}"; do
    if grep -i "$topic" OPERATIONS_GUIDE.md > /dev/null; then
        log_success "Operations guide covers $topic"
    else
        log_error "Operations guide missing $topic coverage"
        exit 1
    fi
done

log_success "🎉 Documentation completeness validated successfully!"
echo "Production deployment and operations documentation is complete."