#!/bin/bash
# =============================================================================
# LOCALTRIAGE E2E Test Runner
# Uses Newman (Postman CLI) to run API tests
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTION="${SCRIPT_DIR}/postman_collection.json"
ENV_LOCAL="${SCRIPT_DIR}/postman_environment_local.json"
ENV_STAGING="${SCRIPT_DIR}/postman_environment_staging.json"
REPORTS_DIR="${SCRIPT_DIR}/../../reports/e2e"

# Default values
ENVIRONMENT="local"
BASE_URL=""
REPORTERS="cli,htmlextra,junit"
BAIL_ON_FAILURE=false

# Help message
show_help() {
    echo "LOCALTRIAGE E2E Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENV        Environment to use: local, staging, or custom URL"
    echo "  -u, --url URL        Custom base URL (overrides environment)"
    echo "  -r, --reporters      Newman reporters (default: cli,htmlextra,junit)"
    echo "  -b, --bail           Stop on first failure"
    echo "  -f, --folder FOLDER  Run only specific folder from collection"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run against localhost:8000"
    echo "  $0 -e staging                # Run against staging environment"
    echo "  $0 -u http://localhost:9000  # Run against custom URL"
    echo "  $0 -f 'Health & Status'      # Run only health checks"
    echo "  $0 -b                        # Stop on first failure"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -u|--url)
            BASE_URL="$2"
            shift 2
            ;;
        -r|--reporters)
            REPORTERS="$2"
            shift 2
            ;;
        -b|--bail)
            BAIL_ON_FAILURE=true
            shift
            ;;
        -f|--folder)
            FOLDER="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check for Newman
if ! command -v newman &> /dev/null; then
    echo -e "${YELLOW}Newman not found. Installing...${NC}"
    npm install -g newman newman-reporter-htmlextra
fi

# Create reports directory
mkdir -p "${REPORTS_DIR}"

# Determine environment file
case $ENVIRONMENT in
    local)
        ENV_FILE="${ENV_LOCAL}"
        ;;
    staging)
        ENV_FILE="${ENV_STAGING}"
        ;;
    *)
        ENV_FILE="${ENV_LOCAL}"
        ;;
esac

# Build Newman command
NEWMAN_CMD="newman run ${COLLECTION}"
NEWMAN_CMD+=" --environment ${ENV_FILE}"

# Override base URL if provided
if [ -n "${BASE_URL}" ]; then
    NEWMAN_CMD+=" --env-var base_url=${BASE_URL}"
fi

# Add reporters
IFS=',' read -ra REPORTER_ARRAY <<< "${REPORTERS}"
for reporter in "${REPORTER_ARRAY[@]}"; do
    NEWMAN_CMD+=" --reporters ${reporter}"
    
    case $reporter in
        htmlextra)
            NEWMAN_CMD+=" --reporter-htmlextra-export ${REPORTS_DIR}/e2e-report.html"
            ;;
        junit)
            NEWMAN_CMD+=" --reporter-junit-export ${REPORTS_DIR}/e2e-results.xml"
            ;;
    esac
done

# Add folder filter if specified
if [ -n "${FOLDER}" ]; then
    NEWMAN_CMD+=" --folder '${FOLDER}'"
fi

# Add bail option
if [ "${BAIL_ON_FAILURE}" = true ]; then
    NEWMAN_CMD+=" --bail"
fi

# Run tests
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LOCALTRIAGE E2E Tests${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Environment: ${YELLOW}${ENVIRONMENT}${NC}"
if [ -n "${BASE_URL}" ]; then
    echo -e "Base URL: ${YELLOW}${BASE_URL}${NC}"
fi
echo -e "Reports: ${YELLOW}${REPORTS_DIR}${NC}"
echo ""

# Execute Newman
echo -e "${GREEN}Running tests...${NC}"
echo ""
eval ${NEWMAN_CMD}

EXIT_CODE=$?

# Summary
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All E2E tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Some E2E tests failed.${NC}"
    echo -e "${RED}========================================${NC}"
fi

echo ""
echo "Reports saved to: ${REPORTS_DIR}"
echo "  - HTML Report: ${REPORTS_DIR}/e2e-report.html"
echo "  - JUnit XML: ${REPORTS_DIR}/e2e-results.xml"

exit $EXIT_CODE
