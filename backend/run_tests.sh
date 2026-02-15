#!/bin/bash
# Helper script to run FantaSanremo backend tests

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}FantaSanremo Backend Test Suite${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓ Using virtual environment${NC}"
    VENV_PYTHON=".venv/bin/python"
    VENV_PYTEST=".venv/bin/pytest"
else
    echo -e "${RED}✗ Virtual environment not found${NC}"
    echo "Please create it first with: uv venv .venv"
    exit 1
fi

# Check if pytest is installed
if [ ! -f "$VENV_PYTEST" ]; then
    echo -e "${RED}✗ pytest not found in virtual environment${NC}"
    echo "Installing test dependencies..."
    uv pip install -r requirements.txt -r tests/requirements-test.txt
fi

# Parse command line arguments
TEST_TYPE=""
COVERAGE=false
VERBOSE=false
DIAGNOSTICS=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --api) TEST_TYPE="tests/test_api.py" ;;
        --models) TEST_TYPE="tests/test_models.py" ;;
        --ml) TEST_TYPE="tests/test_ml.py" ;;
        --coverage) COVERAGE=true ;;
        --verbose) VERBOSE=true ;;
        --diagnostics) DIAGNOSTICS=true ;;
        --help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --api       Run only API tests"
            echo "  --models    Run only model tests"
            echo "  --ml        Run only ML tests"
            echo "  --coverage  Generate coverage report"
            echo "  --verbose   Show detailed output"
            echo "  --diagnostics  Run ML diagnostics after tests"
            echo "  --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                 # Run all tests"
            echo "  ./run_tests.sh --api           # Run API tests only"
            echo "  ./run_tests.sh --coverage      # Run with coverage report"
            echo "  ./run_tests.sh --ml --diagnostics  # Run ML tests + diagnostics"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
    shift
done

# Build pytest command
PYTEST_CMD="$VENV_PYTEST"

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=. --cov-report=html --cov-report=term"
fi

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add test path
if [ -n "$TEST_TYPE" ]; then
    PYTEST_CMD="$PYTEST_CMD $TEST_TYPE"
else
    PYTEST_CMD="$PYTEST_CMD tests/"
fi

# Add additional options
PYTEST_CMD="$PYTEST_CMD --tb=short --disable-warnings"

echo -e "${GREEN}Running: $PYTEST_CMD${NC}"
echo ""

# Run tests
eval $PYTEST_CMD
TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

DIAG_EXIT_CODE=0
if [ "$DIAGNOSTICS" = true ]; then
    echo ""
    echo -e "${YELLOW}Running ML diagnostics...${NC}"
    echo ""

    uv run python ml/diagnostics/report_unified_format.py
    DIAG_EXIT_CODE=$?

    uv run python ml/diagnostics/check_full_pipeline.py
    if [ $? -ne 0 ]; then
        DIAG_EXIT_CODE=1
    fi

    echo ""
    if [ $DIAG_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ ML diagnostics passed${NC}"
    else
        echo -e "${RED}✗ ML diagnostics failed${NC}"
    fi
fi

if [ "$COVERAGE" = true ]; then
    echo ""
    echo -e "${YELLOW}Coverage report generated in htmlcov/${NC}"
    echo "Open htmlcov/index.html in your browser to view"
fi

FINAL_EXIT_CODE=$TEST_EXIT_CODE
if [ "$DIAGNOSTICS" = true ] && [ $DIAG_EXIT_CODE -ne 0 ]; then
    FINAL_EXIT_CODE=$DIAG_EXIT_CODE
fi

exit $FINAL_EXIT_CODE
