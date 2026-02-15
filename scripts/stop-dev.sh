#!/bin/bash

# FantaSanremo Team Builder - Stop Development Environment
# This script stops the development servers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Print header
print_message "${BLUE}" "=========================================="
print_message "${BLUE}" "Stopping Development Environment"
print_message "${BLUE}" "=========================================="
echo ""

# Stop backend
if [ -f .backend.pid ]; then
    BACKEND_PID=$(cat .backend.pid)
    print_message "${YELLOW}" "Stopping backend (PID: $BACKEND_PID)..."
    if kill $BACKEND_PID 2>/dev/null; then
        print_message "${GREEN}" "✓ Backend stopped"
    else
        print_message "${YELLOW}" "Backend process already stopped"
    fi
    rm .backend.pid
else
    print_message "${YELLOW}" "No backend PID file found"
fi

# Stop frontend
if [ -f .frontend.pid ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    print_message "${YELLOW}" "Stopping frontend (PID: $FRONTEND_PID)..."
    if kill $FRONTEND_PID 2>/dev/null; then
        print_message "${GREEN}" "✓ Frontend stopped"
    else
        print_message "${YELLOW}" "Frontend process already stopped"
    fi
    rm .frontend.pid
else
    print_message "${YELLOW}" "No frontend PID file found"
fi

# Kill any remaining processes on ports
print_message "${YELLOW}" "Cleaning up any remaining processes..."

# Check port 8000
BACKEND_PORT_PID=$(lsof -ti:8000 2>/dev/null || true)
if [ -n "$BACKEND_PORT_PID" ]; then
    kill -9 $BACKEND_PORT_PID 2>/dev/null || true
    print_message "${GREEN}" "✓ Cleared port 8000"
fi

# Check port 5173
FRONTEND_PORT_PID=$(lsof -ti:5173 2>/dev/null || true)
if [ -n "$FRONTEND_PORT_PID" ]; then
    kill -9 $FRONTEND_PORT_PID 2>/dev/null || true
    print_message "${GREEN}" "✓ Cleared port 5173"
fi

echo ""
print_message "${GREEN}" "=========================================="
print_message "${GREEN}" "Development environment stopped"
print_message "${GREEN}" "=========================================="
