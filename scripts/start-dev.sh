#!/bin/bash

# FantaSanremo Team Builder - Development Environment Startup Script
# This script starts both backend and frontend in development mode with hot reload

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Print header
print_message "${BLUE}" "=========================================="
print_message "${BLUE}" "FantaSanremo Team Builder - Dev Mode"
print_message "${BLUE}" "=========================================="
echo ""

# Check prerequisites
print_message "${YELLOW}" "Checking prerequisites..."

if ! command_exists uv; then
    print_message "${RED}" "Error: uv is not installed"
    print_message "${YELLOW}" "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command_exists npm; then
    print_message "${RED}" "Error: npm is not installed"
    exit 1
fi

print_message "${GREEN}" "✓ uv found"
print_message "${GREEN}" "✓ npm found"
echo ""

# Create necessary directories
print_message "${YELLOW}" "Creating necessary directories..."
mkdir -p backend/db
mkdir -p logs
print_message "${GREEN}" "✓ Directories created"
echo ""

# Backend setup
print_message "${YELLOW}" "Setting up backend..."
cd backend

# Sync dependencies with uv (creates venv, installs deps from pyproject.toml)
print_message "${YELLOW}" "Syncing Python dependencies with uv..."
uv sync
print_message "${GREEN}" "✓ Backend dependencies synced"

# Check if database needs to be populated
if [ ! -f "db/fantasanremo.db" ]; then
    print_message "${YELLOW}" "Database not found. Populating database..."
    uv run python populate_db.py
    print_message "${GREEN}" "✓ Database populated"
else
    print_message "${GREEN}" "✓ Database exists"
fi

cd ..
echo ""

# Start backend in background
print_message "${YELLOW}" "Starting backend server..."
cd backend
nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Save PID for later cleanup
echo $BACKEND_PID > .backend.pid
print_message "${GREEN}" "✓ Backend started (PID: $BACKEND_PID)"
echo ""

# Wait for backend to be ready
print_message "${YELLOW}" "Waiting for backend to be ready..."
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    print_message "${GREEN}" "✓ Backend is healthy"
else
    print_message "${RED}" "✗ Backend health check failed"
    cat logs/backend.log
    exit 1
fi
echo ""

# Frontend setup
print_message "${YELLOW}" "Setting up frontend..."
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    print_message "${YELLOW}" "Installing frontend dependencies..."
    npm install --silent
    print_message "${GREEN}" "✓ Frontend dependencies installed"
else
    print_message "${GREEN}" "✓ Frontend dependencies exist"
fi

cd ..
echo ""

# Start frontend in background
print_message "${YELLOW}" "Starting frontend server..."
cd frontend
nohup npm run dev -- --host > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Save PID for later cleanup
echo $FRONTEND_PID > .frontend.pid
print_message "${GREEN}" "✓ Frontend started (PID: $FRONTEND_PID)"
echo ""

# Wait for frontend to be ready
print_message "${YELLOW}" "Waiting for frontend to be ready..."
sleep 3

# Print success message
echo ""
print_message "${GREEN}" "=========================================="
print_message "${GREEN}" "Development environment is ready!"
print_message "${GREEN}" "=========================================="
echo ""
print_message "${BLUE}" "Backend:"
print_message "${GREEN}" "  URL:  http://localhost:8000"
print_message "${GREEN}" "  API:  http://localhost:8000/docs"
print_message "${GREEN}" "  Log:  logs/backend.log"
echo ""
print_message "${BLUE}" "Frontend:"
print_message "${GREEN}" "  URL:  http://localhost:5173"
print_message "${GREEN}" "  Log:  logs/frontend.log"
echo ""
print_message "${YELLOW}" "To stop the servers, run: ./scripts/stop-dev.sh"
print_message "${YELLOW}" "To view logs, run: tail -f logs/backend.log or logs/frontend.log"
echo ""

# Monitor logs for a few seconds
print_message "${YELLOW}" "Monitoring startup logs (Ctrl+C to exit monitoring)..."
sleep 2

# Show tail of logs
echo -e "\n${BLUE}=== Backend Log ===${NC}"
tail -n 10 logs/backend.log
echo ""
echo -e "${BLUE}=== Frontend Log ===${NC}"
tail -n 10 logs/frontend.log
echo ""

print_message "${GREEN}" "Servers are running in background. Use ./scripts/stop-dev.sh to stop."
