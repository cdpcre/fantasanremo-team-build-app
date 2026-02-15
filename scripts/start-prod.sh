#!/bin/bash

# FantaSanremo Team Builder - Production Deployment Script
# This script builds and deploys the application using Docker Compose

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
print_message "${BLUE}" "FantaSanremo Team Builder - Production"
print_message "${BLUE}" "=========================================="
echo ""

# Check prerequisites
print_message "${YELLOW}" "Checking prerequisites..."

if ! command_exists docker; then
    print_message "${RED}" "Error: Docker is not installed"
    print_message "${YELLOW}" "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command_exists docker-compose; then
    print_message "${RED}" "Error: Docker Compose is not installed"
    print_message "${YELLOW}" "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi

print_message "${GREEN}" "✓ Docker found"
print_message "${GREEN}" "✓ Docker Compose found"
echo ""

# Create necessary directories
print_message "${YELLOW}" "Creating necessary directories..."
mkdir -p data
mkdir -p db
mkdir -p logs
print_message "${GREEN}" "✓ Directories created"
echo ""

# Load environment variables
if [ -f .env ]; then
    print_message "${YELLOW}" "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
    print_message "${GREEN}" "✓ Environment loaded"
else
    print_message "${YELLOW}" "No .env file found, using .env.example"
    if [ -f .env.example ]; then
        cp .env.example .env
        print_message "${YELLOW}" "Created .env from .env.example - please review and update if needed"
    fi
fi
echo ""

# Stop any existing containers
print_message "${YELLOW}" "Stopping any existing containers..."
docker-compose down 2>/dev/null || true
print_message "${GREEN}" "✓ Containers stopped"
echo ""

# Build backend image
print_message "${YELLOW}" "Building backend Docker image..."
docker-compose build backend
print_message "${GREEN}" "✓ Backend image built"
echo ""

# Build frontend image
print_message "${YELLOW}" "Building frontend Docker image..."
docker-compose build frontend
print_message "${GREEN}" "✓ Frontend image built"
echo ""

# Start containers
print_message "${YELLOW}" "Starting containers..."
docker-compose up -d
print_message "${GREEN}" "✓ Containers started"
echo ""

# Wait for services to be healthy
print_message "${YELLOW}" "Waiting for services to be healthy..."
sleep 5

# Check service health
print_message "${YELLOW}" "Checking service health..."

# Check backend health
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_message "${GREEN}" "✓ Backend is healthy"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_message "${RED}" "✗ Backend health check failed"
    print_message "${YELLOW}" "Check logs with: docker-compose logs backend"
    exit 1
fi

echo ""

# Check frontend
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost/ > /dev/null 2>&1; then
        print_message "${GREEN}" "✓ Frontend is healthy"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_message "${RED}" "✗ Frontend health check failed"
    print_message "${YELLOW}" "Check logs with: docker-compose logs frontend"
    exit 1
fi

echo ""

# Print success message
echo ""
print_message "${GREEN}" "=========================================="
print_message "${GREEN}" "Deployment successful!"
print_message "${GREEN}" "=========================================="
echo ""
print_message "${BLUE}" "Application is now running:"
print_message "${GREEN}" "  Frontend:  http://localhost"
print_message "${GREEN}" "  Backend:   http://localhost:8000"
print_message "${GREEN}" "  API Docs:  http://localhost:8000/docs"
echo ""
print_message "${YELLOW}" "Useful commands:"
print_message "${YELLOW}" "  View logs:     docker-compose logs -f"
print_message "${YELLOW}" "  Stop:          docker-compose down"
print_message "${YELLOW}" "  Restart:       docker-compose restart"
print_message "${YELLOW}" "  Check status:  docker-compose ps"
echo ""
print_message "${BLUE}" "Container status:"
docker-compose ps
echo ""
print_message "${GREEN}" "=========================================="
print_message "${GREEN}" "Deployment complete!"
print_message "${GREEN}" "=========================================="
