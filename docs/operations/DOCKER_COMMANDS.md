# Docker Quick Reference

Quick commands for working with the FantaSanremo Team Builder Docker setup.

## Development

### Start Development Environment

```bash
# Using script (recommended)
./scripts/start-dev.sh

# Using Docker Compose
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f
```

### Stop Development Environment

```bash
# Using script
./scripts/stop-dev.sh

# Using Docker Compose
docker-compose -f docker-compose.dev.yml down
```

## Production

### Deploy to Production

```bash
# Using script (recommended)
./scripts/start-prod.sh

# Using Docker Compose
docker-compose up -d --build
```

### Stop Production

```bash
docker-compose down
```

## Common Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Check Status

```bash
# Container status
docker-compose ps

# Service health
curl http://localhost:8000/health  # Backend
curl http://localhost/              # Frontend
```

### Restart Services

```bash
# All services
docker-compose restart

# Specific service
docker-compose restart backend
docker-compose restart frontend
```

### Execute Commands

```bash
# Backend shell
docker-compose exec backend bash

# Run Python command
docker-compose exec backend python populate_db.py

# Frontend shell
docker-compose exec frontend sh
```

### Rebuild Services

```bash
# Rebuild all
docker-compose build

# Rebuild specific service
docker-compose build backend
docker-compose build frontend

# Rebuild without cache
docker-compose build --no-cache backend
```

## Troubleshooting

### Clean Everything

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (deletes database!)
docker-compose down -v

# Clean Docker system
docker system prune -a
```

### View Resource Usage

```bash
docker stats
```

### Inspect Container

```bash
# Container details
docker-compose exec backend env

# Container logs
docker-compose logs backend
```

## Environment Variables

### Create .env file

```bash
cp .env.example .env
# Edit .env with your settings
```

### View Current Environment

```bash
docker-compose config
```

## Docker Compose Files

- **docker-compose.yml** - Production setup
- **docker-compose.dev.yml** - Development with hot reload

## Services

### Backend

- **Port:** 8000
- **Health Check:** http://localhost:8000/health
- **API Docs:** http://localhost:8000/docs
- **Container:** fantasanremo-backend

### Frontend

- **Port:** 80 (production), 5173 (development)
- **URL:** http://localhost
- **Container:** fantasanremo-frontend

## Useful Tips

### Automatic Restart

```bash
# Always restart on failure
docker-compose up -d --restart always
```

### Resource Limits

Edit `docker-compose.yml`:
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
```

### Network Debugging

```bash
# List networks
docker network ls

# Inspect network
docker network inspect fantasanremo-network
```

## Backup

### Backup Database

```bash
# Copy database from container
docker cp fantasanremo-backend:/app/db/fantasanremo.db ./backup/fantasanremo-$(date +%Y%m%d).db
```

### Restore Database

```bash
# Copy database to container
docker cp ./backup/fantasanremo.db fantasanremo-backend:/app/db/fantasanremo.db

# Restart backend
docker-compose restart backend
```

## Monitoring

### Real-time Monitoring

```bash
# Watch logs
docker-compose logs -f --tail=100

# Watch container stats
watch -n 1 'docker stats --no-stream'
```

### Health Checks

```bash
# Backend health
while true; do
  curl -s http://localhost:8000/health && echo " - OK" || echo " - FAILED"
  sleep 5
done
```
