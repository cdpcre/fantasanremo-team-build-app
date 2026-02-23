# FantaSanremo Team Builder - Deployment Guide

Complete deployment guide for development, staging, and production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Environment Variables](#environment-variables)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

**For Local Development:**
- Python 3.11+
- Node.js 18+
- npm or yarn

**For Docker Deployment:**
- Docker 20.10+
- Docker Compose 2.0+

**For Cloud Deployment:**
- Cloud provider account (AWS, GCP, Azure, etc.)
- CLI tools for your chosen provider

## Local Development

### Quick Start

The easiest way to start development:

```bash
# Start everything
./scripts/start-dev.sh

# Stop everything
./scripts/stop-dev.sh
```

### Manual Setup

**Backend Setup:**

```bash
cd backend

# Sync dependencies (creates venv, installs from pyproject.toml)
uv sync

# Initialize database (first time only)
uv run python populate_db.py

# Start server
uv run uvicorn main:app --reload --port 8000
```

**Frontend Setup:**

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Access the application:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Docker Deployment

### Development with Docker

For development with hot reload:

```bash
# Start development containers
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop containers
docker-compose -f docker-compose.dev.yml down
```

### Production Deployment

**Automated deployment:**

```bash
# Run the production deployment script
./scripts/start-prod.sh
```

**Manual deployment:**

```bash
# Build and start containers
docker-compose up -d --build

# Check container status
docker-compose ps

# View logs
docker-compose logs -f
```

**Access the application:**
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Docker Commands Reference

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d backend

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View logs
docker-compose logs -f [service_name]

# Execute command in container
docker-compose exec backend python -c "print('Hello')"

# Rebuild specific service
docker-compose build backend

# Restart service
docker-compose restart backend

# View resource usage
docker stats
```

## Cloud Deployment

### Deployment Options

#### 1. Full Stack Deployment (Recommended)

Deploy both frontend and backend together using Docker.

**AWS ECS/Fargate:**
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name fantasanremo

# Build and push images to ECR
# (See AWS documentation for detailed setup)
```

**Google Cloud Run:**
```bash
# Build and deploy
gcloud run deploy fantasanremo-backend \
  --source ./backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

gcloud run deploy fantasanremo-frontend \
  --source ./frontend \
  --platform managed \
  --region us-central1 \
  --set-env-vars VITE_API_URL=<backend-url>
```

**Azure Container Instances:**
```bash
# Create resource group
az group create --name fantasanremo-rg --location eastus

# Create container
az container create \
  --resource-group fantasanremo-rg \
  --name fantasanremo \
  --image your-registry/fantasanremo:latest \
  --ports 80 8000
```

#### 2. Separate Frontend and Backend

**Frontend on Vercel:**

1. Connect GitHub repository
2. Configure build settings:
   - Root directory: `frontend`
   - Build command: `npm run build`
   - Output directory: `dist`
3. Add environment variable:
   - `VITE_API_URL`: Your backend URL

**Backend on Railway/Render/Fly.io:**

1. Connect GitHub repository
2. Configure:
   - Root directory: Project root
   - Build context: `.` (use Dockerfile.backend)
3. Add environment variables (see below)

#### 3. Serverless Deployment

**Backend on AWS Lambda:**
- Use AWS Serverless Application Model (SAM)
- Deploy with API Gateway
- Configure RDS or DynamoDB for database

**Frontend on AWS S3 + CloudFront:**
- Build frontend: `npm run build`
- Upload to S3 bucket
- Configure CloudFront distribution

## Environment Variables

### Required Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

**Backend Variables:**

```bash
# Database
DATABASE_URL=sqlite:///./db/fantasanremo.db

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://yourdomain.com

# API
API_BASE_PATH=/api

# Application
APP_NAME=FantaSanremo Team Builder
APP_VERSION=1.0.0
ENVIRONMENT=production
```

**Frontend Variables:**

```bash
# API URL (for production)
VITE_API_URL=https://api.yourdomain.com
```

### Production Environment Variables

```bash
# Backend
DATABASE_URL=postgresql://user:pass@host:5432/dbname
CORS_ORIGINS=https://yourdomain.com
ENVIRONMENT=production
LOG_LEVEL=INFO

# Frontend
VITE_API_URL=https://api.yourdomain.com
```

## Monitoring and Logging

### Health Checks

**Backend health check:**
```bash
curl http://localhost:8000/health
```

**Frontend health check:**
```bash
curl http://localhost/
```

### Logging

**Development logs:**
```bash
# Backend
tail -f logs/backend.log

# Frontend
tail -f logs/frontend.log
```

**Docker logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Monitoring Tools

**Application Monitoring:**
- Sentry for error tracking
- Datadog or New Relic for APM
- Prometheus + Grafana for metrics

**Log Aggregation:**
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Splunk
- CloudWatch Logs (AWS)
- Cloud Logging (GCP)

## Troubleshooting

### Common Issues

**1. Port already in use**

```bash
# Find process using port
lsof -i:8000
lsof -i:5173

# Kill process
kill -9 <PID>

# Or use with Docker
docker-compose down
```

**2. Database not found**

```bash
# Re-populate database
cd backend
python populate_db.py
```

**3. CORS errors**

Check your CORS origins in `.env`:
```bash
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

**4. Docker build fails**

```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

**5. Frontend can't reach backend**

- Check if backend is running: `curl http://localhost:8000/health`
- Verify CORS settings
- Check proxy configuration in `vite.config.ts`
- Verify API URL in frontend environment

### Debug Mode

**Enable debug logging:**

```bash
# Backend
export LOG_LEVEL=DEBUG

# Frontend
# Check browser console for detailed logs
```

**Run backend in debug mode:**

```bash
uvicorn main:app --reload --log-level debug
```

### Performance Issues

**Backend optimization:**
```bash
# Use multiple workers
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000

# Use gunicorn with uvicorn workers
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

**Frontend optimization:**
- Enable code splitting (already configured)
- Use lazy loading for routes
- Optimize images and assets
- Enable production build optimizations

## Security Considerations

### Production Security Checklist

- [ ] Change default passwords
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Enable security headers
- [ ] Regular dependency updates
- [ ] Implement backup strategy
- [ ] Set up monitoring and alerts
- [ ] Configure WAF (Web Application Firewall)

### SSL/TLS Setup

**Using nginx with Let's Encrypt:**

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal (configured automatically)
sudo certbot renew --dry-run
```

## Backup and Recovery

### Database Backup

```bash
# Backup SQLite database
cp backend/db/fantasanremo.db backups/fantasanremo-$(date +%Y%m%d).db

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp backend/db/fantasanremo.db backups/fantasanremo-$DATE.db
find backups/ -name "fantasanremo-*.db" -mtime +7 -delete
```

### Recovery

```bash
# Restore from backup
cp backups/fantasanremo-20240101.db backend/db/fantasanremo.db

# Restart services
docker-compose restart backend
```

## Scaling Considerations

### Horizontal Scaling

**Backend:**
- Use load balancer (nginx, ALB, etc.)
- Run multiple instances
- Use PostgreSQL for shared database
- Implement Redis for caching

**Frontend:**
- Deploy to CDN (CloudFront, Cloudflare)
- Use static site hosting (S3, Netlify, Vercel)
- Enable CDN caching

### Vertical Scaling

**Increase resources:**
```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

## Support

For issues and questions:
- Check the troubleshooting section above
- Review logs for error messages
- Check GitHub Issues
- Contact the development team

## Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Documentation](https://docs.docker.com/)
- [React Deployment](https://vitejs.dev/guide/build.html)
- [nginx Configuration](https://nginx.org/en/docs/)
