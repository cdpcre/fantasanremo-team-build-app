import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from database import Base, engine
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from routers import artisti, predizioni, storico, team

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set log levels for specific modules
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("fastapi").setLevel(logging.INFO)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

# Create application logger
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize database on startup without import-time side effects."""
    auto_create = os.getenv("FS_AUTO_CREATE_DB", "true").lower() in {"1", "true", "yes"}
    if not auto_create:
        fs_auto_create = os.getenv("FS_AUTO_CREATE_DB")
        logger.info("Database auto-create disabled (FS_AUTO_CREATE_DB=%s)", fs_auto_create)
        yield
        return
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
    yield


app = FastAPI(
    title="FantaSanremo Team Builder API",
    description="API per FantaSanremo 2026 - Artisti, Storico, Predizioni ML",
    version="1.0.0",
    lifespan=lifespan,
)

logger.info("Initializing FantaSanremo Team Builder API v1.0.0")

# CORS configuration - restrictive and secure
# Origins are read from environment variable for flexibility across environments
# Only specific methods and headers are allowed to reduce attack surface
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # Specific origins only (no wildcards)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Only required methods
    allow_headers=["Content-Type", "Authorization"],  # Only required headers
)
logger.info(f"CORS middleware configured with {len(cors_origins)} allowed origins: {cors_origins}")

# Include routers
app.include_router(artisti.router, prefix="/api/artisti", tags=["artisti"])
app.include_router(storico.router, prefix="/api/storico", tags=["storico"])
app.include_router(predizioni.router, prefix="/api/predizioni", tags=["predizioni"])
app.include_router(team.router, prefix="/api/team", tags=["team"])
logger.info("Routers registered: artisti, storico, predizioni, team")


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing information"""
    start_time = datetime.now()

    # Log incoming request
    logger.info(
        f"Incoming request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        },
    )

    try:
        response = await call_next(request)

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} - "
            f"Status: {response.status_code}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            },
        )

        return response
    except Exception as e:
        # Log errors
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(
            f"Request failed: {request.method} {request.url.path} - Error: {str(e)}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round(duration * 1000, 2),
                "error": str(e),
            },
            exc_info=True,
        )
        raise


@app.get("/")
async def root():
    logger.debug("Root endpoint accessed")
    return {"message": "FantaSanremo Team Builder API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint accessed")
    return {"status": "healthy"}
