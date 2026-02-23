# FantaSanremo Team Builder 2026

[Live Demo on Vercel](https://fantasanremo-team-build-app.vercel.app/)

Full-stack application for artist analysis, performance history, and FantaSanremo team building with ML predictions.

## Features

- Frontend: React + TypeScript + Vite + Tailwind
- Backend: FastAPI (Python 3.11+)
- Datasets and Data/ML pipelines
- Team Builder with budget and captain validation
- Standalone deploy mode for Vercel (static frontend + data snapshot)

## Project Structure

```text
fantasanremo_team_builder/
├── frontend/                 # React UI
├── backend/                  # FastAPI API + ML
├── data/                     # JSON data sources
├── db/                       # Local SQLite database
├── docs/                     # Technical documentation
└── scripts/                  # Build/deploy/pipeline utilities
```

## Local Setup

### Quick Start

```bash
./scripts/start-dev.sh
```

### Manual Setup

```bash
# Backend
cd backend
uv sync
uv run python populate_db.py
uv run uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Local endpoints:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## Vercel Deploy

This repository supports a standalone mode: it generates a static frontend that reads a local snapshot (`VITE_API_MODE=local`), thus not requiring the backend to be deployed on Vercel.

### 1) Prepare standalone bundle

```bash
bash scripts/prepare_vercel_bundle.sh
```

Output: `vercel_app/` folder ready for deployment.

### 2) Deploy to Vercel

```bash
bash scripts/deploy_vercel_standalone.sh
```

The deploy script:

- uses `~/.codex/skills/vercel-deploy/scripts/deploy.sh` if available
- otherwise uses the official `vercel` CLI if installed

Alternatively, you can directly import the `vercel_app/` folder into Vercel.

## Full-Stack Deployment

- Frontend: Vercel (`frontend` folder), variable `VITE_API_URL=https://<your-backend>`
- Backend: Render/Railway/Fly/Docker
- Backend CORS: include the Vercel domain in `CORS_ORIGINS`

## Security and Secrets

### Rules

- Do not commit real `.env` files
- Use `.env.example` as a template
- Keep secrets only in provider environment variables

### Quick leakage check

```bash
rg -n "(AKIA|ASIA|ghp_|github_pat_|xoxb-|sk-|BEGIN .*PRIVATE KEY)" -S \
  --glob '!**/node_modules/**' --glob '!**/*.lock'
```


## Useful Commands

```bash
# standard frontend build
cd frontend && npm run build

# standalone frontend build
cd frontend && npm run build:standalone

# run frontend tests
cd frontend && npm run test:run

# frontend linting
cd frontend && npm run lint

# data pipeline
uv run python scripts/run_pipeline.py
```

## Documentation

- `docs/operations/DEPLOYMENT.md`
- `docs/frontend/TESTING.md`
- `docs/frontend/DESIGN_SYSTEM.md`
- `docs/ml/ML_STATUS_REPORT.md`
- `docs/ml/notebooks/README.md`
