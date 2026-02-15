# FantaSanremo Team Builder 2026

Applicazione full-stack per analisi artisti, storico performance e costruzione squadra FantaSanremo con predizioni ML.

## Cosa Include

- Frontend React + TypeScript + Vite + Tailwind
- Backend FastAPI (Python 3.11+)
- Dataset e pipeline dati/ML
- Team Builder con validazione budget/capitano
- Modalita deploy Vercel standalone (frontend statico + snapshot dati)

## Struttura Progetto

```text
fantasanremo_team_builder/
├── frontend/                 # UI React
├── backend/                  # API FastAPI + ML
├── data/                     # sorgenti dati JSON
├── db/                       # database locale SQLite
├── docs/                     # documentazione tecnica
└── scripts/                  # utility di build/deploy/pipeline
```

## Avvio Locale

### Opzione rapida

```bash
./scripts/start-dev.sh
```

### Opzione manuale

```bash
# Backend
cd backend
uv sync
uv run python populate_db.py
uv run uvicorn main:app --reload --port 8000

# Frontend (nuovo terminale)
cd frontend
npm install
npm run dev
```

Endpoint locali:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Docs API: `http://localhost:8000/docs`

## Deploy Vercel

Questo repository supporta una modalita standalone: genera un frontend statico che legge uno snapshot locale (`VITE_API_MODE=local`), quindi non richiede deploy del backend su Vercel.

### 1) Prepara bundle standalone

```bash
bash scripts/prepare_vercel_bundle.sh
```

Output: cartella `vercel_app/` pronta al deploy.

### 2) Deploy su Vercel

```bash
bash scripts/deploy_vercel_standalone.sh
```

Il deploy script:

- usa `~/.codex/skills/vercel-deploy/scripts/deploy.sh` se disponibile
- altrimenti usa la CLI ufficiale `vercel` se installata

In alternativa puoi importare direttamente la cartella `vercel_app/` su Vercel.

## Deploy Full-Stack

- Frontend: Vercel (cartella `frontend`), variabile `VITE_API_URL=https://<tuo-backend>`
- Backend: Render/Railway/Fly/Docker
- CORS backend: includere dominio Vercel in `CORS_ORIGINS`

## Sicurezza e Secrets

### Regole

- Non committare `.env` reali
- Usare `.env.example` come template
- Tenere segreti solo su environment variables del provider

### Check rapido leakage

```bash
rg -n "(AKIA|ASIA|ghp_|github_pat_|xoxb-|sk-|BEGIN .*PRIVATE KEY)" -S \
  --glob '!**/node_modules/**' --glob '!**/*.lock'
```

## Pubblicazione GitHub con History Pulita (1 Commit)

Per pubblicare una versione pubblica senza cronologia precedente usa:

```bash
bash scripts/create_public_release_repo.sh /tmp/fantasanremo_public
```

Lo script:

- copia il progetto senza `.git` e artefatti locali
- inizializza un nuovo repository git
- crea una singola commit iniziale pronta per push su una nuova repo GitHub

Quando mi fornisci la repo GitHub, facciamo push della copia pulita.

## Comandi Utili

```bash
# build frontend standard
cd frontend && npm run build

# build frontend standalone
cd frontend && npm run build:standalone

# test frontend
cd frontend && npm run test:run

# lint frontend
cd frontend && npm run lint

# pipeline dati
uv run python scripts/run_pipeline.py
```

## Documentazione

- `docs/operations/DEPLOYMENT.md`
- `docs/frontend/TESTING.md`
- `docs/frontend/DESIGN_SYSTEM.md`
- `docs/ml/ML_STATUS_REPORT.md`
- `docs/ml/notebooks/README.md`
