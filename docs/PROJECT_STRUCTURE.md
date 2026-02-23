# Project Structure

## Core Directories

### `backend/`

FastAPI backend con API, data pipeline e ML.

#### `backend/data_pipeline/`

- `config.py`: configurazione ambiente
- `pipeline.py`: orchestrazione fetch/validate/transform/load/report
- `sources/`: connettori sorgenti
- `validators/`: schema, business rules, data quality
- `storage/`: persistenza su DB

#### `backend/ml/`

Pipeline ML per previsione punteggio e categoria.

- `feature_builder.py`: entry point unico feature engineering
- `data_preparation.py`: dataset storico/time-aware e target provenance
- `quality_checks.py`: gate qualita prima del training
- `train.py`: training regressori + classificatore dedicato
- `predict.py`: inferenza 2026 (score + category + uncertainty)
- `benchmark.py`: summary metriche e regole go/no-go
- `score_categories.py`: utilities soglie/range LOW-MEDIUM-HIGH
- `interaction_features.py`: feature interazione
- `models/`: artifact `.pkl` + metadata `.json`

#### `backend/routers/`

- endpoint API principali (`predizioni`, `team`, ecc.)

#### `backend/tests/`

- test API, ML e quality checks

### `frontend/`

App React + TypeScript + Vite.

#### `frontend/src/`

- `api/`: client API
- `components/`: componenti UI
- `pages/`: pagine applicative
- `types/`: tipi condivisi
- `utils/`: utility frontend

### `data/`

Dataset canonici e file di supporto:

- `artisti_2026_enriched.json`
- `storico_fantasanremo_unified.json`
- `classifiche_finali.json`
- `voti_stampa.json`
- `regolamento_2026.json`

### `scripts/`

- `run_pipeline.py`: CLI pipeline dati + training ML
- `ml_benchmark.py`: confronto baseline/candidate
- script update/sync dati

### `docs/`

Documentazione progetto.

- `docs/DATA_PIPELINE.md`
- `docs/VALIDATION_STRATEGY.md`
- `docs/ML_FEATURES.md`
- `docs/ARTIST_CATEGORIZATION.md`
- `docs/ml/ML_STATUS_REPORT.md`
- `docs/ml/notebooks/` (README, quickstart, summary)
- `docs/frontend/` (guide UI/testing)
- `docs/testing/` (test backend)
- `docs/operations/` (deployment/docker)

## ML Artifact Layout

Artifact principali in `backend/ml/models/`:

- regressori: `rf_model.pkl`, `gb_model.pkl`, `ridge_model.pkl`, `xgb_model.pkl`, `lgbm_model.pkl`
- classificatore: `category_classifier_model.pkl`
- metadata: `ensemble_meta.json`
- output predizioni: `predictions_2026.json`
- benchmark: `benchmark_*.json`

## Quick Navigation

- training ML: `backend/ml/train.py`
- prediction runtime: `backend/ml/predict.py`
- benchmark gates: `backend/ml/benchmark.py`
- quality chain: `backend/ml/quality_checks.py`
- feature chain: `backend/ml/feature_builder.py`
