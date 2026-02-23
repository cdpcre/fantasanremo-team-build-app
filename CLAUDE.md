# Fantasanremo Team Builder - Claude Guidelines

## Project Overview
Fantasanremo fantasy game team builder application with enhanced data pipeline and ML 
feature engineering. The system uses historical data from 2020-2025 to predict artist 
performance for 2026, with 50+ engineered features across multiple categories.

## Development Guidelines

### Python Environment
- This project uses **uv** as the package manager
- Always use `uv run` or `uv run python` to execute Python scripts
- Do NOT use `python` directly

### Command Examples
```bash
# Run Python scripts
uv run python scripts/update_fantasanremo_data.py

# Run backend server
cd backend && uv run uvicorn main:app --reload

# Install dependencies
cd backend && uv sync

# Run data pipeline
uv run python scripts/run_pipeline.py

# Run ML training
uv run python scripts/run_pipeline.py --ml-training
```

### Git Workflow

#### Pre-commit Hook
A pre-commit hook automatically runs Ruff linter before each commit. The hook will:
- Check for linting issues with `ruff check`
- Verify code formatting with `ruff format --check`
- Block the commit if issues are found

#### Manual Linter Commands
```bash
# Check for issues
uv run ruff check backend/

# Auto-fix issues
uv run ruff check --fix backend/

# Format code
uv run ruff format backend/

# Check specific file
uv run ruff check backend/ml/train.py
```

#### Ruff Configuration
- Line length: 100 characters
- Target Python: 3.10+
- Rules enabled: E, F, I, N, W, UP
- Ignores: E402 (sys.path imports), E741 (ML vars), N803/N806 (ML naming)

See `pyproject.toml` for full configuration.

## Working with Subagents

When working with subagents or AI assistants on this project, provide them with this context:

### Essential Background
1. **Project Type:** Full-stack Fantasy Sanremo team builder with ML predictions
2. **Tech Stack:**
   - Backend: Python, FastAPI, SQLite
   - Frontend: React + Vite, TypeScript, Tailwind CSS
   - ML: scikit-learn, XGBoost, LightGBM, 50+ engineered features (21 selected for regression)
   - Package Manager: uv (NOT pip or npm for Python)
3. **Data:** Historical artist data 2020-2025, predicting 2026 performance

### Critical Guidelines for Subagents
- **Always use `uv run`** for Python commands (never `python` or `pip` directly)
- **Run linter before committing:** `uv run ruff check --fix backend/ && uv run ruff format backend/`
- **ML Features:** 50+ features across 7 categories (see `docs/ml/`)
- **Validation:** Time-based splits (2020-2022+2024 train, 2023/2025 val, 2026 pred)
- **Documentation:** Check `docs/` directory for domain knowledge

### Common Subagent Tasks
- **Feature Implementation:** Read relevant docs in `docs/ml/` and `docs/data/`
- **Bug Fixes:** Check `docs/testing/` for test strategies
- **Frontend Work:** Review `docs/frontend/` for patterns and error handling
- **ML Work:** Study `docs/ML_FEATURES.md` and validation strategy
- **Deployment:** Follow `docs/operations/DEPLOYMENT.md` and `DOCKER_COMMANDS.md`

## Specialized Documentation

### Core Documentation
- **`docs/PROJECT_STRUCTURE.md`** - Complete project structure and file descriptions
- **`docs/DATA_PIPELINE.md`** - Data ingestion, validation, and processing

### Machine Learning (docs/ml/)
- **`docs/ML_FEATURES.md`** - Complete ML feature engineering documentation (50+ features)
- **`docs/ml/ML_STATUS_REPORT.md`** - ML pipeline status and next steps
- **`docs/ml/DEBUTTANTI_FEATURES.md`** - Debuttanti (new artist) proxy features
- **`docs/ml/notebooks/README.md`** - Jupyter notebooks overview
- **`docs/ml/notebooks/NOTEBOOKS_SUMMARY.md`** - Detailed notebook descriptions
- **`docs/ml/notebooks/QUICKSTART.md`** - Quick start guide for ML notebooks

### Validation & Strategy
- **`docs/VALIDATION_STRATEGY.md`** - Time-based validation approach (train/val/test splits)
- **`docs/ARTIST_CATEGORIZATION.md`** - 7 artist archetypes (Viral, Veteran, Indie, etc.)

### Frontend (docs/frontend/)
- **`docs/frontend/TESTING.md`** - Frontend testing strategies and patterns
- **`docs/frontend/ERROR_HANDLING_GUIDE.md`** - Error handling best practices

### Operations (docs/operations/)
- **`docs/operations/DEPLOYMENT.md`** - Deployment procedures and environments
- **`docs/operations/DOCKER_COMMANDS.md`** - Docker commands and container management

### Testing & Data
- **`docs/testing/BACKEND_TESTS.md`** - Backend testing documentation
- **`docs/data/FANTASANREMO_DATA.md`** - Fantasanremo data source documentation

### Key Concepts

#### ML Training Architecture
The training pipeline uses `FeatureBuilder` (in `backend/ml/feature_builder.py`) as the single
entry point for all feature engineering:

```
FeatureBuilder.load_sources() → build_training_frame() → split_by_years() → train
```

- `build_training_frame()` calls `_attach_static_features()` which integrates ALL 50+ features
  (biographical, genre, caratteristiche, regulatory, categorization, debuttanti)
- `get_feature_columns()` dynamically discovers feature columns from the built DataFrame
- `train.py:train_models()` uses `FeatureBuilder` directly — no legacy loaders
- **Ensemble**: 5 models (RF + GB + Ridge + XGBoost + LightGBM) with dynamic inverse-MAE weights
- **Category head**: dedicated LOW/MEDIUM/HIGH classifier with threshold-strategy selection
- **Feature selection**: Adaptive 5:1 samples-to-features ratio (21 features from 50+)
- **Target normalization**: Z-score per year to handle inter-year score variability
- **Hyperparameter tuning**: GridSearchCV with LeaveOneGroupOut when n_samples >= 50
- **Training data**: 108 samples from `classifiche_finali.json` (all ranked artists 2020-2025)
- **Quality audit**: target provenance tracked (`real` vs `estimated_*`) with warning thresholds

#### ML Features (50+ available, ~21 selected)
1. **Historical** (15) - Performance history, consistency, momentum
2. **Genre** (7) - Music genre-based performance
3. **Characteristics** (6) - Charisma, stage presence, viralità
4. **Regulatory** (12) - Bonus/malus potential by sponsor
5. **Biographical** (10) - Age, experience, generational cohorts
6. **Archetypes** (7) - Artist categories
7. **Debuttanti** (11) - Proxy features for new artists
8. **Voto stampa** (1) - Press vote feature

Feature selection is automatic: `SelectKBest(f_regression)` with 5:1 ratio reduces 50+ to ~21.

#### Validation Strategy
- **Training:** 2020, 2021, 2022, 2024 — **108 samples** from classifiche_finali.json
- **Validation 1:** 2023 (~28 artists, mid-period check)
- **Validation 2:** 2025 (~29 artists, recent performance check)
- **Prediction:** 2026 (30 artists)
- **Cross-validation:** LeaveOneGroupOut (by year) — no temporal leakage
- **Hyperparameter tuning:** GridSearchCV automatic when n_samples >= 50

Benchmark gates:
- Regression: MAE relative improvement >= 3% OR ensemble RMSE improvement >= 5%
- Classification: macro-F1 delta >= +0.04
- Stability: balanced accuracy drop not worse than -0.02

### Diagnostic Commands
```bash
# Run data pipeline (dry run, verbose)
uv run python scripts/run_pipeline.py --dry-run --verbose

# Run ML training
uv run python scripts/run_pipeline.py --ml-training

# Check feature builder output
uv run python -c "
from backend.ml.feature_builder import FeatureBuilder
fb = FeatureBuilder()
sources = fb.load_sources()
df = fb.build_training_frame(sources)
print(f'Features: {len(fb.get_feature_columns(df))}')
print(f'Rows: {len(df)}')
print(fb.get_feature_columns(df))
"

# Verify ensemble metadata
cat backend/ml/models/ensemble_meta.json | python -m json.tool

# Quick check predictions
uv run python -c "
import json
with open('backend/ml/models/predictions_2026.json') as f:
    preds = json.load(f)
scores = [p['punteggio_predetto'] for p in preds]
print(f'Predizioni: {len(preds)}, Range: {min(scores):.0f}-{max(scores):.0f}')
"

# Verify selected features and weights
uv run python -c "
import json
with open('backend/ml/models/ensemble_meta.json') as f:
    meta = json.load(f)
print(f'Features: {len(meta[\"selected_features\"])}')
print(f'Weights: {meta[\"ensemble_weights\"]}')
for f in meta['selected_features']:
    print(f'  - {f}')
"
```
