# Fantasanremo Data Pipeline Documentation

## Overview

La data pipeline gestisce ingestione, validazione e trasformazione dati, e puo poi avviare la pipeline ML.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                            │
├─────────────────────────────────────────────────────────────────┤
│  JSON canonicali + fonti web + DB locale                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline Orchestration                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Fetch      - Load da sorgenti                               │
│  2. Validate   - Schema + business rules + quality checks       │
│  3. Transform  - Normalizzazione e coerenza cross-file          │
│  4. Load       - Upsert su SQLite                               │
│  5. Report     - Report qualita                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ML Pipeline (optional)                   │
├─────────────────────────────────────────────────────────────────┤
│  • Training: 2020/2021/2022/2024                               │
│  • Validation: 2023 + 2025                                      │
│  • Regressori: RF, GB, Ridge, XGB, LGBM                         │
│  • Classificatore dedicato LOW/MEDIUM/HIGH                      │
│  • Output: score numerico + categoria + confidence              │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Steps

### 1. Fetch Data

Sorgenti principali:

- `data/artisti_2026_enriched.json`
- `data/storico_fantasanremo_unified.json`
- `data/classifiche_finali.json`
- `data/voti_stampa.json`
- `data/regolamento_2026.json`

Implementation: `backend/data_pipeline/pipeline.py`.

### 2. Validate Data

Tre livelli:

1. schema
2. business rules
3. data quality

Implementazione:

- `backend/data_pipeline/validators/schema_validator.py`
- `backend/data_pipeline/validators/business_rules.py`
- `backend/data_pipeline/validators/data_quality.py`

### 3. Transform Data

Trasformazioni principali:

- gestione missing values
- normalizzazione/formattazione campi
- controlli di consistenza tra file

### 4. Load to Database

- upsert su SQLite
- transazioni con rollback
- log e tracciamento operazioni

Implementation: `backend/data_pipeline/storage/database_store.py`.

### 5. Generate Report

Report su:

- score qualita
- errori/warning
- copertura sorgenti

## CLI Usage

```bash
# dry run
uv run python scripts/run_pipeline.py --dry-run

# full pipeline
uv run python scripts/run_pipeline.py

# solo alcuni step
uv run python scripts/run_pipeline.py --steps fetch,validate

# solo validazione
uv run python scripts/run_pipeline.py --validate-only

# training ML + prediction artifacts
uv run python scripts/run_pipeline.py --ml-training
```

## ML Hand-off Notes

Quando si esegue `--ml-training`:

- viene chiamato `backend/ml/train.py`
- vengono salvati modelli + metadata in `backend/ml/models/`
- viene generato `predictions_2026.json`

Per benchmark baseline/candidate usare:

```bash
uv run python scripts/ml_benchmark.py --label candidate
```

## Troubleshooting

### Missing source file

Verificare presenza dei file JSON in `data/`.

### Validation failures

Eseguire con `--verbose` per dettaglio warning/errori.

### ML quality warning on target provenance

Controllare `target_source_distribution` e `estimated_target_ratio` in:

- `backend/ml/models/ensemble_meta.json`
- output benchmark JSON
