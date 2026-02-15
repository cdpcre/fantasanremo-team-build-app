# FantaSanremo 2026 - ML Notebooks Summary

## Architettura

I notebook consumano la pipeline di produzione.

```
train.py (single source of truth)
  |
  +-- train_models()
  +-- remove_redundant_features()
  +-- select_features_adaptive()
  +-- select_classifier_features()
  +-- run_training_quality_checks()
```

## Notebook

### 1. `exploratory_data_analysis.ipynb`

Scopo:

- verificare qualita, coverage e distribuzioni dataset
- analizzare provenance target (`real` vs `estimated_*`)
- confermare readiness train/val split temporale

### 2. `feature_engineering.ipynb`

Scopo:

- analizzare feature candidate e correlazioni
- visualizzare feature selezionate per regressione/classificazione
- verificare impatto winsorization/imputation

### 3. `model_training.ipynb`

Scopo:

- eseguire training end-to-end
- confrontare metriche regressione e classificazione
- leggere benchmark baseline/candidate
- ispezionare metadata (`ensemble_meta.json`)

Modelli correnti:

- regressione: RF, GB, Ridge, XGB, LGBM
- classificazione: RF/GB/LogReg/LGBM

## Validazione

- training: 2020, 2021, 2022, 2024
- validation: 2023, 2025
- prediction target: 2026
- cross-validation: `LeaveOneGroupOut` su anno (quando applicabile)

## Benchmark workflow consigliato

```bash
uv run python scripts/run_pipeline.py --ml-training
uv run python scripts/ml_benchmark.py \
  --label candidate \
  --baseline backend/ml/models/benchmark_baseline.json \
  --output backend/ml/models/benchmark_candidate_run.json
```

## Note operative

- se il classificatore dedicato fallisce, prediction usa fallback range score
- controllare sempre `quality_checks` e `classification.diagnostics` nei JSON di output
- notebook = ambiente analisi; codice definitivo in `backend/ml/`
