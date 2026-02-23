# Quick Start - ML Notebooks FantaSanremo 2026

## Prerequisiti

```bash
uv --version
cd backend && uv sync
```

## Avvio

```bash
# dalla root del progetto
uv run jupyter lab
```

Apri `backend/ml/notebooks/` nel browser.

## Ordine consigliato

1. `exploratory_data_analysis.ipynb`
2. `feature_engineering.ipynb`
3. `model_training.ipynb`

## Output attesi da `model_training.ipynb`

- metriche CV regressione (MAE, R2, RMSE, MAPE, directional accuracy)
- metriche categoria (accuracy, macro-F1, balanced accuracy)
- confronto soglie categoria (`quantile_33_66`, `quantile_30_70`, `fixed_default`)
- confronto modelli classificatori candidati
- pesi ensemble regressione
- analisi artifact (`ensemble_meta.json`, benchmark JSON)

## Esecuzione pipeline produzione

```bash
# training + prediction artifacts
uv run python scripts/run_pipeline.py --ml-training

# benchmark con confronto baseline/candidate
uv run python scripts/ml_benchmark.py \
  --label candidate \
  --baseline backend/ml/models/benchmark_baseline.json \
  --output backend/ml/models/benchmark_candidate_run.json
```

## Sperimentazione consigliata

1. prova in notebook
2. porta il cambiamento in `backend/ml/`
3. riesegui training + benchmark
4. valida con test ML

## Troubleshooting

- `ModuleNotFoundError`: avvia Jupyter dalla root progetto
- `FileNotFoundError`: verifica file in `data/`
- classificatore categoria non disponibile: usa fallback range-score e controlla diagnostics nel metadata
