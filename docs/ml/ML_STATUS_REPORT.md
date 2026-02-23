# ML Pipeline Status Report

**Data:** 11 Febbraio 2026  
**Stato:** configurazione ibrida validata (`regressione v4` + `categoria v5`)

## Executive Summary

La pipeline ML produce due output:

1. punteggio numerico 2026 (regressione)
2. categoria `LOW/MEDIUM/HIGH` (classificatore dedicato)

La configurazione ibrida (`candidate-v6-hybrid`) mantiene invariata la regressione di `v4`
e porta in produzione solo i miglioramenti categoria di `v5` (calibrazione + soglie).

## Stato Corrente della Pipeline

### Regressione

- modelli: `rf`, `gb`, `ridge`, `xgb`, `lgbm`
- tuning XGBoost ampliato su parametri robusti per small-tabular
- ensemble dinamico con strategia scelta tra `weighted`, `mean`, `median` su OOF
- strategia selezionata nel run corrente: `weighted`

### Classificazione categoria

- modelli candidati: `rf_classifier`, `gb_classifier`, `logreg_classifier`, `lgbm_classifier`
- test su strategie soglie: `quantile_33_66`, `quantile_30_70`, `fixed_default`
- test su seed di feature selection MI (`[6, 42]`) con scelta automatica del migliore
- configurazione selezionata nel run corrente:
  - soglie: `quantile_30_70`
  - seed feature selection: `6`
  - modello: `rf_classifier`

### Qualita dato

- quality checks: `pass`
- warning: `estimated_target_ratio: 0.889`
- target source distribution: `estimated_from_classifiche=96`, `real=12`

## Benchmark Snapshot

Sorgente: `backend/ml/models/benchmark_candidate_run_v4.json`

### Candidate-v4 (baseline regressione)

- Classification macro-F1: `0.5264`
- Classification balanced accuracy: `0.5426`
- Classification accuracy: `0.5370`
- Best regressor (CV MAE): `xgboost`, `0.5499`
- Best regressor RMSE: `83.6349`
- Ensemble RMSE: `84.8913`

### Delta vs baseline-v1

- `mae_relative_improvement: +3.41%`
- `rmse_relative_improvement: +0.60%`
- `macro_f1_delta: +0.0643`
- `balanced_accuracy_delta: +0.0686`

### Gate outcome

- `classification_pass: true`
- `stability_pass: true`
- `regression_pass: true`
- **Result:** `GO`

### Candidate-v5 (esperimento esteso full-stack)

Sorgente: `backend/ml/models/benchmark_candidate_run_v5.json`

- Classification macro-F1: `0.5875` (vs v4: `0.5264`)
- Classification balanced accuracy: `0.5931` (vs v4: `0.5426`)
- Best regressor (CV MAE): `gradient_boosting`, `0.5732` (vs v4: `0.5499`)
- Best regressor RMSE: `84.7167` (vs v4: `83.6349`)
- **Result:** `NO-GO` (gate regressione non superato)

### Candidate-v6-hybrid (produzione consigliata)

Sorgente: `backend/ml/models/benchmark_candidate_run_v6_hybrid.json`

- Regressione:
  - Best model: `xgboost`
  - `best_mae_cv = 0.5499` (identico a v4)
  - `best_rmse = 83.6349` (identico a v4)
- Classificazione:
  - model: `gb_classifier`
  - `macro-F1 = 0.5875` (vs v4: `0.5264`, `+0.0611`)
  - `balanced_accuracy = 0.5931` (vs v4: `0.5426`)
- Strategia soglie categoria: `fixed_default`
- **Result:** `NO-GO` rispetto al gate standard (regressione non migliora di +3%), ma obiettivo ibrido raggiunto.

### AutoGluon (benchmark isolato)

Sorgente: `backend/ml/models/autogluon_benchmark_v1.json`

- Regressione OOF: `MAE=99.02`, `RMSE=117.94` (scala reale)
- Classificazione OOF: `macro-F1=0.3170`, `balanced_accuracy=0.3232`
- Esito: sottoperformance rispetto alla pipeline custom attuale

### Baseline re-check post data refresh (2026-02-11)

Sorgente: `backend/ml/models/benchmark_baseline_v2_refresh.json`

- Label run: `baseline-v2-refresh-after-group-avg-age`
- Confronto vs baseline corrente (`backend/ml/models/benchmark_baseline.json`):
  - `best_mae_cv`: invariato (`0.5498846980539585`)
  - `best_rmse`: invariato (`83.63494295666146`)
  - `macro_f1`: invariato (`0.5875168146354587`)
  - `balanced_accuracy`: invariato (`0.5930799220272904`)
- Gate: `NO-GO` (delta = `0` su tutte le metriche principali)
- Decisione: baseline **non** aggiornata (si mantiene `baseline-v2`)

## Artifacts prodotti

- `backend/ml/models/rf_model.pkl`
- `backend/ml/models/gb_model.pkl`
- `backend/ml/models/ridge_model.pkl`
- `backend/ml/models/xgb_model.pkl`
- `backend/ml/models/lgbm_model.pkl`
- `backend/ml/models/category_classifier_model.pkl`
- `backend/ml/models/ensemble_meta.json`
- `backend/ml/models/predictions_2026.json`
- `backend/ml/models/benchmark_candidate_run_v4.json`
- `backend/ml/models/benchmark_candidate_run_v5.json`
- `backend/ml/models/benchmark_candidate_run_v6_hybrid.json`
- `backend/ml/models/autogluon_benchmark_v1.json`

## Cosa e gia stato validato

Comandi eseguiti:

```bash
uv run python scripts/run_pipeline.py --ml-training
uv run python scripts/ml_benchmark.py --label candidate-v4 --baseline backend/ml/models/benchmark_baseline.json --output backend/ml/models/benchmark_candidate_run_v4.json
uv run python scripts/ml_benchmark.py --label candidate-v5 --baseline backend/ml/models/benchmark_baseline.json --output backend/ml/models/benchmark_candidate_run_v5.json
uv run python scripts/ml_benchmark.py --label candidate-v6-hybrid --baseline backend/ml/models/benchmark_candidate_run_v4.json --output backend/ml/models/benchmark_candidate_run_v6_hybrid.json
uv run --with autogluon.tabular python scripts/ml_autogluon_benchmark.py --output backend/ml/models/autogluon_benchmark_v1.json --time-limit 30
uv run ruff check backend/ml/train.py backend/ml/predict.py
uv run pytest backend/tests/test_ml_quality_checks.py backend/tests/test_ml_benchmark.py backend/tests/test_ml.py -q
```

Esito:

- training completato
- benchmark aggiornati (`candidate-v4` GO, `candidate-v5` NO-GO, `candidate-v6-hybrid` target ibrido centrato)
- benchmark AutoGluon isolato completato
- lint ok
- test ML: `32 passed`

## Priorita prossime iterazioni

1. mantenere `candidate-v6-hybrid` come asset operativo corrente
2. lavorare su regressione oltre v4 senza perdere i guadagni categoria (micro-cicli MAE/RMSE)
3. ridurre la quota di target stimati aumentando i target `real`

## Riferimenti

- `backend/ml/train.py`
- `backend/ml/predict.py`
- `backend/ml/benchmark.py`
- `backend/ml/quality_checks.py`
- `backend/ml/models/benchmark_candidate_run_v4.json`
- `backend/ml/models/benchmark_candidate_run_v5.json`
- `backend/ml/models/benchmark_candidate_run_v6_hybrid.json`
- `backend/ml/models/autogluon_benchmark_v1.json`
