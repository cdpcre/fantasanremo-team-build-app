# Validation Strategy Documentation

## Overview

La pipeline ML usa validazione temporale e due obiettivi distinti:

1. **Regressione**: previsione del punteggio FantaSanremo 2026.
2. **Classificazione**: previsione categoria `LOW` / `MEDIUM` / `HIGH`.

La separazione temporale evita leakage e rende il risultato piu realistico per il caso d'uso 2026.

## Time-Based Split

### Training set

- Anni: `2020, 2021, 2022, 2024`
- Campioni attuali: `108`
- Artisti unici: `97`

### Validation set

- `2023`: check intermedio
- `2025`: check recente

### Prediction set

- `2026`: roster corrente (30 artisti)

### Biographical completeness policy (roster 2026)

- Obiettivo operativo: `anno_nascita` valorizzato per tutti gli artisti del roster corrente.
- Per entita gruppo senza anno affidabile da fonti web, fallback su eta media roster corrente.
- Valore applicato al caso residuo (`Bambole di Pezza`): anno medio `1986` (29 artisti noti, eta media 2026 = 40).
- Persistenza fallback in `data/overrides.json` per evitare regressioni ai refresh successivi.

## Data Quality Gates

Prima del training viene eseguito `run_training_quality_checks()` (`backend/ml/quality_checks.py`).

Blocchi hard-fail:

- righe duplicate `artista_id + anno`
- missing target sopra soglia
- anni di training mancanti
- missingness massima delle feature sopra soglia
- drift critico per feature numeriche top

Warning non bloccanti:

- feature con missingness elevata
- drift moderato
- quota elevata di target stimati (`estimated_target_ratio`)

Esempio warning corrente:

- `estimated_target_ratio: 0.889 > 0.700`

Distribuzione sorgente target corrente:

- `estimated_from_classifiche: 96`
- `real: 12`

## Feature Processing Chain

Per evitare mismatch train/inference la catena e condivisa tra training e prediction.

1. costruzione dataset con `FeatureBuilder`
2. imputazione mediana per feature numeriche
3. rimozione feature ridondanti:
   - category chain: corr > `0.95`
   - regression chain: corr > `0.98`
4. selezione adattiva regressore (`SelectKBest`, ratio ~5:1)
5. winsorization per feature numeriche (`q02-q98`)
6. salvataggio metadata in `backend/ml/models/ensemble_meta.json`

## Category Chain (Dedicated Classifier)

La categoria non dipende solo dal range statico.

### Selezione soglie category

Vengono testate 3 strategie:

- `quantile_33_66`
- `quantile_30_70`
- `fixed_default`

La strategia con macro-F1 migliore viene usata nel run.

Strategia attuale selezionata: `quantile_30_70`.

In aggiunta, la feature selection del classificatore usa `mutual_info_classif` con
seed candidati (`6`, `42`) e sceglie automaticamente il seed con metrica migliore.

### Modelli classificatori candidati

- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `LogisticRegression` (con `StandardScaler`)
- `LGBMClassifier` (se disponibile)

Nel run corrente il migliore e `rf_classifier`.

## Regression Models

Regressori principali:

- Random Forest
- Gradient Boosting
- Ridge
- XGBoost (se dipendenza disponibile)
- LightGBM (se dipendenza disponibile)

I pesi di ensemble vengono calcolati via inverse-MAE sui validation year.
La strategia finale di ensemble (`weighted`/`mean`/`median`) viene scelta su OOF.

## Metrics

### Regressione

- MAE CV
- R2
- RMSE
- MAPE
- Directional Accuracy

### Classificazione

- Accuracy
- Macro-F1
- Balanced Accuracy

### Benchmark Gate (Go/No-Go)

Implementato in `backend/ml/benchmark.py`:

- regressione: MAE relativo >= 3% **oppure** RMSE ensemble >= 5%
- classificazione: macro-F1 delta >= `+0.04`
- stabilita: balanced accuracy non cala oltre `-0.02`

Risultato candidate-v4 vs baseline-v1:

- `macro_f1_delta = +0.0643` (pass)
- `balanced_accuracy_delta = +0.0686` (pass)
- `mae_relative_improvement = +3.41%` (pass regressione)
- `rmse_relative_improvement = +0.60%` (non sufficiente da sola, ma non necessaria)
- esito finale: `GO`

## Leakage Prevention

- split temporale per anno
- `LeaveOneGroupOut` su anno in CV/tuning quando possibile
- feature storiche calcolate con logica `as_of_year`
- metadata di fill/winsorization riusati in prediction

## Prediction Outputs

Output principale (`backend/ml/predict.py`):

- `punteggio_predetto`
- `score_category` / `livello_performer`
- `category_confidence` (se classifier supporta `predict_proba`)
- `confidence` (agreement tra regressori)
- intervalli di incertezza (conformal se calibratore presente, altrimenti fallback)

## Commands

```bash
# training + prediction artifacts
uv run python scripts/run_pipeline.py --ml-training

# benchmark candidate e confronto con baseline
uv run python scripts/ml_benchmark.py \
  --label candidate-v4 \
  --baseline backend/ml/models/benchmark_baseline.json \
  --output backend/ml/models/benchmark_candidate_run_v4.json

# re-check baseline dopo refresh dati
uv run python scripts/ml_benchmark.py \
  --label baseline-v2-refresh-after-group-avg-age \
  --baseline backend/ml/models/benchmark_baseline.json \
  --output backend/ml/models/benchmark_baseline_v2_refresh.json \
  --skip-save-models
```

## References

- `backend/ml/train.py`
- `backend/ml/predict.py`
- `backend/ml/benchmark.py`
- `backend/ml/quality_checks.py`
- `backend/ml/models/ensemble_meta.json`
- `backend/ml/models/benchmark_candidate_run_v4.json`
