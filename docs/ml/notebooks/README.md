# ML Notebooks - FantaSanremo 2026

Notebook Jupyter per analisi dati, feature engineering e training inspection.

Principio chiave: `backend/ml/train.py` e la source of truth. I notebook non devono duplicare logica di produzione.

## Notebook Disponibili

### 1. `exploratory_data_analysis.ipynb`

Obiettivi:

- qualita e copertura dati
- distribuzioni principali
- verifica readiness del dataset ML

Input principali:

- `artisti_2026_enriched.json`
- `storico_fantasanremo_unified.json`
- `classifiche_finali.json`
- `voti_stampa.json`

### 2. `feature_engineering.ipynb`

Obiettivi:

- esplorazione feature candidate
- controllo correlazioni e ridondanze
- confronto feature selection regressore vs classificatore

Funzioni usate dal codice produzione:

- `remove_redundant_features()`
- `select_features_adaptive()`
- `select_classifier_features()`

### 3. `model_training.ipynb`

Obiettivi:

- esecuzione `train_models()`
- analisi metriche regressione/classificazione
- inspection pesi ensemble
- analisi benchmark baseline/candidate

Pipeline modellistica corrente:

- regressori: RF, GB, Ridge, XGB, LGBM
- classificatore categoria dedicato: RF, GB, LogReg, LGBM
- output: score 2026 + categoria `LOW/MEDIUM/HIGH`

## Architettura Consigliata

```text
Notebook (esplorazione) -> train.py (produzione) -> artifact benchmark/predict
```

## Avvio Rapido

```bash
# dalla root del progetto
uv run jupyter lab
```

Aprire `backend/ml/notebooks/`.

## Buone Pratiche

- usare i notebook per analisi, non per logica business definitiva
- promuovere in `FeatureBuilder` e `train.py` solo modifiche validate
- salvare benchmark in `backend/ml/models/benchmark_*.json`

## Troubleshooting

- `ModuleNotFoundError`: avviare Jupyter dalla root progetto
- `FileNotFoundError`: verificare i file in `data/`
- training lento: GridSearchCV e cross-validation grouped sono costosi su CPU
