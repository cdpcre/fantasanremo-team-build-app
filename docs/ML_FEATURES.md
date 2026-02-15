# ML Feature Engineering Documentation

## Overview

La pipeline feature e costruita da `FeatureBuilder` (`backend/ml/feature_builder.py`) e alimenta due task:

1. regressione punteggio (`punteggio_predetto`)
2. classificazione categoria (`LOW/MEDIUM/HIGH`)

Le feature nascono da una base comune (50+ colonne candidate) e poi vengono selezionate in due spazi distinti:

- regressore: selezione adattiva (~21 feature nel run corrente)
- classificatore: selezione mutual information (~27 feature nel run corrente)

## Data Sources

Feature costruite combinando:

- `data/artisti_2026_enriched.json`
- `data/storico_fantasanremo_unified.json`
- `data/classifiche_finali.json`
- `data/voti_stampa.json`
- `data/regolamento_2026.json`

## Feature Families

### 1. Historical / time-aware

Esempi:

- `avg_position`
- `position_variance`
- `position_trend`
- `participations`
- `best_position`
- `recent_avg`
- `top10_finishes`
- `consistency_score`
- `momentum_score`
- `years_since_last`

Origine: `backend/ml/data_preparation.py` + arricchimenti in `FeatureBuilder`.

### 2. Biographical

Esempi:

- `artist_age`
- `career_length`
- `is_veteran`
- `is_debuttante`
- coorti generazionali (`gen_z`, `millennial`, `gen_x`, `boomer`)

Origine: `backend/ml/biographical_features.py`.

### 3. Genre

Esempi:

- `genre_avg_performance`
- `genre_trend`
- `genre_mainstream_pop`
- `genre_rap_urban`
- `genre_rock_indie`

Origine: `backend/ml/genre_features.py`.

### 4. Caratteristiche / social / bonus history

Esempi:

- `viral_potential`
- `social_followers_total`
- `has_bonus_history`
- `bonus_count`

Origine: `backend/ml/caratteristiche_features.py`.

### 5. Regulatory

Esempi:

- `has_ad_personam_bonus`
- `ad_personam_bonus_count`
- `ad_personam_bonus_points`

Origine: `backend/ml/regulatory_features.py`.

### 6. Archetypes

One-hot degli archetipi (es. `VETERAN_PERFORMER`, `INDIE_DARLING`, `POP_MAINSTREAM`).

Origine: `backend/ml/categorization.py`.

### 7. Interaction features

Feature derivate tra segnali base per catturare effetti non lineari.

Esempi:

- `age_viral_interaction`
- `experience_pop_mainstream`
- `bonus_experience_interaction`

Origine: `backend/ml/interaction_features.py`.

## Feature Processing Pipeline

### Common preprocessing

1. sostituzione `inf/-inf` con `NaN`
2. imputazione mediana per numeriche
3. rimozione feature altamente correlate (`corr > 0.95`)
4. winsorization (`q02-q98`) su subset selezionato

### Regressor feature selection

- funzione: `select_features_adaptive()`
- criterio: massimo feature proporzionale ai campioni (ratio ~5:1)
- metodo: `SelectKBest(f_regression)`

### Classifier feature selection

- funzione: `select_classifier_features()`
- metodo: `SelectKBest(mutual_info_classif)`
- controlli su minimo/massimo feature e robustezza fallback

## Persisted Metadata

Per garantire coerenza train/inference, in `backend/ml/models/ensemble_meta.json` vengono salvati:

- `selected_features`
- `feature_fill_values`
- `winsorization_bounds`
- `category_classifier_features`
- `category_classifier_feature_fill_values`
- `category_classifier_winsorization_bounds`
- `score_thresholds` + strategia scelta

## Feature Inspection Commands

```bash
# ispeziona colonne feature candidate
uv run python -c "
from backend.ml.feature_builder import FeatureBuilder
fb = FeatureBuilder()
sources = fb.load_sources()
df = fb.build_training_frame(sources)
cols = fb.get_feature_columns(df)
print('rows:', len(df))
print('features:', len(cols))
print(cols)
"

# ispeziona feature effettivamente usate dal run corrente
uv run python -c "
import json
with open('backend/ml/models/ensemble_meta.json') as f:
    meta = json.load(f)
print('regression features:', len(meta.get('selected_features', [])))
print('classifier features:', len(meta.get('category_classifier_features', [])))
"
```

## Notes

- Il classificatore dedicato puo usare un set feature diverso dal regressore.
- La categoria finale in prediction usa il classificatore quando disponibile; altrimenti fallback su range punteggio.
- La qualita dei target (reale vs stimato) influenza direttamente la qualita della selezione feature.

## References

- `backend/ml/feature_builder.py`
- `backend/ml/train.py`
- `backend/ml/data_preparation.py`
- `backend/ml/interaction_features.py`
- `backend/ml/models/ensemble_meta.json`
