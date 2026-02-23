# ML Improvements: Literature + Experiments

## Goal

Migliorare la pipeline su dataset tabulare piccolo (108 campioni train effettivi) con split temporale per anno.

## Letteratura Consultata

1. Grinsztajn et al. (NeurIPS 2022), *Why do tree-based models still outperform deep learning on tabular data?*  
   https://openreview.net/forum?id=Fp7__phQszn
2. Chen & Guestrin (KDD 2016), *XGBoost: A Scalable Tree Boosting System*  
   https://dl.acm.org/doi/10.1145/2939672.2939785
3. Ke et al. (NeurIPS 2017), *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*  
   https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
4. Prokhorenkova et al. (NeurIPS 2018), *CatBoost: unbiased boosting with categorical features*  
   https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac4124863285549-Abstract.html
5. Scikit-learn docs, `mutual_info_classif` (stima MI non parametrica, con `random_state`)  
   https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html

## Razionale Applicato al Progetto

- Modelli tree-based come backbone per small-tabular: confermati come baseline più robusta.
- Tuning più mirato su XGBoost per ridurre MAE senza cambiare la chain dati.
- Inserimento `LGBMClassifier` tra i candidati categoria (oltre a RF/GB/LogReg).
- Feature selection categoria resa deterministica e ripetibile:
  - seed candidati `6` e `42`
  - scelta automatica del seed migliore via CV.
- Scelta automatica strategia soglie (`q33_66`, `q30_70`, `fixed`) mantenuta.
- Ensemble regressione valutato su più strategie (`weighted`, `mean`, `median`) con selezione su OOF.
- Micro-ciclo successivo dedicato solo alla regressione:
  - category chain lasciata invariata
  - regression chain resa più conservativa sul pruning correlazione (`corr>0.98`)
  - random search XGBoost focalizzata su LOGO-CV per massimizzare MAE improvement.

## Esperimenti Eseguiti

- sweep randomizzato iperparametri regressori (RF/ETR/XGB/LGBM) su LOGO-CV
- sweep randomizzato classificatori (RF/ETR/GB/LogReg/XGB/LGBM) su LOGO-CV
- confronto strategie soglia categoria
- confronto seed feature-selection MI
- confronto ensemble regressione (`weighted`/`mean`/`median`)

## Esito Integrato

### candidate-v3

Sorgente: `backend/ml/models/benchmark_candidate_run_v3.json`

- regressione (best MAE CV): `0.5595` (baseline: `0.5693`)
- classificazione macro-F1: `0.5264` (baseline: `0.4622`)
- delta macro-F1: `+0.0643`
- delta balanced accuracy: `+0.0686`
- esito gate: `NO-GO` (blocco sulla regressione)

### candidate-v4 (micro-ciclo regressione)

Sorgente: `backend/ml/models/benchmark_candidate_run_v4.json`

- regressione (best MAE CV): `0.5499` (baseline: `0.5693`)
- delta MAE relativo: `+3.41%` (sopra soglia gate +3%)
- RMSE best regressor: `83.6349`
- classificazione invariata rispetto a v3 (macro-F1 `0.5264`)
- esito gate: `GO`

### candidate-v5 (source-weights + robust-loss + calibration/threshold + CatBoost)

Sorgente: `backend/ml/models/benchmark_candidate_run_v5.json`

- regressione (best MAE CV): `0.5732` (baseline: `0.5693`, v4: `0.5499`)
- RMSE best regressor: `84.7167` (v4: `83.6349`)
- classificazione macro-F1: `0.5875` (baseline: `0.4622`, v4: `0.5264`)
- delta macro-F1 vs baseline: `+0.1253`
- delta balanced accuracy vs baseline: `+0.1191`
- esito gate: `NO-GO` (blocco regressione)

### candidate-v6-hybrid (regressione v4 + categoria v5)

Sorgente: `backend/ml/models/benchmark_candidate_run_v6_hybrid.json`

- regressione (best MAE CV): `0.5499` (identica a v4)
- RMSE best regressor: `83.6349` (identica a v4)
- classificazione macro-F1: `0.5875` (vs v4 `0.5264`, delta `+0.0611`)
- balanced accuracy: `0.5931` (vs v4 `0.5426`)
- strategia soglie categoria selezionata: `fixed_default`
- esito pratico: configurazione ibrida consigliata per produzione

### AutoGluon benchmark (isolato, no TabPFN)

Sorgente: `backend/ml/models/autogluon_benchmark_v1.json`

- configurazione: `presets=medium_quality_faster_train`, `time_limit=30s` per fold
- split: LOGO su anni train (`2020, 2021, 2022, 2024`)
- regressione OOF: `MAE=99.02`, `RMSE=117.94` (scala reale)
- classificazione OOF: `macro-F1=0.3170`, `balanced_accuracy=0.3232`
- conclusione: in questa configurazione AutoGluon sottoperforma nettamente la pipeline attuale.

## Note

Il ciclo v3 ha sbloccato la classificazione; il ciclo v4 ha poi sbloccato anche la regressione.
Il ciclo v6-hybrid combina i due vantaggi: regressione stabile (v4) + categoria più forte (v5).
Il collo di bottiglia residuo resta la qualità target (`real` vs `estimated`), non la capacità del modello.
