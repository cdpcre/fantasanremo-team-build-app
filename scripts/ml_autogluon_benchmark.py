#!/usr/bin/env python
"""Run an isolated AutoGluon benchmark on the same Fantasanremo training split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml.feature_builder import FeatureBuilder
from backend.ml.score_categories import categorize_scores, derive_score_thresholds


def _prepare_train_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    np.ndarray,
    list[str],
    dict,
]:
    builder = FeatureBuilder()
    sources = builder.load_sources()
    full_df = builder.build_training_frame(sources)
    splits = builder.split_by_years(
        full_df, builder.config.training_years, builder.config.validation_years
    )
    train_df = splits["train"].copy()
    if train_df.empty:
        raise RuntimeError("No training data available for AutoGluon benchmark.")

    feature_cols = builder.get_feature_columns(train_df)
    X = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(float(X[col].median()))
        else:
            X[col] = X[col].fillna("missing")

    y = train_df["punteggio_reale"].astype(float)
    groups = train_df["anno"].values
    thresholds = derive_score_thresholds(y.values, 0.30, 0.70)
    y_labels = np.array(categorize_scores(y.values, thresholds))
    return X, y, y_labels, groups, feature_cols, thresholds


def run_autogluon_cv(
    *,
    presets: str,
    time_limit: int,
) -> dict:
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:  # pragma: no cover - runtime dependency gate
        raise RuntimeError(f"AutoGluon is not available in this environment: {exc}") from exc

    X, y, y_labels, groups, feature_cols, thresholds = _prepare_train_data()

    logo = LeaveOneGroupOut()
    reg_oof = np.zeros(len(X), dtype=float)
    cls_oof = np.empty(len(X), dtype=object)

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        print(f"[AutoGluon] Fold {fold_idx}: train={len(train_idx)} test={len(test_idx)}")
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx].values
        cls_train = y_labels[train_idx]

        reg_train = X_train.copy()
        reg_train["_target"] = y_train

        cls_train_df = X_train.copy()
        cls_train_df["_target"] = cls_train

        with TemporaryDirectory(prefix="ag_reg_") as reg_path:
            reg_predictor = TabularPredictor(
                label="_target",
                problem_type="regression",
                eval_metric="mae",
                path=reg_path,
                verbosity=0,
            )
            reg_predictor.fit(
                reg_train,
                presets=presets,
                time_limit=time_limit,
            )
            reg_fold_pred = reg_predictor.predict(X_test)
            reg_oof[test_idx] = reg_fold_pred.to_numpy(dtype=float)

        with TemporaryDirectory(prefix="ag_cls_") as cls_path:
            cls_predictor = TabularPredictor(
                label="_target",
                problem_type="multiclass",
                eval_metric="f1_macro",
                path=cls_path,
                verbosity=0,
            )
            cls_predictor.fit(
                cls_train_df,
                presets=presets,
                time_limit=time_limit,
            )
            cls_fold_pred = cls_predictor.predict(X_test)
            cls_oof[test_idx] = cls_fold_pred.astype(str).to_numpy()

    reg_mae = float(mean_absolute_error(y.values, reg_oof))
    reg_rmse = float(np.sqrt(mean_squared_error(y.values, reg_oof)))
    cls_macro_f1 = float(f1_score(y_labels, cls_oof, average="macro", zero_division=0))
    cls_bal_acc = float(balanced_accuracy_score(y_labels, cls_oof))
    cls_acc = float(accuracy_score(y_labels, cls_oof))

    return {
        "regression": {
            "mae_oof": reg_mae,
            "rmse_oof": reg_rmse,
        },
        "classification": {
            "macro_f1_oof": cls_macro_f1,
            "balanced_accuracy_oof": cls_bal_acc,
            "accuracy_oof": cls_acc,
            "thresholds": thresholds,
        },
        "features": len(feature_cols),
        "samples": int(len(X)),
        "groups": [int(x) for x in sorted(np.unique(groups).tolist())],
        "presets": presets,
        "time_limit": int(time_limit),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AutoGluon CV benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/ml/models/autogluon_benchmark.json"),
    )
    parser.add_argument("--presets", default="medium_quality_faster_train")
    parser.add_argument("--time-limit", type=int, default=30)
    args = parser.parse_args()

    result = run_autogluon_cv(presets=args.presets, time_limit=args.time_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"AutoGluon benchmark written to: {args.output}")
    print(
        "Regression MAE="
        f"{result['regression']['mae_oof']:.3f}, "
        f"Classification macro-F1={result['classification']['macro_f1_oof']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
