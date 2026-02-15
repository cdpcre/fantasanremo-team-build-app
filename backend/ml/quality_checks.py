"""Lightweight data quality checks for ML training."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _safe_float(value: float | int | np.floating | None, default: float = 0.0) -> float:
    """Return a finite float, falling back to default."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _select_drift_features(
    train_df: pd.DataFrame,
    feature_cols: Iterable[str],
    top_n: int = 10,
) -> list[str]:
    """Pick numeric features most associated with the target for drift checks."""
    candidates: list[tuple[str, float]] = []

    if "punteggio_reale" not in train_df.columns:
        return []

    for col in feature_cols:
        if col not in train_df.columns:
            continue
        series = train_df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        non_null = series.dropna()
        if len(non_null) < 8:
            continue
        if non_null.std() == 0:
            continue

        target = train_df.loc[series.notna(), "punteggio_reale"]
        aligned = series.dropna()
        if len(target) != len(aligned):
            target = target.loc[aligned.index]
        if len(target) < 8:
            continue

        corr = aligned.corr(target)
        score = abs(_safe_float(corr, 0.0))
        candidates.append((col, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in candidates[:top_n]]


def _compute_drift_scores(train_df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """
    Compute simple cross-year drift score for each feature.

    Drift score:
      (max yearly mean - min yearly mean) / global std
    """
    if not features:
        return {}

    if "anno" not in train_df.columns:
        return {}

    scores: dict[str, float] = {}
    grouped = train_df.groupby("anno", sort=True)

    for feature in features:
        if feature not in train_df.columns:
            continue
        series = train_df[feature]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        global_std = _safe_float(series.std(ddof=0), 0.0)
        if global_std <= 1e-9:
            continue

        year_means = grouped[feature].mean().dropna()
        if len(year_means) < 2:
            continue

        spread = float(year_means.max() - year_means.min())
        scores[feature] = spread / global_std

    return scores


def run_training_quality_checks(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    training_years: list[int],
    *,
    max_target_missing_ratio: float = 0.01,
    max_feature_missing_ratio: float = 0.60,
    drift_warning_threshold: float = 2.5,
    drift_critical_threshold: float = 8.0,
    max_estimated_target_ratio_warning: float = 0.70,
) -> dict:
    """
    Run lightweight quality checks before ML training.

    Returns a dict with:
      - status: "pass" | "fail"
      - failed_checks: list[str]
      - warnings: list[str]
      - summary: dict
    """
    failed_checks: list[str] = []
    warnings: list[str] = []

    if train_df.empty:
        return {
            "status": "fail",
            "failed_checks": ["training_frame_empty"],
            "warnings": [],
            "summary": {},
        }

    # 1. Duplicate artist-year rows.
    duplicate_count = 0
    if {"artista_id", "anno"}.issubset(train_df.columns):
        duplicate_count = int(train_df.duplicated(subset=["artista_id", "anno"]).sum())
        if duplicate_count > 0:
            failed_checks.append(f"duplicate_artist_year_rows:{duplicate_count}")

    # 2. Missing target ratio.
    target_missing_ratio = 1.0
    if "punteggio_reale" not in train_df.columns:
        failed_checks.append("missing_target_column")
    else:
        target_missing_ratio = _safe_float(train_df["punteggio_reale"].isna().mean(), 1.0)
        if target_missing_ratio > max_target_missing_ratio:
            failed_checks.append(
                f"target_missing_ratio:{target_missing_ratio:.3f}>{max_target_missing_ratio:.3f}"
            )

    # 3. Training-year coverage.
    present_years = sorted(int(y) for y in train_df["anno"].dropna().unique())
    missing_years = sorted(set(training_years) - set(present_years))
    if missing_years:
        failed_checks.append(f"missing_training_years:{','.join(map(str, missing_years))}")

    # 4. Missingness on feature space.
    existing_feature_cols = [c for c in feature_cols if c in train_df.columns]
    feature_missing_mean = 0.0
    feature_missing_max = 0.0
    high_missing_features: list[str] = []
    if existing_feature_cols:
        missing_rates = train_df[existing_feature_cols].isna().mean()
        feature_missing_mean = _safe_float(missing_rates.mean(), 0.0)
        feature_missing_max = _safe_float(missing_rates.max(), 0.0)
        if feature_missing_max > max_feature_missing_ratio:
            failed_checks.append(
                f"feature_missing_ratio_max:{feature_missing_max:.3f}>{max_feature_missing_ratio:.3f}"
            )
        high_missing_features = missing_rates[missing_rates > 0.25].index.tolist()
        if high_missing_features:
            warnings.append("high_missing_features:" + ",".join(sorted(high_missing_features)[:10]))

    # 5. Drift by year on top numeric features.
    drift_features = _select_drift_features(train_df, existing_feature_cols, top_n=10)
    drift_scores = _compute_drift_scores(train_df, drift_features)
    max_drift_feature = None
    max_drift_score = 0.0
    if drift_scores:
        max_drift_feature = max(drift_scores, key=drift_scores.get)
        max_drift_score = _safe_float(drift_scores[max_drift_feature], 0.0)
        if max_drift_score > drift_critical_threshold:
            failed_checks.append(
                f"critical_drift:{max_drift_feature}={max_drift_score:.2f}>{drift_critical_threshold:.2f}"
            )
        elif max_drift_score > drift_warning_threshold:
            warnings.append(
                f"drift_warning:{max_drift_feature}={max_drift_score:.2f}>{drift_warning_threshold:.2f}"
            )

    # 6. Target provenance audit.
    source_distribution: dict[str, int] = {}
    estimated_ratio = 0.0
    if "punteggio_source" in train_df.columns:
        source_distribution = (
            train_df["punteggio_source"].fillna("missing").value_counts().to_dict()
        )
        total = max(int(len(train_df)), 1)
        estimated_count = sum(
            count for key, count in source_distribution.items() if str(key).startswith("estimated_")
        )
        estimated_ratio = estimated_count / total
        if estimated_ratio > max_estimated_target_ratio_warning:
            warnings.append(
                "estimated_target_ratio:"
                f"{estimated_ratio:.3f}>{max_estimated_target_ratio_warning:.3f}"
            )

    summary = {
        "rows": int(len(train_df)),
        "artists": int(train_df["artista_id"].nunique()) if "artista_id" in train_df.columns else 0,
        "years": present_years,
        "duplicate_artist_year_rows": duplicate_count,
        "target_missing_ratio": target_missing_ratio,
        "feature_missing_ratio_mean": feature_missing_mean,
        "feature_missing_ratio_max": feature_missing_max,
        "drift_features_checked": drift_features,
        "max_drift_feature": max_drift_feature,
        "max_drift_score": max_drift_score,
        "target_source_distribution": source_distribution,
        "estimated_target_ratio": estimated_ratio,
    }

    status = "fail" if failed_checks else "pass"
    return {
        "status": status,
        "failed_checks": failed_checks,
        "warnings": warnings,
        "summary": summary,
    }
