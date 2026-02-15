"""Utilities for score-range based categorization."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

SCORE_LABELS = ("LOW", "MEDIUM", "HIGH")
DEFAULT_THRESHOLDS = {"low_max": 180.0, "medium_max": 270.0}


def derive_score_thresholds(
    scores: Iterable[float],
    low_quantile: float = 0.33,
    high_quantile: float = 0.66,
) -> dict[str, float]:
    """Derive LOW/MEDIUM/HIGH score thresholds from historical targets."""
    arr = np.asarray(list(scores), dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size < 3:
        return DEFAULT_THRESHOLDS.copy()

    low_max = float(np.quantile(arr, low_quantile))
    medium_max = float(np.quantile(arr, high_quantile))

    # Guard against degenerate quantiles on tiny/noisy samples.
    if not np.isfinite(low_max) or not np.isfinite(medium_max) or low_max >= medium_max:
        return DEFAULT_THRESHOLDS.copy()

    return {"low_max": low_max, "medium_max": medium_max}


def categorize_score(score: float, thresholds: dict[str, float] | None) -> str:
    """Map a numeric score to LOW/MEDIUM/HIGH."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    low_max = float(thresholds.get("low_max", DEFAULT_THRESHOLDS["low_max"]))
    medium_max = float(thresholds.get("medium_max", DEFAULT_THRESHOLDS["medium_max"]))

    if score <= low_max:
        return "LOW"
    if score <= medium_max:
        return "MEDIUM"
    return "HIGH"


def categorize_scores(scores: Iterable[float], thresholds: dict[str, float] | None) -> list[str]:
    """Vectorized helper returning category label for each score."""
    return [categorize_score(float(score), thresholds) for score in scores]
