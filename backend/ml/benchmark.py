"""Benchmark helpers for regression + category classification models."""

from __future__ import annotations

from datetime import datetime, timezone

REGRESSION_MODEL_KEYS = (
    "random_forest",
    "gradient_boosting",
    "ridge",
    "xgboost",
    "lightgbm",
)


def _to_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick_best_regressor(metrics: dict) -> tuple[str | None, dict]:
    candidates: list[tuple[str, float]] = []
    for key in REGRESSION_MODEL_KEYS:
        payload = metrics.get(key)
        if not isinstance(payload, dict):
            continue
        mae = _to_float(payload.get("mae_cv"))
        if mae == mae:  # not NaN
            candidates.append((key, mae))
    if not candidates:
        return None, {}
    best_name, _best_mae = min(candidates, key=lambda item: item[1])
    return best_name, metrics.get(best_name, {})


def summarize_metrics(metrics: dict, *, label: str = "run") -> dict:
    """Build a compact benchmark summary from training metrics."""
    best_model_name, best_payload = _pick_best_regressor(metrics)
    if isinstance(metrics.get("ensemble"), dict):
        ensemble_payload = metrics.get("ensemble", {})
    else:
        ensemble_payload = {}
    category_payload = (
        metrics.get("category_classifier", {})
        if isinstance(metrics.get("category_classifier"), dict)
        else {}
    )
    category_diagnostics = metrics.get("category_classifier_diagnostics", {})
    if isinstance(category_payload.get("diagnostics"), dict):
        category_diagnostics = category_payload["diagnostics"]

    return {
        "label": label,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "regression": {
            "best_model": best_model_name,
            "best_mae_cv": _to_float(best_payload.get("mae_cv")),
            "best_r2": _to_float(best_payload.get("r2")),
            "best_rmse": _to_float(best_payload.get("rmse")),
            "best_mape": _to_float(best_payload.get("mape")),
            "ensemble_rmse": _to_float(ensemble_payload.get("rmse")),
            "ensemble_r2": _to_float(ensemble_payload.get("r2")),
            "ensemble_mape": _to_float(ensemble_payload.get("mape")),
            "ensemble_range_macro_f1": _to_float(ensemble_payload.get("category_macro_f1")),
        },
        "classification": {
            "model": category_payload.get("model_name"),
            "macro_f1": _to_float(category_payload.get("category_macro_f1")),
            "balanced_accuracy": _to_float(category_payload.get("category_balanced_accuracy")),
            "accuracy": _to_float(category_payload.get("category_accuracy")),
            "features": int(category_payload.get("features", 0))
            if category_payload.get("features") is not None
            else 0,
            "diagnostics": category_diagnostics,
        },
        "selected_features_count": len(metrics.get("selected_features", [])),
        "category_threshold_strategy": metrics.get("category_threshold_strategy"),
        "target_source_distribution": metrics.get("data_stats", {}).get(
            "target_source_distribution", {}
        ),
        "quality_checks": metrics.get("quality_checks"),
        "best_params": metrics.get("best_params"),
    }


def evaluate_go_no_go(
    baseline: dict,
    candidate: dict,
    *,
    mae_improvement_required: float = 0.03,
    rmse_improvement_required: float = 0.05,
    macro_f1_delta_required: float = 0.04,
    max_balanced_acc_drop: float = 0.02,
) -> dict:
    """
    Evaluate candidate model against baseline.

    Gate rules:
      1. Regression improvement: MAE improves by >= 3% OR ensemble RMSE improves by >= 5%.
      2. Classification improvement: macro-F1 improves by >= +0.04.
      3. Stability guard: balanced accuracy does not drop by more than 0.02.
    """
    base_reg = baseline.get("regression", {})
    cand_reg = candidate.get("regression", {})
    base_cls = baseline.get("classification", {})
    cand_cls = candidate.get("classification", {})

    base_mae = _to_float(base_reg.get("best_mae_cv"))
    cand_mae = _to_float(cand_reg.get("best_mae_cv"))
    base_rmse = _to_float(base_reg.get("ensemble_rmse"))
    cand_rmse = _to_float(cand_reg.get("ensemble_rmse"))
    base_f1 = _to_float(base_cls.get("macro_f1"))
    cand_f1 = _to_float(cand_cls.get("macro_f1"))
    base_bal = _to_float(base_cls.get("balanced_accuracy"))
    cand_bal = _to_float(cand_cls.get("balanced_accuracy"))

    mae_rel_improvement = (
        (base_mae - cand_mae) / base_mae if base_mae == base_mae and base_mae != 0 else float("nan")
    )
    rmse_rel_improvement = (
        (base_rmse - cand_rmse) / base_rmse
        if base_rmse == base_rmse and base_rmse != 0
        else float("nan")
    )
    macro_f1_delta = (
        cand_f1 - base_f1 if cand_f1 == cand_f1 and base_f1 == base_f1 else float("nan")
    )
    balanced_acc_delta = (
        cand_bal - base_bal if cand_bal == cand_bal and base_bal == base_bal else float("nan")
    )

    regression_pass = (
        mae_rel_improvement == mae_rel_improvement
        and mae_rel_improvement >= mae_improvement_required
    ) or (
        rmse_rel_improvement == rmse_rel_improvement
        and rmse_rel_improvement >= rmse_improvement_required
    )
    classification_pass = (
        macro_f1_delta == macro_f1_delta and macro_f1_delta >= macro_f1_delta_required
    )
    stability_pass = (
        balanced_acc_delta == balanced_acc_delta and balanced_acc_delta >= -max_balanced_acc_drop
    )

    approved = regression_pass and classification_pass and stability_pass
    return {
        "approved": approved,
        "rules": {
            "regression_pass": regression_pass,
            "classification_pass": classification_pass,
            "stability_pass": stability_pass,
        },
        "deltas": {
            "mae_relative_improvement": mae_rel_improvement,
            "rmse_relative_improvement": rmse_rel_improvement,
            "macro_f1_delta": macro_f1_delta,
            "balanced_accuracy_delta": balanced_acc_delta,
        },
        "thresholds": {
            "mae_improvement_required": mae_improvement_required,
            "rmse_improvement_required": rmse_improvement_required,
            "macro_f1_delta_required": macro_f1_delta_required,
            "max_balanced_acc_drop": max_balanced_acc_drop,
        },
    }
