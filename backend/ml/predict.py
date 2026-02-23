import json
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_builder import FeatureBuilder
from .score_categories import (
    DEFAULT_THRESHOLDS,
    categorize_score,
    categorize_scores,
)
from .train import load_models
from .uncertainty import (
    add_uncertainty_to_predictions,
    bootstrap_prediction_intervals,
    load_calibrator,
)


def _inverse_normalize(predictions: np.ndarray, year_stats: dict) -> np.ndarray:
    """Inverse z-score normalization using global mean/std from training."""
    if not year_stats:
        return predictions
    global_mean = np.mean([s["mean"] for s in year_stats.values()])
    global_std = np.mean([s["std"] for s in year_stats.values()])
    if global_std == 0:
        global_std = 1.0
    return predictions * global_std + global_mean


def _apply_winsorization(
    X: pd.DataFrame, bounds: dict[str, dict[str, float]] | None
) -> pd.DataFrame:
    """Clip features using training-time winsorization bounds."""
    if not bounds:
        return X
    out = X.copy()
    for col, limits in bounds.items():
        if col not in out.columns:
            continue
        lower = limits.get("lower")
        upper = limits.get("upper")
        if lower is None or upper is None:
            continue
        out[col] = out[col].clip(lower=lower, upper=upper)
    return out


def predict_2026(artisti_df: pd.DataFrame, storico_df: pd.DataFrame) -> list[dict]:
    """
    Generate 2026 predictions for all artists.

    Args:
        artisti_df: DataFrame with artist data (id, nome, quotazione_2026)
        storico_df: DataFrame with historical FantaSanremo results

    Returns:
        List of dicts with keys: artista_id, punteggio_predetto, confidence, livello_performer
    """
    # Load trained models and ensemble metadata
    models, meta = load_models()

    if models is None:
        raise RuntimeError("Models not trained. Run train.py first.")

    category_model = models.get("category_classifier")
    regression_models = {
        name: model for name, model in models.items() if name != "category_classifier"
    }
    if not regression_models:
        raise RuntimeError("No regression models available. Run train.py first.")

    # Build features aligned with training
    builder = FeatureBuilder()
    sources = builder.load_sources()
    prediction_year = builder.config.prediction_year
    features_df = builder.build_prediction_frame(sources, prediction_year)

    if features_df.empty:
        return []

    # Use selected features from training if available
    if meta and meta.get("selected_features"):
        feature_cols = [c for c in meta["selected_features"] if c in features_df.columns]
    else:
        feature_cols = builder.get_feature_columns(features_df)

    X = features_df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    fill_values = meta.get("feature_fill_values", {}) if meta else {}
    for col in feature_cols:
        X[col] = X[col].fillna(fill_values.get(col, 0.0))
    winsorization_bounds = meta.get("winsorization_bounds", {}) if meta else {}
    X = _apply_winsorization(X, winsorization_bounds)

    # Load ensemble strategy and weights.
    ensemble_strategy = (
        str(meta.get("ensemble_strategy", "weighted")).lower() if meta else "weighted"
    )
    if meta and meta.get("ensemble_weights"):
        weights = {
            name: weight
            for name, weight in meta["ensemble_weights"].items()
            if name in regression_models
        }
    else:
        weights = {}

    if not weights:
        weights = {name: 1.0 / len(regression_models) for name in regression_models}
    else:
        total_weight = sum(weights.values())
        if total_weight <= 0:
            weights = {name: 1.0 / len(regression_models) for name in regression_models}
        else:
            weights = {name: weight / total_weight for name, weight in weights.items()}

    # Generate predictions using weighted ensemble
    predictions = {}
    for name, model in regression_models.items():
        predictions[name] = model.predict(X)

    prediction_matrix = np.column_stack(list(predictions.values()))
    if ensemble_strategy == "median" and prediction_matrix.shape[1] > 1:
        ensemble_pred = np.median(prediction_matrix, axis=1)
    elif ensemble_strategy == "mean" and prediction_matrix.shape[1] > 1:
        ensemble_pred = np.mean(prediction_matrix, axis=1)
    else:
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            w = weights.get(name, 0)
            ensemble_pred += w * pred

    # Inverse normalize predictions back to real scale
    year_stats = meta.get("year_stats", {}) if meta else {}
    ensemble_pred = _inverse_normalize(ensemble_pred, year_stats)

    # Clamp predictions to reasonable range
    ensemble_pred = np.clip(ensemble_pred, 50, 800)

    # Add conformal prediction intervals
    conformal_predictor = load_calibrator()
    conformal_lower = None
    conformal_upper = None

    if conformal_predictor is not None:
        # Get conformal intervals on normalized scale
        global_mean = np.mean([s["mean"] for s in year_stats.values()]) if year_stats else 0
        global_std = np.mean([s["std"] for s in year_stats.values()]) if year_stats else 1
        if global_std == 0:
            global_std = 1.0

        # Normalize predictions for conformal prediction
        pred_norm = (ensemble_pred - global_mean) / global_std
        lower_norm, upper_norm = conformal_predictor.predict(pred_norm)

        # Denormalize intervals
        conformal_lower = lower_norm * global_std + global_mean
        conformal_upper = upper_norm * global_std + global_mean

        # Clip intervals to reasonable range
        conformal_lower = np.clip(conformal_lower, 0, 800)
        conformal_upper = np.clip(conformal_upper, 50, 1000)
    else:
        # Fallback: use simple heuristic based on prediction variance
        all_preds = np.array(list(predictions.values()))
        if len(all_preds) > 1:
            denorm_preds = np.array([_inverse_normalize(pred, year_stats) for pred in all_preds])
            std_estimate = np.std(denorm_preds, axis=0)
            conformal_lower = ensemble_pred - 1.96 * std_estimate  # Approx 95% CI
            conformal_upper = ensemble_pred + 1.96 * std_estimate
        else:
            # Conservative default interval
            conformal_lower = ensemble_pred * 0.8
            conformal_upper = ensemble_pred * 1.2

    # Generate bootstrap intervals as alternative uncertainty estimate
    bootstrap_intervals = bootstrap_prediction_intervals(regression_models, X, n_bootstrap=50)

    # Calculate confidence based on prediction variance between models
    all_preds = np.array(list(predictions.values()))
    if len(all_preds) > 1:
        # Inverse normalize each model's predictions before computing variance
        denorm_preds = np.array([_inverse_normalize(pred, year_stats) for pred in all_preds])
        variance = np.std(denorm_preds, axis=0)
        max_var = variance.max() if variance.max() > 0 else 1
        confidence = 1 - (variance / max_var)
    else:
        confidence = np.full(len(X), 0.5)

    score_thresholds = (
        meta.get("score_thresholds", DEFAULT_THRESHOLDS) if meta else DEFAULT_THRESHOLDS
    )
    fallback_categories = categorize_scores(ensemble_pred, score_thresholds)
    category_predictions = fallback_categories
    category_confidence: np.ndarray | None = None

    if category_model is not None:
        try:
            category_feature_cols = meta.get("category_classifier_features", []) if meta else []
            if category_feature_cols:
                category_fill_values = (
                    meta.get("category_classifier_feature_fill_values", {}) if meta else {}
                )
                X_category = pd.DataFrame(index=features_df.index)
                for col in category_feature_cols:
                    if col in features_df.columns:
                        X_category[col] = features_df[col]
                    else:
                        X_category[col] = category_fill_values.get(col, 0.0)
                X_category = X_category.replace([np.inf, -np.inf], np.nan)
                for col in category_feature_cols:
                    X_category[col] = X_category[col].fillna(category_fill_values.get(col, 0.0))
                category_bounds = (
                    meta.get("category_classifier_winsorization_bounds", {}) if meta else {}
                )
                X_category = _apply_winsorization(X_category, category_bounds)
            else:
                X_category = X

            predicted_labels = category_model.predict(X_category)
            allowed_labels = {"LOW", "MEDIUM", "HIGH"}
            category_predictions = []
            for idx, label in enumerate(predicted_labels):
                normalized = str(label).upper()
                category_predictions.append(
                    normalized if normalized in allowed_labels else fallback_categories[idx]
                )

            if hasattr(category_model, "predict_proba"):
                probabilities = category_model.predict_proba(X_category)
                category_confidence = probabilities.max(axis=1)
        except Exception:
            category_predictions = fallback_categories
            category_confidence = None

    # Add uncertainty intervals to predictions
    results_with_uncertainty = add_uncertainty_to_predictions(
        predictions=[
            {
                "artista_id": int(features_df.iloc[i]["artista_id"]),
                "punteggio_predetto": float(np.round(ensemble_pred[i], 2)),
                "confidence": float(np.round(confidence[i], 2)),
                "livello_performer": category_predictions[i],
                "score_category": category_predictions[i],
                "category_confidence": (
                    float(np.round(category_confidence[i], 3))
                    if category_confidence is not None
                    else None
                ),
            }
            for i in range(len(features_df))
        ],
        lower_bounds=conformal_lower,
        upper_bounds=conformal_upper,
        bootstrap_intervals=bootstrap_intervals,
    )

    results = results_with_uncertainty

    return results


def generate_predictions_simple(artisti: list[dict], storico: list[dict]) -> list[dict]:
    """
    Simplified prediction function that works with raw data.
    """
    # Convert to DataFrames
    artisti_df = pd.DataFrame(artisti)
    storico_df = pd.DataFrame(storico) if storico else pd.DataFrame()

    if storico_df.empty:
        # No historical data - use quotation-based heuristic
        results = []
        thresholds = DEFAULT_THRESHOLDS
        for artista in artisti:
            quotazione = artista.get("quotazione_2026", 14)

            # Base score from quotation
            base_scores = {17: 400, 16: 350, 15: 280, 14: 220, 13: 180}

            score = float(base_scores.get(quotazione, 250))
            level = categorize_score(score, thresholds)

            results.append(
                {
                    "artista_id": artista["id"],
                    "punteggio_predetto": round(score, 2),
                    "confidence": 0.5,
                    "livello_performer": level,
                    "score_category": level,
                }
            )

        return results

    return predict_2026(artisti_df, storico_df)


def save_predictions(predictions: list[dict], output_path: Path | None = None) -> Path:
    """
    Save predictions to JSON.

    Args:
        predictions: List of prediction dicts
        output_path: Optional output file path

    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = Path(__file__).parent / "models" / "predictions_2026.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    # Test prediction
    print("Loading models and generating predictions...")

    # This would be called from the main app after data is loaded
    print("Use generate_predictions_simple() from main application")
