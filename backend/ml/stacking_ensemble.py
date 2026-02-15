"""Stacking ensemble implementation for Fantasanremo ML predictions.

This module provides a stacking ensemble that uses a meta-learner (Ridge regression)
to combine predictions from base models, using LeaveOneGroupOut cross-validation
to prevent temporal leakage.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut


class GroupKFold:
    """Custom CV splitter that respects group labels."""

    def __init__(self, groups):
        """Initialize with group labels."""
        self.groups = groups
        self.logo = LeaveOneGroupOut()

    def split(self, X, y=None, groups=None):
        """Split data respecting group labels."""
        return self.logo.split(X, y, self.groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.logo.get_n_splits(groups=self.groups)


def train_stacking_ensemble(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
) -> StackingRegressor:
    """
    Train stacking ensemble with LeaveOneGroupOut CV to prevent temporal leakage.

    Args:
        models: Dictionary of trained base models with keys: rf, gb, ridge, xgb, lgbm
        X_train: Training feature matrix
        y_train: Training target values
        groups: Group labels for LeaveOneGroupOut CV (typically year values)
        feature_cols: List of feature column names

    Returns:
        Trained StackingRegressor model
    """
    # Define estimators for stacking
    estimators = [
        ("rf", models["rf"]),
        ("gb", models["gb"]),
        ("ridge", models["ridge"]),
        ("xgb", models["xgb"]),
        ("lgbm", models["lgbm"]),
    ]

    # Configure LeaveOneGroupOut CV for meta-learner training
    # This prevents temporal leakage by holding out entire years
    cv_splitter = GroupKFold(groups)

    # Create stacking ensemble with Ridge meta-learner
    # Ridge provides regularization to prevent overfitting on small validation sets
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=cv_splitter,
        n_jobs=1,  # Run sequentially to avoid memory issues
        verbose=0,
    )

    # Fit the stacking model
    # The CV parameter ensures base model predictions for meta-training are
    # generated using out-of-fold predictions, preventing data leakage
    stacking_model.fit(X_train, y_train)

    return stacking_model


def evaluate_stacking_model(
    stack_model: StackingRegressor, X_val: np.ndarray, y_val: np.ndarray
) -> dict[str, float]:
    """
    Evaluate stacking model on validation set.

    Args:
        stack_model: Trained StackingRegressor
        X_val: Validation feature matrix
        y_val: Validation target values

    Returns:
        Dictionary with MAE, RMSE, and R² scores
    """
    predictions = stack_model.predict(X_val)

    mae = float(mean_absolute_error(y_val, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_val, predictions)))
    r2 = float(r2_score(y_val, predictions))

    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_stacking_on_multiple_years(
    stack_model: StackingRegressor, val_frames: dict, feature_cols: list[str], year_stats: dict
) -> dict[str, dict[str, float]]:
    """
    Evaluate stacking model on multiple validation years.

    Args:
        stack_model: Trained StackingRegressor
        val_frames: Dictionary mapping years to validation DataFrames
        feature_cols: List of feature column names
        year_stats: Dictionary with mean/std for each year (for normalization)

    Returns:
        Dictionary mapping years to their evaluation metrics
    """
    results = {}

    # Calculate global normalization stats
    global_mean = np.mean([s["mean"] for s in year_stats.values()])
    global_std = np.mean([s["std"] for s in year_stats.values()])
    if global_std == 0:
        global_std = 1.0

    for year, val_df in val_frames.items():
        if val_df.empty or val_df["punteggio_reale"].isna().all():
            continue

        # Normalize validation targets
        val_y = val_df["punteggio_reale"]
        val_y_norm = (val_y - global_mean) / global_std

        # Get features
        X_val = val_df[feature_cols].fillna(0)

        if X_val.empty:
            continue

        # Evaluate
        metrics = evaluate_stacking_model(stack_model, X_val, val_y_norm)
        results[int(year)] = metrics

    return results


def save_stacking_model(stack_model: StackingRegressor, path: Path | str) -> Path:
    """
    Save stacking model to disk.

    Args:
        stack_model: Trained StackingRegressor
        path: Path to save the model (without extension)

    Returns:
        Path to saved model file
    """
    path = Path(path)
    path = path.with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(stack_model, path)
    return path


def load_stacking_model(path: Path | str) -> StackingRegressor | None:
    """
    Load stacking model from disk.

    Args:
        path: Path to the saved model file

    Returns:
        Loaded StackingRegressor or None if file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        return None

    return joblib.load(path)


def save_stacking_metadata(
    metrics: dict, selected_features: list[str], models_dir: Path | str
) -> Path:
    """
    Save stacking ensemble metadata to disk.

    Args:
        metrics: Dictionary with validation metrics for each year
        selected_features: List of selected feature names
        models_dir: Directory to save metadata

    Returns:
        Path to saved metadata file
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    meta = {"validation_metrics": metrics, "selected_features": selected_features}

    meta_path = models_dir / "stacking_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta_path


def load_stacking_metadata(models_dir: Path | str) -> dict | None:
    """
    Load stacking ensemble metadata from disk.

    Args:
        models_dir: Directory containing metadata file

    Returns:
        Metadata dictionary or None if file doesn't exist
    """
    meta_path = Path(models_dir) / "stacking_meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        return json.load(f)
