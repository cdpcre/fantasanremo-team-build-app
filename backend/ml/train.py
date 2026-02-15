import json
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneGroupOut,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_builder import FeatureBuilder
from .quality_checks import run_training_quality_checks
from .score_categories import DEFAULT_THRESHOLDS, categorize_scores, derive_score_thresholds
from .stacking_ensemble import (
    evaluate_stacking_on_multiple_years,
    load_stacking_model,
    save_stacking_metadata,
    save_stacking_model,
    train_stacking_ensemble,
)

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ModuleNotFoundError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    CatBoostClassifier = None


SOURCE_WEIGHT_MAP: dict[str, float] = {
    "real": 1.0,
    "estimated_from_storico_position": 0.8,
    "estimated_from_classifiche": 0.55,
    "missing": 0.55,
}
DEFAULT_SOURCE_WEIGHT = 0.55


def remove_redundant_features(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Remove one feature from each pair with correlation > threshold."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        highly_correlated = upper.index[upper[col] > threshold].tolist()
        if highly_correlated:
            to_drop.add(col)
    return list(to_drop)


def select_features_adaptive(
    X: pd.DataFrame, y: pd.Series, feature_cols: list[str], ratio: int = 5
) -> list[str]:
    """Select features adaptively: max features = n_samples / ratio."""
    max_features = max(2, len(X) // ratio)
    if len(feature_cols) <= max_features:
        return feature_cols

    selector = SelectKBest(f_regression, k=max_features)
    selector.fit(X[feature_cols], y)
    mask = selector.get_support()
    return [col for col, keep in zip(feature_cols, mask) if keep]


def select_classifier_features(
    X: pd.DataFrame,
    y_labels: list[str],
    feature_cols: list[str],
    *,
    ratio: int = 4,
    min_features: int = 8,
    max_features_cap: int = 28,
    random_state: int | None = 42,
) -> list[str]:
    """Select features for category classification with mutual information."""
    if len(feature_cols) <= min_features:
        return feature_cols

    max_by_ratio = max(min_features, len(X) // ratio)
    k = min(len(feature_cols), max_features_cap, max_by_ratio)
    if k >= len(feature_cols):
        return feature_cols

    try:
        score_func = lambda values, labels: mutual_info_classif(  # noqa: E731
            values,
            labels,
            random_state=random_state,
        )
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X[feature_cols], y_labels)
        mask = selector.get_support()
        selected = [col for col, keep in zip(feature_cols, mask) if keep]
        return selected if selected else feature_cols
    except Exception:
        return feature_cols


def normalize_targets(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Z-score normalize targets by year. Returns (normalized df, stats dict)."""
    df = train_df.copy()
    year_stats = {}
    for year in df["anno"].unique():
        mask = df["anno"] == year
        scores = df.loc[mask, "punteggio_reale"]
        mean_val = float(scores.mean())
        std_val = float(scores.std())
        if std_val == 0 or np.isnan(std_val):
            std_val = 1.0
        year_stats[int(year)] = {"mean": mean_val, "std": std_val}
        df.loc[mask, "punteggio_reale"] = (scores - mean_val) / std_val
    return df, year_stats


def inverse_normalize_predictions(
    predictions: np.ndarray | pd.Series, year_stats: dict[int, dict[str, float]]
) -> np.ndarray:
    """Inverse z-score normalization using global mean/std from training."""
    preds = np.asarray(predictions, dtype=float)
    if not year_stats:
        return preds

    means = [stats["mean"] for stats in year_stats.values()]
    stds = [stats["std"] for stats in year_stats.values()]
    global_mean = float(np.mean(means)) if means else 0.0
    global_std = float(np.mean(stds)) if stds else 1.0
    if global_std == 0:
        global_std = 1.0

    return preds * global_std + global_mean


def compute_winsorization_bounds(
    X: pd.DataFrame,
    *,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
) -> dict[str, dict[str, float]]:
    """Compute per-feature clipping bounds to limit outlier impact."""
    bounds: dict[str, dict[str, float]] = {}
    for col in X.columns:
        series = X[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        lower = float(series.quantile(lower_quantile))
        upper = float(series.quantile(upper_quantile))
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue
        if lower > upper:
            continue
        bounds[col] = {"lower": lower, "upper": upper}
    return bounds


def apply_winsorization(
    X: pd.DataFrame, bounds: dict[str, dict[str, float]] | None
) -> pd.DataFrame:
    """Apply clipping bounds to a feature matrix."""
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


def calculate_enhanced_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate enhanced regression metrics beyond MAE.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary containing R², RMSE, MAPE, and directional accuracy
    """
    metrics = {}

    # R² (R-squared) - Coefficient of determination
    try:
        r2 = r2_score(y_true, y_pred)
        metrics["r2"] = float(r2)
    except ValueError:
        metrics["r2"] = float("nan")

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["rmse"] = float(rmse)

    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-10
    absolute_percentage_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
    mape = np.mean(absolute_percentage_errors) * 100  # Convert to percentage
    metrics["mape"] = float(mape)

    # Directional Accuracy
    # Percentage of times the model correctly predicts the direction (up/down)
    # relative to the mean of true values
    if len(y_true) > 1:
        true_mean = np.mean(y_true)
        true_direction = y_true > true_mean
        pred_direction = y_pred > true_mean
        directional_acc = np.mean(true_direction == pred_direction)
        metrics["directional_accuracy"] = float(directional_acc)
    else:
        metrics["directional_accuracy"] = float("nan")

    return metrics


def calculate_category_metrics(
    y_true_labels: list[str], y_pred_labels: list[str]
) -> dict[str, float]:
    """Calculate LOW/MEDIUM/HIGH classification metrics."""
    return {
        "category_accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "category_macro_f1": float(
            f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        ),
        "category_balanced_accuracy": float(balanced_accuracy_score(y_true_labels, y_pred_labels)),
    }


class CalibratedThresholdClassifier:
    """Apply probability calibration + class multipliers on top of a base classifier."""

    def __init__(
        self,
        estimator: object,
        class_labels: list[str],
        *,
        probability_calibrator: object | None = None,
        class_multipliers: list[float] | np.ndarray | None = None,
    ):
        self.estimator = estimator
        self.class_labels = list(class_labels)
        self.classes_ = np.array(self.class_labels)
        self.probability_calibrator = probability_calibrator
        if class_multipliers is None:
            self.class_multipliers = np.ones(len(self.class_labels), dtype=float)
        else:
            arr = np.asarray(class_multipliers, dtype=float)
            if arr.shape != (len(self.class_labels),):
                arr = np.ones(len(self.class_labels), dtype=float)
            self.class_multipliers = arr

    def _adjust_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        probs = probabilities
        if self.probability_calibrator is not None:
            calibrated = self.probability_calibrator.predict_proba(np.clip(probs, 1e-6, 1 - 1e-6))
            calibrator_classes = getattr(self.probability_calibrator, "classes_", self.class_labels)
            probs = _align_probabilities(calibrated, calibrator_classes, self.class_labels)

        probs = np.asarray(probs, dtype=float) * self.class_multipliers.reshape(1, -1)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        return probs / row_sums

    def predict_proba(self, X) -> np.ndarray:
        raw = self.estimator.predict_proba(X)
        base_classes = getattr(self.estimator, "classes_", self.class_labels)
        aligned = _align_probabilities(raw, base_classes, self.class_labels)
        return self._adjust_probabilities(aligned)

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def build_source_sample_weights(
    train_df: pd.DataFrame,
    *,
    source_col: str = "punteggio_source",
    weight_map: dict[str, float] | None = None,
    default_weight: float = DEFAULT_SOURCE_WEIGHT,
) -> np.ndarray:
    """Build per-row sample weights from target source quality."""
    if source_col not in train_df.columns:
        return np.ones(len(train_df), dtype=float)

    source_weight_map = weight_map or SOURCE_WEIGHT_MAP
    sources = train_df[source_col].fillna("missing").astype(str)
    return np.array([source_weight_map.get(src, default_weight) for src in sources], dtype=float)


def _slice_rows(X, indices: np.ndarray):
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return X[indices]


def _extract_pipeline_last_step_name(estimator: object) -> str | None:
    if isinstance(estimator, Pipeline) and estimator.steps:
        return estimator.steps[-1][0]
    return None


def _fit_estimator_with_optional_sample_weight(
    estimator: object,
    X_train,
    y_train,
    *,
    sample_weight: np.ndarray | None = None,
) -> object:
    """Fit an estimator, routing sample_weight when supported."""
    if sample_weight is None:
        estimator.fit(X_train, y_train)
        return estimator

    try:
        estimator.fit(X_train, y_train, sample_weight=sample_weight)
        return estimator
    except (TypeError, ValueError) as exc:
        if "sample_weight" not in str(exc):
            raise

    step_name = _extract_pipeline_last_step_name(estimator)
    if step_name is not None:
        try:
            estimator.fit(X_train, y_train, **{f"{step_name}__sample_weight": sample_weight})
            return estimator
        except (TypeError, ValueError):
            pass

    estimator.fit(X_train, y_train)
    return estimator


def _fit_grid_search_with_optional_sample_weight(
    search: GridSearchCV,
    X,
    y,
    *,
    groups: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> GridSearchCV:
    """Fit GridSearchCV while preserving optional sample-weight support."""
    fit_kwargs: dict[str, object] = {}
    if groups is not None:
        fit_kwargs["groups"] = groups

    if sample_weight is None:
        search.fit(X, y, **fit_kwargs)
        return search

    try:
        search.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        return search
    except (TypeError, ValueError) as exc:
        if "sample_weight" not in str(exc):
            raise

    step_name = _extract_pipeline_last_step_name(search.estimator)
    if step_name is not None:
        search.fit(X, y, **fit_kwargs, **{f"{step_name}__sample_weight": sample_weight})
        return search

    search.fit(X, y, **fit_kwargs)
    return search


def _resolve_cv_splits(cv, X, y, groups: np.ndarray | None) -> list[tuple[np.ndarray, np.ndarray]]:
    if hasattr(cv, "split"):
        if groups is not None:
            return list(cv.split(X, y, groups))
        return list(cv.split(X, y))
    if isinstance(cv, int):
        kfold = KFold(n_splits=cv, shuffle=False)
        return list(kfold.split(X, y))
    raise TypeError(f"Unsupported cv type: {type(cv)!r}")


def _cross_val_mae_scores(
    estimator: object,
    X,
    y,
    *,
    cv,
    groups: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Cross-validated negative MAE scores with optional sample-weighted fitting."""
    y_array = np.asarray(y)
    scores: list[float] = []
    for train_idx, test_idx in _resolve_cv_splits(cv, X, y, groups):
        model = clone(estimator)
        X_train = _slice_rows(X, train_idx)
        y_train = y_array[train_idx]
        fit_weights = sample_weight[train_idx] if sample_weight is not None else None
        _fit_estimator_with_optional_sample_weight(
            model, X_train, y_train, sample_weight=fit_weights
        )
        pred = model.predict(_slice_rows(X, test_idx))
        mae = mean_absolute_error(y_array[test_idx], pred)
        scores.append(-float(mae))
    return np.array(scores, dtype=float)


def _align_probabilities(
    probabilities: np.ndarray,
    source_classes: list[str] | np.ndarray,
    target_classes: list[str] | np.ndarray,
) -> np.ndarray:
    """Align probability columns from source_classes to target_classes."""
    source = [str(c) for c in source_classes]
    target = [str(c) for c in target_classes]
    probs = np.asarray(probabilities, dtype=float)
    if source == target:
        return probs

    aligned = np.zeros((probs.shape[0], len(target)), dtype=float)
    index_by_label = {label: idx for idx, label in enumerate(target)}
    for src_idx, src_label in enumerate(source):
        dst_idx = index_by_label.get(src_label)
        if dst_idx is not None:
            aligned[:, dst_idx] = probs[:, src_idx]

    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return aligned / row_sums


def _cross_val_predict_estimator(
    estimator: object,
    X,
    y,
    *,
    cv,
    groups: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    method: str = "predict",
    class_labels: list[str] | None = None,
):
    """Cross-validated predictions with optional sample-weighted fitting."""
    y_array = np.asarray(y)
    splits = _resolve_cv_splits(cv, X, y, groups)
    n_samples = len(y_array)

    if method == "predict_proba":
        labels = class_labels or sorted({str(label) for label in y_array})
        oof = np.zeros((n_samples, len(labels)), dtype=float)
    else:
        oof = np.empty(n_samples, dtype=object if y_array.dtype.kind in {"U", "S", "O"} else float)

    for train_idx, test_idx in splits:
        model = clone(estimator)
        X_train = _slice_rows(X, train_idx)
        y_train = y_array[train_idx]
        fit_weights = sample_weight[train_idx] if sample_weight is not None else None
        _fit_estimator_with_optional_sample_weight(
            model, X_train, y_train, sample_weight=fit_weights
        )

        fold_X = _slice_rows(X, test_idx)
        if method == "predict_proba":
            raw_probs = model.predict_proba(fold_X)
            fold_probs = _align_probabilities(raw_probs, getattr(model, "classes_", labels), labels)
            oof[test_idx, :] = fold_probs
        else:
            oof[test_idx] = getattr(model, method)(fold_X)

    return oof


def _fit_probability_calibrator(
    probabilities: np.ndarray,
    y_labels: list[str],
    *,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression | None:
    """Fit a lightweight multinomial calibrator over model probabilities."""
    unique_labels = sorted(set(y_labels))
    if len(unique_labels) < 2:
        return None

    calibrator = LogisticRegression(
        C=1.0,
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
    )
    features = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    calibrator.fit(features, y_labels, sample_weight=sample_weight)
    return calibrator


def _optimize_probability_multipliers(
    probabilities: np.ndarray,
    y_true_labels: list[str],
    class_labels: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    """Tune class-wise probability multipliers to maximize macro-F1."""
    if probabilities.size == 0:
        return np.ones(len(class_labels), dtype=float), {
            "category_accuracy": float("nan"),
            "category_macro_f1": float("nan"),
            "category_balanced_accuracy": float("nan"),
        }

    candidates = [0.8, 0.9, 1.0, 1.1, 1.2]
    best_score: tuple[float, float, float] | None = None
    best_multipliers = np.ones(len(class_labels), dtype=float)
    best_metrics: dict[str, float] = {}
    probs = np.asarray(probabilities, dtype=float)

    for multipliers in product(candidates, repeat=len(class_labels)):
        mult = np.asarray(multipliers, dtype=float).reshape(1, -1)
        adjusted = probs * mult
        row_sums = adjusted.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        adjusted /= row_sums
        pred_labels = [class_labels[idx] for idx in np.argmax(adjusted, axis=1)]
        metrics = calculate_category_metrics(y_true_labels, pred_labels)
        score = (
            metrics["category_macro_f1"],
            metrics["category_balanced_accuracy"],
            metrics["category_accuracy"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_multipliers = np.asarray(multipliers, dtype=float)
            best_metrics = metrics

    return best_multipliers, best_metrics


def train_dedicated_category_classifier(
    X: pd.DataFrame,
    y_labels: list[str],
    cv,
    groups: np.ndarray | None,
    sample_weight: np.ndarray | None = None,
) -> tuple[object | None, dict]:
    """Train and select a dedicated classifier for LOW/MEDIUM/HIGH categories."""
    unique_groups = np.unique(groups) if groups is not None else np.array([])
    use_grouped_cv = groups is not None and len(unique_groups) >= 2
    can_tune = len(X) >= 60 and use_grouped_cv and len(unique_groups) >= 3
    class_labels = ["LOW", "MEDIUM", "HIGH"]
    diagnostics = {
        "grouped_cv": bool(use_grouped_cv),
        "hyperparam_tuning_enabled": bool(can_tune),
        "lgbm_classifier_available": LGBMClassifier is not None,
        "catboost_classifier_available": CatBoostClassifier is not None,
        "sample_weighting_enabled": sample_weight is not None,
        "probability_calibration": "logistic_oof",
        "threshold_tuning": "class_probability_multipliers",
    }

    classifier_specs: dict[str, dict] = {
        "rf_classifier": {
            "builder": lambda: RandomForestClassifier(
                n_estimators=420,
                max_depth=4,
                min_samples_leaf=1,
                min_samples_split=10,
                max_features=1.0,
                bootstrap=True,
                class_weight="balanced",
                random_state=42,
            ),
            "param_grid": {
                "n_estimators": [320, 420],
                "max_depth": [4, 5],
                "min_samples_leaf": [1, 2],
                "min_samples_split": [6, 10],
                "max_features": [1.0],
                "bootstrap": [True],
            },
        },
        "gb_classifier": {
            "builder": lambda: GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
            "param_grid": {},
        },
        "logreg_classifier": {
            "builder": lambda: Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "logreg",
                        LogisticRegression(
                            C=0.8,
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "param_grid": {},
        },
    }
    if LGBMClassifier is not None:
        classifier_specs["lgbm_classifier"] = {
            "builder": lambda: LGBMClassifier(
                n_estimators=160,
                max_depth=-1,
                num_leaves=7,
                learning_rate=0.05,
                min_child_samples=3,
                subsample=0.9,
                colsample_bytree=0.6,
                reg_alpha=0.0,
                reg_lambda=5.0,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
                n_jobs=1,
            ),
            "param_grid": {
                "n_estimators": [120, 160],
                "learning_rate": [0.03, 0.05],
                "num_leaves": [7, 15],
                "max_depth": [-1, 4],
                "min_child_samples": [3],
            },
        }
    if CatBoostClassifier is not None:
        classifier_specs["catboost_classifier"] = {
            "builder": lambda: CatBoostClassifier(
                iterations=350,
                depth=5,
                learning_rate=0.05,
                l2_leaf_reg=6.0,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            ),
            "param_grid": {},
        }

    results: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}
    y_true = list(y_labels)

    for model_name, spec in classifier_specs.items():
        try:
            estimator = spec["builder"]()
            best_params = None
            if can_tune and spec.get("param_grid"):
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=spec["param_grid"],
                    cv=cv,
                    scoring="f1_macro",
                    n_jobs=1,
                )
                _fit_grid_search_with_optional_sample_weight(
                    search,
                    X,
                    y_labels,
                    groups=groups if use_grouped_cv else None,
                    sample_weight=sample_weight,
                )
                estimator = search.best_estimator_
                best_params = search.best_params_

            probs_oof = _cross_val_predict_estimator(
                estimator,
                X,
                y_labels,
                cv=cv,
                groups=groups if use_grouped_cv else None,
                sample_weight=sample_weight,
                method="predict_proba",
                class_labels=class_labels,
            )
            calibrator = _fit_probability_calibrator(probs_oof, y_true, sample_weight=sample_weight)
            calibrated_probs = probs_oof
            if calibrator is not None:
                calibrated_raw = calibrator.predict_proba(np.clip(probs_oof, 1e-6, 1 - 1e-6))
                calibrated_probs = _align_probabilities(
                    calibrated_raw,
                    getattr(calibrator, "classes_", class_labels),
                    class_labels,
                )

            multipliers, tuned_metrics = _optimize_probability_multipliers(
                calibrated_probs,
                y_true,
                class_labels,
            )
            pred_labels = [
                class_labels[idx]
                for idx in np.argmax(calibrated_probs * multipliers.reshape(1, -1), axis=1)
            ]
            tuned_metrics = calculate_category_metrics(y_true, pred_labels)
            model_metrics = {
                **tuned_metrics,
                "best_params": best_params,
                "threshold_multipliers": {
                    label: float(mult)
                    for label, mult in zip(class_labels, multipliers, strict=False)
                },
                "probability_calibrated": calibrator is not None,
            }
            results[model_name] = model_metrics

            final_estimator = clone(estimator)
            _fit_estimator_with_optional_sample_weight(
                final_estimator,
                X,
                y_labels,
                sample_weight=sample_weight,
            )
            wrapped_estimator = CalibratedThresholdClassifier(
                final_estimator,
                class_labels,
                probability_calibrator=calibrator,
                class_multipliers=multipliers,
            )
            fitted_models[model_name] = wrapped_estimator
        except Exception as exc:
            print(f"  Warning: could not train {model_name}: {exc}")

    if not fitted_models:
        return None, {"best_model": None, "per_model": {}, "diagnostics": diagnostics}

    def _score(item: tuple[str, dict[str, float]]) -> tuple[float, float, float]:
        _name, metrics = item
        return (
            metrics["category_macro_f1"],
            metrics["category_balanced_accuracy"],
            metrics["category_accuracy"],
        )

    best_model_name, best_metrics = max(results.items(), key=_score)
    best_metric_values = {k: v for k, v in best_metrics.items() if k.startswith("category_")}
    return fitted_models[best_model_name], {
        "best_model": best_model_name,
        "best_metrics": best_metric_values,
        "best_params": best_metrics.get("best_params"),
        "per_model": results,
        "diagnostics": diagnostics,
    }


def evaluate_on_validation(models: dict, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
    """Evaluate each model on a validation set, return MAE per model."""
    mae_scores = {}
    for name, model in models.items():
        pred = model.predict(X_val)
        mae_scores[name] = float(mean_absolute_error(y_val, pred))
    return mae_scores


def compute_ensemble_weights(val_mae: dict[str, float]) -> dict[str, float]:
    """Compute inverse-MAE normalized weights for ensemble."""
    inv_mae = {name: 1.0 / max(mae, 1e-6) for name, mae in val_mae.items()}
    total = sum(inv_mae.values())
    return {name: val / total for name, val in inv_mae.items()}


def nested_cross_validation(
    models_dict: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    param_grids: dict,
) -> dict:
    """
    Perform nested cross-validation for robust model evaluation.

    Outer loop: LeaveOneGroupOut (by year) for unbiased performance estimation.
    Inner loop: LeaveOneGroupOut for hyperparameter tuning without leakage.

    Args:
        models_dict: Dictionary of model names to instantiated model classes
        X: Feature matrix
        y: Target values
        groups: Group labels (years) for splitting
        param_grids: Dictionary of model names to parameter grids for GridSearchCV

    Returns:
        Dict containing:
            - outer_mae_per_fold: MAE scores for each outer fold (year held out)
            - mean_outer_mae: Mean MAE across all outer folds
            - std_outer_mae: Std of MAE across all outer folds
            - best_params_per_fold: Best hyperparameters found in each outer fold
            - outer_years: Which years were held out in each fold
            - model_names: Names of models evaluated
    """
    outer_cv = LeaveOneGroupOut()
    outer_folds = list(outer_cv.split(X, y, groups))

    results = {
        "outer_mae_per_fold": {},
        "mean_outer_mae": {},
        "std_outer_mae": {},
        "best_params_per_fold": {},
        "outer_years": [],
        "model_names": list(models_dict.keys()),
    }

    for model_name in models_dict:
        results["outer_mae_per_fold"][model_name] = []
        results["best_params_per_fold"][model_name] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        held_out_year = int(np.unique(groups[test_idx])[0])
        results["outer_years"].append(held_out_year)

        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        groups_train_outer = groups[train_idx]

        for model_name, model_class in models_dict.items():
            if model_name in param_grids:
                inner_cv = LeaveOneGroupOut()

                grid_search = GridSearchCV(
                    model_class(),
                    param_grids[model_name],
                    cv=inner_cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=1,
                )

                grid_search.fit(X_train_outer, y_train_outer, groups=groups_train_outer)
                best_model = grid_search.best_estimator_
                results["best_params_per_fold"][model_name].append(grid_search.best_params_)
            else:
                best_model = model_class()
                best_model.fit(X_train_outer, y_train_outer)
                results["best_params_per_fold"][model_name].append(None)

            y_pred = best_model.predict(X_test_outer)
            fold_mae = float(mean_absolute_error(y_test_outer, y_pred))
            results["outer_mae_per_fold"][model_name].append(fold_mae)

    for model_name in models_dict:
        maes = results["outer_mae_per_fold"][model_name]
        results["mean_outer_mae"][model_name] = float(np.mean(maes))
        results["std_outer_mae"][model_name] = float(np.std(maes))

    return results


def train_models(use_stacking: bool = False) -> tuple[dict, dict]:
    """
    Train ensemble models using real historical data.

    Args:
        use_stacking: If True, train stacking ensemble in addition to base models

    Returns:
        Tuple of (models_dict, metrics_dict)
        models_dict: {"rf": model, "gb": model, "ridge": model, ...}
        If use_stacking=True, also includes "stacking" key
    """
    print("Loading data and preparing time-aware features...")
    builder = FeatureBuilder()
    sources = builder.load_sources()
    full_df = builder.build_training_frame(sources)
    splits = builder.split_by_years(
        full_df, builder.config.training_years, builder.config.validation_years
    )
    train_df = splits["train"]

    if train_df.empty:
        raise RuntimeError("No training data available. Check input data sources.")

    # Run data-quality checks before any imputation/normalization.
    all_feature_cols = builder.get_feature_columns(full_df)
    quality_report = run_training_quality_checks(
        train_df,
        all_feature_cols,
        builder.config.training_years,
    )
    if quality_report["status"] != "pass":
        failed = "; ".join(quality_report["failed_checks"])
        raise RuntimeError(f"Training blocked by ML quality checks: {failed}")
    if quality_report["warnings"]:
        print(f"  Quality warnings: {'; '.join(quality_report['warnings'])}")

    raw_targets = train_df["punteggio_reale"].copy()
    baseline_score_thresholds = derive_score_thresholds(raw_targets)

    # Normalize targets by year
    train_df, year_stats = normalize_targets(train_df)

    # Feature columns (exclude identifiers)
    feature_cols = all_feature_cols
    X_raw = train_df[feature_cols].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    feature_fill_values = {
        col: float(X_raw[col].median()) if pd.api.types.is_numeric_dtype(X_raw[col]) else 0.0
        for col in feature_cols
    }
    for col in feature_cols:
        fill_value = feature_fill_values.get(col, 0.0)
        X_raw[col] = X_raw[col].fillna(fill_value)
    y = train_df["punteggio_reale"]
    # Hybrid mode:
    # - category branch uses source-based sample weights (v5)
    # - regression branch keeps v4 behavior (no sample weighting)
    category_sample_weights = build_source_sample_weights(train_df)
    weight_summary = {
        "min": (
            float(np.min(category_sample_weights)) if len(category_sample_weights) else float("nan")
        ),
        "max": (
            float(np.max(category_sample_weights)) if len(category_sample_weights) else float("nan")
        ),
        "mean": (
            float(np.mean(category_sample_weights))
            if len(category_sample_weights)
            else float("nan")
        ),
    }
    regression_sample_weights: np.ndarray | None = None

    # Keep category feature-space stable, while using a stricter cutoff for regression.
    category_redundant = remove_redundant_features(X_raw, threshold=0.95)
    category_base_feature_cols = [c for c in feature_cols if c not in category_redundant]
    if category_redundant:
        print(
            f"  Category: removed {len(category_redundant)} redundant features "
            f"(corr>0.95): {category_redundant}"
        )

    regression_redundant = remove_redundant_features(X_raw, threshold=0.98)
    feature_cols = [c for c in feature_cols if c not in regression_redundant]
    if regression_redundant:
        print(
            f"  Regression: removed {len(regression_redundant)} redundant features "
            f"(corr>0.98): {regression_redundant}"
        )

    # Dedicated category feature space and threshold strategy.
    groups = train_df["anno"].values
    n_groups = len(np.unique(groups))
    cv_for_category = LeaveOneGroupOut() if n_groups >= 2 else 2
    groups_for_category = groups if n_groups >= 2 else None

    threshold_candidates = {
        "quantile_33_66": baseline_score_thresholds,
        "quantile_30_70": derive_score_thresholds(raw_targets.values, 0.30, 0.70),
        "fixed_default": DEFAULT_THRESHOLDS.copy(),
    }
    category_feature_selection_seeds = [6, 42]
    category_strategy_runs: dict[str, dict] = {}
    chosen_strategy_name = None
    chosen_strategy_score = (-1.0, -1.0, -1.0)
    chosen_category_model = None
    chosen_category_meta = None
    score_thresholds = baseline_score_thresholds
    true_categories = categorize_scores(raw_targets.values, score_thresholds)
    category_feature_cols: list[str] = []
    category_feature_fill_values: dict[str, float] = {}
    category_winsorization_bounds: dict[str, dict[str, float]] = {}
    X_category = X_raw[feature_cols].copy()

    for strategy_name, thresholds in threshold_candidates.items():
        strategy_labels = categorize_scores(raw_targets.values, thresholds)
        seed_runs: dict[str, dict] = {}
        strategy_best_payload: dict | None = None
        strategy_best_score = (-1.0, -1.0, -1.0)

        for fs_seed in category_feature_selection_seeds:
            strategy_feature_cols = select_classifier_features(
                X_raw,
                strategy_labels,
                category_base_feature_cols,
                random_state=fs_seed,
            )
            strategy_fill_values = {
                col: (
                    float(X_raw[col].median()) if pd.api.types.is_numeric_dtype(X_raw[col]) else 0.0
                )
                for col in strategy_feature_cols
            }
            strategy_X = X_raw[strategy_feature_cols].copy()
            for col in strategy_feature_cols:
                strategy_X[col] = strategy_X[col].fillna(strategy_fill_values.get(col, 0.0))
            strategy_bounds = compute_winsorization_bounds(strategy_X)
            strategy_X = apply_winsorization(strategy_X, strategy_bounds)

            model, model_meta = train_dedicated_category_classifier(
                strategy_X,
                strategy_labels,
                cv=cv_for_category,
                groups=groups_for_category,
                sample_weight=category_sample_weights,
            )
            best_metrics = (
                model_meta.get("best_metrics", {}) if isinstance(model_meta, dict) else {}
            )
            score_tuple = (
                float(best_metrics.get("category_macro_f1", float("nan"))),
                float(best_metrics.get("category_balanced_accuracy", float("nan"))),
                float(best_metrics.get("category_accuracy", float("nan"))),
            )

            seed_key = str(fs_seed)
            seed_runs[seed_key] = {
                "feature_selection_seed": fs_seed,
                "best_model": (
                    model_meta.get("best_model") if isinstance(model_meta, dict) else None
                ),
                "best_metrics": best_metrics,
                "feature_count": len(strategy_feature_cols),
            }

            if model is None or any(np.isnan(score_tuple)):
                continue
            if score_tuple > strategy_best_score:
                strategy_best_score = score_tuple
                strategy_best_payload = {
                    "model": model,
                    "model_meta": model_meta,
                    "feature_cols": strategy_feature_cols,
                    "feature_fill_values": strategy_fill_values,
                    "winsorization_bounds": strategy_bounds,
                    "X_category": strategy_X,
                    "feature_selection_seed": fs_seed,
                    "score_tuple": score_tuple,
                }

        strategy_best_metrics = (
            strategy_best_payload["model_meta"].get("best_metrics", {})
            if strategy_best_payload and isinstance(strategy_best_payload.get("model_meta"), dict)
            else {}
        )
        category_strategy_runs[strategy_name] = {
            "thresholds": thresholds,
            "distribution": {
                label: int(strategy_labels.count(label)) for label in ["LOW", "MEDIUM", "HIGH"]
            },
            "best_seed": (
                strategy_best_payload.get("feature_selection_seed")
                if strategy_best_payload
                else None
            ),
            "best_model": (
                strategy_best_payload["model_meta"].get("best_model")
                if (
                    strategy_best_payload
                    and isinstance(strategy_best_payload.get("model_meta"), dict)
                )
                else None
            ),
            "best_metrics": strategy_best_metrics,
            "seed_runs": seed_runs,
        }

        if strategy_best_payload is None:
            continue
        if strategy_best_payload["score_tuple"] > chosen_strategy_score:
            chosen_strategy_score = strategy_best_payload["score_tuple"]
            chosen_strategy_name = strategy_name
            chosen_category_model = strategy_best_payload["model"]
            chosen_category_meta = strategy_best_payload["model_meta"]
            chosen_category_meta["feature_selection_seed"] = strategy_best_payload[
                "feature_selection_seed"
            ]
            score_thresholds = thresholds
            true_categories = strategy_labels
            category_feature_cols = strategy_best_payload["feature_cols"]
            category_feature_fill_values = strategy_best_payload["feature_fill_values"]
            category_winsorization_bounds = strategy_best_payload["winsorization_bounds"]
            X_category = strategy_best_payload["X_category"]

    if chosen_strategy_name is None:
        chosen_strategy_name = "quantile_33_66"
        chosen_category_model = None
        chosen_category_meta = {
            "best_model": None,
            "best_metrics": {},
            "per_model": {},
            "feature_selection_seed": None,
        }

    category_threshold_eval = {
        "best_strategy": chosen_strategy_name,
        "best_thresholds": score_thresholds,
        "per_strategy": category_strategy_runs,
    }
    print(
        "  Category threshold strategy: "
        f"{chosen_strategy_name} -> "
        f"LOW < {score_thresholds['low_max']:.1f}, "
        f"MEDIUM < {score_thresholds['medium_max']:.1f}"
    )
    print(f"  Category feature set: {len(category_feature_cols)} features")
    print(
        f"  Category feature selection seed: {chosen_category_meta.get('feature_selection_seed')}"
    )

    # FASE 3B: Adaptive feature selection (5:1 samples:features ratio)
    feature_cols = select_features_adaptive(X_raw, y, feature_cols, ratio=5)
    print(f"  Selected {len(feature_cols)} features for {len(X_raw)} samples")

    # Light outlier control on numeric features (regression only).
    winsorization_bounds = compute_winsorization_bounds(X_raw[feature_cols])
    X = apply_winsorization(X_raw[feature_cols], winsorization_bounds)

    print(f"Training on {len(X)} samples from {len(train_df['artista_id'].unique())} artists")
    print(f"Target score range (normalized): {y.min():.2f} - {y.max():.2f} (avg: {y.mean():.2f})")
    print(
        "Score category thresholds: "
        f"LOW < {score_thresholds['low_max']:.1f}, "
        f"MEDIUM < {score_thresholds['medium_max']:.1f}, "
        "HIGH >= medium threshold"
    )

    # FASE 1B: Time-aware cross-validation
    groups = train_df["anno"].values
    n_groups = len(np.unique(groups))

    if n_groups >= 2:
        logo = LeaveOneGroupOut()
        cv = logo
    else:
        cv = 2

    # FASE 1A: Regularized models (optional deps handled gracefully)
    model_specs: dict[str, tuple[str, callable]] = {
        "rf": (
            "Random Forest (regularized)",
            lambda: RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=3,
                max_features=0.5,
                random_state=42,
            ),
        ),
        "gb": (
            "Gradient Boosting (regularized)",
            lambda: GradientBoostingRegressor(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.05,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=42,
            ),
        ),
        "ridge": ("Ridge Regression", lambda: Ridge(alpha=10.0)),
    }

    if XGBRegressor is not None:
        model_specs["xgb"] = (
            "XGBoost (regularized)",
            lambda: XGBRegressor(
                n_estimators=500,
                max_depth=2,
                learning_rate=0.15,
                min_child_weight=2,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=10.0,
                subsample=0.6,
                colsample_bytree=0.5,
                random_state=42,
            ),
        )
    else:
        print("Skipping XGBoost: package not installed")

    if LGBMRegressor is not None:
        model_specs["lgbm"] = (
            "LightGBM (regularized)",
            lambda: LGBMRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.05,
                min_child_samples=5,
                reg_lambda=5.0,
                subsample=0.8,
                random_state=42,
                verbose=-1,
            ),
        )
    else:
        print("Skipping LightGBM: package not installed")

    models: dict[str, object] = {}
    cv_scores: dict[str, np.ndarray] = {}
    for model_key, (label, model_factory) in model_specs.items():
        print(f"\nTraining {label}...")
        model = model_factory()
        scores = _cross_val_mae_scores(
            model,
            X,
            y,
            cv=cv,
            groups=groups if n_groups >= 2 else None,
            sample_weight=regression_sample_weights,
        )
        _fit_estimator_with_optional_sample_weight(
            model, X, y, sample_weight=regression_sample_weights
        )
        models[model_key] = model
        cv_scores[model_key] = scores

    # FASE 5A: Hyperparameter tuning with GridSearchCV (when enough samples)
    if len(X) >= 50 and n_groups >= 3:
        print("\nRunning GridSearchCV (enough samples for tuning)...")
        best_params = {}

        param_grid_rf = {
            "n_estimators": [80, 100, 120],
            "max_depth": [4, 5, 6],
            "min_samples_leaf": [1, 2, 3],
            "max_features": [0.5, 0.7],
        }
        grid_rf = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid_rf,
            cv=logo,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        _fit_grid_search_with_optional_sample_weight(
            grid_rf,
            X,
            y,
            groups=groups,
            sample_weight=regression_sample_weights,
        )
        models["rf"] = grid_rf.best_estimator_
        best_params["rf"] = grid_rf.best_params_
        print(f"  RF best params: {grid_rf.best_params_} (MAE: {-grid_rf.best_score_:.3f})")

        param_grid_gb = {
            "n_estimators": [80, 100, 120],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
        grid_gb = GridSearchCV(
            GradientBoostingRegressor(min_samples_leaf=3, random_state=42),
            param_grid_gb,
            cv=logo,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        _fit_grid_search_with_optional_sample_weight(
            grid_gb,
            X,
            y,
            groups=groups,
            sample_weight=regression_sample_weights,
        )
        models["gb"] = grid_gb.best_estimator_
        best_params["gb"] = grid_gb.best_params_
        print(f"  GB best params: {grid_gb.best_params_} (MAE: {-grid_gb.best_score_:.3f})")

        param_grid_ridge = {"alpha": [2.0, 5.0, 8.0, 12.0]}
        grid_ridge = GridSearchCV(
            Ridge(),
            param_grid_ridge,
            cv=logo,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        _fit_grid_search_with_optional_sample_weight(
            grid_ridge,
            X,
            y,
            groups=groups,
            sample_weight=regression_sample_weights,
        )
        models["ridge"] = grid_ridge.best_estimator_
        best_params["ridge"] = grid_ridge.best_params_
        ridge_mae = -grid_ridge.best_score_
        print(f"  Ridge best params: {grid_ridge.best_params_} (MAE: {ridge_mae:.3f})")

        if "xgb" in models and XGBRegressor is not None:
            param_grid_xgb = {
                "n_estimators": [300, 500],
                "max_depth": [2, 3],
                "learning_rate": [0.1, 0.15],
                "reg_lambda": [8.0, 10.0],
                "reg_alpha": [0.0, 0.1],
                "min_child_weight": [2],
                "subsample": [0.6],
                "colsample_bytree": [0.5, 0.7],
                "gamma": [0.05, 0.1],
            }
            grid_xgb = GridSearchCV(
                XGBRegressor(random_state=42),
                param_grid_xgb,
                cv=logo,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )
            _fit_grid_search_with_optional_sample_weight(
                grid_xgb,
                X,
                y,
                groups=groups,
                sample_weight=regression_sample_weights,
            )
            models["xgb"] = grid_xgb.best_estimator_
            best_params["xgb"] = grid_xgb.best_params_
            print(f"  XGB best params: {grid_xgb.best_params_} (MAE: {-grid_xgb.best_score_:.3f})")

        if "lgbm" in models and LGBMRegressor is not None:
            param_grid_lgbm = {
                "n_estimators": [80, 100, 120],
                "max_depth": [3, 4],
                "learning_rate": [0.05, 0.1],
                "reg_lambda": [1.0, 3.0, 5.0],
            }
            grid_lgbm = GridSearchCV(
                LGBMRegressor(min_child_samples=5, subsample=0.8, random_state=42, verbose=-1),
                param_grid_lgbm,
                cv=logo,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )
            _fit_grid_search_with_optional_sample_weight(
                grid_lgbm,
                X,
                y,
                groups=groups,
                sample_weight=regression_sample_weights,
            )
            models["lgbm"] = grid_lgbm.best_estimator_
            best_params["lgbm"] = grid_lgbm.best_params_
            print(
                f"  LGBM best params: {grid_lgbm.best_params_} (MAE: {-grid_lgbm.best_score_:.3f})"
            )
    else:
        best_params = None

    # Refresh CV scores in case tuned estimators replaced base defaults.
    for model_key, model in models.items():
        cv_scores[model_key] = _cross_val_mae_scores(
            model,
            X,
            y,
            cv=cv,
            groups=groups if n_groups >= 2 else None,
            sample_weight=regression_sample_weights,
        )

    # FASE 1C: Dynamic ensemble weights from validation sets
    val_frames = splits.get("val", {})
    all_val_mae: dict[str, list[float]] = {name: [] for name in models}

    for year, val_df in val_frames.items():
        if val_df.empty or val_df["punteggio_reale"].isna().all():
            continue
        val_normalized = val_df.copy()
        val_y = val_normalized["punteggio_reale"]
        # Normalize validation targets using training year stats (use global mean/std)
        global_mean = np.mean([s["mean"] for s in year_stats.values()])
        global_std = np.mean([s["std"] for s in year_stats.values()])
        if global_std == 0:
            global_std = 1.0
        val_y_norm = (val_y - global_mean) / global_std

        X_val = val_normalized[feature_cols].copy()
        X_val = X_val.replace([np.inf, -np.inf], np.nan)
        for col in feature_cols:
            X_val[col] = X_val[col].fillna(feature_fill_values.get(col, 0.0))
        X_val = apply_winsorization(X_val, winsorization_bounds)
        if X_val.empty:
            continue
        val_mae = evaluate_on_validation(models, X_val, val_y_norm)
        for name, mae in val_mae.items():
            all_val_mae[name].append(mae)
        val_str = " ".join(f"{k.upper()}={v:.3f}" for k, v in val_mae.items())
        print(f"  Validation {year}: {val_str}")

    # Compute average MAE across validation sets
    avg_val_mae = {}
    for name, maes in all_val_mae.items():
        avg_val_mae[name] = float(np.mean(maes)) if maes else float(-cv_scores[name].mean())

    ensemble_weights = compute_ensemble_weights(avg_val_mae)
    print(f"\nEnsemble weights: {', '.join(f'{k}={v:.3f}' for k, v in ensemble_weights.items())}")

    # Train stacking ensemble if requested
    stacking_metrics = None
    required_for_stacking = {"rf", "gb", "ridge", "xgb", "lgbm"}
    if use_stacking and required_for_stacking.issubset(models.keys()):
        print("\nTraining stacking ensemble...")
        print("  Using LeaveOneGroupOut CV to prevent temporal leakage")

        # Extract base models for stacking (core 5 models)
        base_models = {
            "rf": models["rf"],
            "gb": models["gb"],
            "ridge": models["ridge"],
            "xgb": models["xgb"],
            "lgbm": models["lgbm"],
        }

        # Train stacking model
        stacking_model = train_stacking_ensemble(base_models, X, y, groups, feature_cols)
        models["stacking"] = stacking_model

        # Evaluate stacking on validation sets
        stacking_val_metrics = evaluate_stacking_on_multiple_years(
            stacking_model, val_frames, feature_cols, year_stats
        )

        # Print validation results
        print("  Stacking validation results:")
        for year, year_metrics in stacking_val_metrics.items():
            print(f"    {year}: MAE={year_metrics['mae']:.3f}, R²={year_metrics['r2']:.3f}")

        stacking_metrics = stacking_val_metrics
    elif use_stacking:
        print(
            "Skipping stacking ensemble: requires rf/gb/ridge/xgb/lgbm and optional dependencies."
        )

    # Calculate metrics
    metric_name_map = {
        "rf": "random_forest",
        "gb": "gradient_boosting",
        "ridge": "ridge",
        "xgb": "xgboost",
        "lgbm": "lightgbm",
    }
    metrics: dict[str, dict] = {}
    for model_key, scores in cv_scores.items():
        model_name = metric_name_map.get(model_key, model_key)
        metrics[model_name] = {
            "mae_cv": float(-scores.mean()),
            "mae_std": float(scores.std()),
            "samples": len(X),
            "features": len(feature_cols),
        }

    # Calculate enhanced metrics (regression + score-category classification)
    print(
        "\nCalculating enhanced metrics (R², RMSE, MAPE, DirAcc + category Accuracy/F1/BalAcc)..."
    )
    true_categories = categorize_scores(raw_targets.values, score_thresholds)
    oof_predictions: dict[str, np.ndarray] = {}

    for model_key, model in models.items():
        if model_key == "stacking":
            continue
        model_name = metric_name_map.get(model_key, model_key)
        try:
            # Get out-of-fold predictions
            y_pred_cv = _cross_val_predict_estimator(
                model,
                X,
                y,
                cv=cv,
                groups=groups if n_groups >= 2 else None,
                sample_weight=regression_sample_weights,
                method="predict",
            )
            y_pred_real = inverse_normalize_predictions(y_pred_cv, year_stats)
            pred_categories = categorize_scores(y_pred_real, score_thresholds)

            # Cache OOF predictions for ensemble diagnostics
            oof_predictions[model_key] = y_pred_cv

            # Calculate enhanced regression + category metrics on real-score scale
            enhanced = calculate_enhanced_metrics(raw_targets.values, y_pred_real)
            category_metrics = calculate_category_metrics(true_categories, pred_categories)

            metrics[model_name].update(enhanced)
            metrics[model_name].update(category_metrics)
            print(
                f"  {model_name}: R²={enhanced['r2']:.3f}, "
                f"RMSE={enhanced['rmse']:.3f}, "
                f"MAPE={enhanced['mape']:.1f}%, "
                f"DirAcc={enhanced['directional_accuracy']:.1%}, "
                f"CatF1={category_metrics['category_macro_f1']:.3f}"
            )
        except Exception as e:
            print(f"  Warning: Could not calculate enhanced metrics for {model_name}: {e}")
            metrics[model_name].update(
                {
                    "r2": float("nan"),
                    "rmse": float("nan"),
                    "mape": float("nan"),
                    "directional_accuracy": float("nan"),
                    "category_accuracy": float("nan"),
                    "category_macro_f1": float("nan"),
                    "category_balanced_accuracy": float("nan"),
                }
            )

    ensemble_strategy = "weighted"
    ensemble_candidates_metrics: dict[str, dict[str, float]] = {}
    if oof_predictions:
        ensemble_oof_weighted = np.zeros(len(X), dtype=float)
        ordered_keys = list(oof_predictions.keys())
        for model_key, oof_pred in oof_predictions.items():
            ensemble_oof_weighted += ensemble_weights.get(model_key, 0.0) * oof_pred

        stacked_oof = np.column_stack([oof_predictions[key] for key in ordered_keys])
        ensemble_oof_candidates = {
            "weighted": ensemble_oof_weighted,
            "mean": np.mean(stacked_oof, axis=1),
            "median": np.median(stacked_oof, axis=1),
        }

        best_ensemble_score: tuple[float, float, float] | None = None
        for candidate_name, candidate_pred_norm in ensemble_oof_candidates.items():
            candidate_pred_real = inverse_normalize_predictions(candidate_pred_norm, year_stats)
            candidate_categories = categorize_scores(candidate_pred_real, score_thresholds)
            candidate_metrics = calculate_enhanced_metrics(raw_targets.values, candidate_pred_real)
            candidate_category_metrics = calculate_category_metrics(
                true_categories, candidate_categories
            )
            candidate_mae = float(mean_absolute_error(raw_targets.values, candidate_pred_real))
            ensemble_candidates_metrics[candidate_name] = {
                "mae": candidate_mae,
                **candidate_metrics,
                **candidate_category_metrics,
            }

            # Prioritize lower RMSE, then lower MAE, then higher macro-F1.
            score_tuple = (
                -candidate_metrics["rmse"],
                -candidate_mae,
                candidate_category_metrics["category_macro_f1"],
            )
            if best_ensemble_score is None or score_tuple > best_ensemble_score:
                best_ensemble_score = score_tuple
                ensemble_strategy = candidate_name

        selected_ensemble = ensemble_candidates_metrics.get(ensemble_strategy, {})
        metrics["ensemble"] = {
            "samples": len(X),
            "features": len(feature_cols),
            "strategy": ensemble_strategy,
            **selected_ensemble,
        }

    # Dedicated category classifier (separate objective from score regression)
    print("\nTraining dedicated category classifier (LOW/MEDIUM/HIGH)...")
    category_classifier = chosen_category_model
    category_classifier_meta = chosen_category_meta or {"best_model": None, "best_metrics": {}}
    if category_classifier is not None:
        models["category_classifier"] = category_classifier
        best_name = category_classifier_meta.get("best_model")
        best_metrics = category_classifier_meta.get("best_metrics", {})
        metrics["category_classifier"] = {
            "model_name": best_name,
            "samples": len(X_category),
            "features": len(category_feature_cols),
            "feature_selection_seed": category_classifier_meta.get("feature_selection_seed"),
            "best_params": category_classifier_meta.get("best_params"),
            "diagnostics": category_classifier_meta.get("diagnostics", {}),
            **best_metrics,
        }
        metrics["category_classifier_candidates"] = category_classifier_meta.get("per_model", {})
        print(
            f"  Best classifier: {best_name} "
            f"(macro-F1={best_metrics.get('category_macro_f1', float('nan')):.3f}, "
            f"bal-acc={best_metrics.get('category_balanced_accuracy', float('nan')):.3f})"
        )
    else:
        metrics["category_classifier"] = None
        metrics["category_classifier_candidates"] = {}
        metrics["category_classifier_diagnostics"] = category_classifier_meta.get("diagnostics", {})
        print("  Warning: category classifier training failed; using score-threshold fallback.")

    metrics.update(
        {
            "ensemble_weights": ensemble_weights,
            "ensemble_strategy": ensemble_strategy,
            "ensemble_candidate_metrics": ensemble_candidates_metrics,
            "year_stats": year_stats,
            "selected_features": feature_cols,
            "feature_fill_values": {k: feature_fill_values[k] for k in feature_cols},
            "winsorization_bounds": {
                k: winsorization_bounds[k] for k in feature_cols if k in winsorization_bounds
            },
            "category_classifier_features": category_feature_cols,
            "category_classifier_feature_fill_values": {
                k: category_feature_fill_values[k] for k in category_feature_cols
            },
            "category_classifier_winsorization_bounds": {
                k: category_winsorization_bounds[k]
                for k in category_feature_cols
                if k in category_winsorization_bounds
            },
            "category_feature_selection_seed": category_classifier_meta.get(
                "feature_selection_seed"
            ),
            "score_thresholds": score_thresholds,
            "category_threshold_strategy": category_threshold_eval.get("best_strategy"),
            "category_threshold_strategies": category_threshold_eval.get("per_strategy"),
            "removed_redundant": regression_redundant,
            "removed_redundant_category": category_redundant,
            "best_params": best_params,
            "quality_checks": quality_report,
            "sample_weighting": {
                "enabled": True,
                "source_col": "punteggio_source",
                "source_weight_map": SOURCE_WEIGHT_MAP,
                "weight_summary": weight_summary,
                "applied_to": {
                    "classification": True,
                    "regression": False,
                },
            },
            "category_classifier_diagnostics": category_classifier_meta.get("diagnostics", {}),
            "data_stats": {
                "total_artists": int(train_df["artista_id"].nunique()),
                "artists_with_history": int(train_df["artista_id"].nunique()),
                "debuttanti": int((train_df["is_debuttante"] == 1).sum())
                if "is_debuttante" in train_df.columns
                else 0,
                "total_records": len(train_df),
                "missing_ratio_selected_features": float(
                    train_df[feature_cols].isna().mean().mean()
                ),
                "category_distribution": {
                    label: int(true_categories.count(label)) for label in ["LOW", "MEDIUM", "HIGH"]
                },
                "target_source_distribution": train_df["punteggio_source"]
                .fillna("missing")
                .value_counts()
                .to_dict()
                if "punteggio_source" in train_df.columns
                else {},
            },
        }
    )

    # Feature importance (from Random Forest)
    if "rf" in models:
        feature_importance = dict(
            zip(feature_cols, models["rf"].feature_importances_.astype(float))
        )
        metrics["feature_importance"] = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        )
    else:
        metrics["feature_importance"] = {}

    # Add stacking metrics if available
    if stacking_metrics:
        metrics["stacking_validation"] = stacking_metrics

    return models, metrics


def save_models(models: dict, metrics: dict | None = None):
    """Save trained models and ensemble metadata to disk."""
    from pathlib import Path

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        if name == "stacking":
            # Save stacking model using dedicated function
            stacking_path = save_stacking_model(model, models_dir / "stacking_model")
            print(f"  {name} saved to: {stacking_path}")
        else:
            path = models_dir / f"{name}_model.pkl"
            joblib.dump(model, path)
            print(f"  {name} saved to: {path}")

    # Save ensemble weights, year stats, and selected features
    if metrics:
        category_classifier_metrics = metrics.get("category_classifier")
        meta = {
            "ensemble_weights": metrics.get("ensemble_weights", {}),
            "ensemble_strategy": metrics.get("ensemble_strategy", "weighted"),
            "ensemble_candidate_metrics": metrics.get("ensemble_candidate_metrics", {}),
            "year_stats": metrics.get("year_stats", {}),
            "selected_features": metrics.get("selected_features", []),
            "feature_fill_values": metrics.get("feature_fill_values", {}),
            "winsorization_bounds": metrics.get("winsorization_bounds", {}),
            "category_classifier_features": metrics.get("category_classifier_features", []),
            "category_classifier_feature_fill_values": metrics.get(
                "category_classifier_feature_fill_values", {}
            ),
            "category_classifier_winsorization_bounds": metrics.get(
                "category_classifier_winsorization_bounds", {}
            ),
            "category_feature_selection_seed": metrics.get("category_feature_selection_seed"),
            "score_thresholds": metrics.get("score_thresholds", {}),
            "category_threshold_strategy": metrics.get("category_threshold_strategy"),
            "category_threshold_strategies": metrics.get("category_threshold_strategies"),
            "quality_checks": metrics.get("quality_checks"),
            "sample_weighting": metrics.get("sample_weighting"),
            "category_classifier_diagnostics": metrics.get("category_classifier_diagnostics"),
            "category_classifier_model": (
                category_classifier_metrics.get("model_name")
                if isinstance(category_classifier_metrics, dict)
                else None
            ),
            "category_classifier_metrics": (
                category_classifier_metrics
                if isinstance(category_classifier_metrics, dict)
                else None
            ),
            "stacking_validation": metrics.get("stacking_validation"),
        }
        meta_path = models_dir / "ensemble_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  Ensemble metadata saved to: {meta_path}")

        # Save stacking metadata separately if available
        if metrics.get("stacking_validation"):
            save_stacking_metadata(
                metrics["stacking_validation"],
                metrics["selected_features"],
                models_dir,
            )
            print(f"  Stacking metadata saved to: {models_dir / 'stacking_meta.json'}")


def load_models(load_stacking: bool = False) -> tuple[dict | None, dict | None]:
    """
    Load trained models and ensemble metadata from disk.

    Args:
        load_stacking: If True, also load the stacking ensemble model

    Returns:
        Tuple of (models_dict, metadata_dict)
    """
    from pathlib import Path

    models_dir = Path(__file__).parent / "models"
    model_names = ["rf", "gb", "ridge", "xgb", "lgbm", "category_classifier"]
    models = {}

    for name in model_names:
        path = models_dir / f"{name}_model.pkl"
        try:
            models[name] = joblib.load(path)
        except (OSError, FileNotFoundError, ModuleNotFoundError, ImportError):
            pass
        except Exception:
            pass

    # Load stacking model if requested
    if load_stacking:
        stacking_model = load_stacking_model(models_dir / "stacking_model.pkl")
        if stacking_model is not None:
            models["stacking"] = stacking_model

    if not models:
        return None, None

    # Load ensemble metadata
    meta_path = models_dir / "ensemble_meta.json"
    meta = None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, FileNotFoundError, json.JSONDecodeError):
        pass

    return models, meta


if __name__ == "__main__":
    import os

    os.makedirs("backend/ml/models", exist_ok=True)

    print("=" * 70)
    print("FantaSanremo 2026 - ML Model Training")
    print("   Training on REAL historical data (2020-2025)")
    print("=" * 70)

    models, metrics = train_models()

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    print("\nModel Performance (Cross-Validation):")
    for model_name in ["random_forest", "gradient_boosting", "ridge", "xgboost", "lightgbm"]:
        if model_name in metrics:
            m = metrics[model_name]
            mae_str = f"MAE = {m['mae_cv']:.2f} +/- {m['mae_std']:.2f}"

            # Add enhanced metrics if available
            extra_metrics = []
            if "r2" in m:
                extra_metrics.append(f"R²={m['r2']:.3f}")
            if "rmse" in m:
                extra_metrics.append(f"RMSE={m['rmse']:.3f}")
            if "mape" in m:
                extra_metrics.append(f"MAPE={m['mape']:.1f}%")
            if "directional_accuracy" in m:
                extra_metrics.append(f"DirAcc={m['directional_accuracy']:.1%}")

            if extra_metrics:
                metrics_str = ", ".join(extra_metrics)
                print(f"   {model_name:20s}: {mae_str}, {metrics_str}")
            else:
                print(f"   {model_name:20s}: {mae_str}")

    print("\nEnsemble Weights:")
    for name, weight in metrics.get("ensemble_weights", {}).items():
        print(f"   {name}: {weight:.3f}")

    if metrics.get("category_classifier"):
        cls = metrics["category_classifier"]
        print("\nDedicated Category Classifier:")
        print(f"   Model:                     {cls.get('model_name')}")
        print(f"   Category accuracy:         {cls.get('category_accuracy', float('nan')):.3f}")
        print(f"   Category macro-F1:         {cls.get('category_macro_f1', float('nan')):.3f}")
        print(
            "   Category balanced-acc:     "
            f"{cls.get('category_balanced_accuracy', float('nan')):.3f}"
        )

    print("\nData Statistics:")
    print(f"   Total artists:             {metrics['data_stats']['total_artists']}")
    print(f"   Artists with history:       {metrics['data_stats']['artists_with_history']}")
    print(f"   Debuttanti (no history):    {metrics['data_stats']['debuttanti']}")
    print(f"   Historical records used:    {metrics['data_stats']['total_records']}")

    print("\nModel Configuration:")
    print(f"   Features used:              {metrics['random_forest']['features']}")
    print(f"   Training samples:           {metrics['random_forest']['samples']}")

    print("\nTop 5 Most Important Features:")
    for i, (feat, importance) in enumerate(list(metrics["feature_importance"].items())[:5], 1):
        bar = "#" * int(importance * 50)
        print(f"   {i}. {feat:30s} {importance:.3f}  {bar}")

    print("\n" + "=" * 70)
    save_models(models, metrics)
    print("\nModels saved to backend/ml/models/")
    print("Training complete!\n")
