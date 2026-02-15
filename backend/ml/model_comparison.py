"""Model comparison utilities using A/B testing framework.

This module provides functions to compare different model versions
using statistical tests and bootstrap analysis.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .ab_testing import generate_comparison_report, save_comparison_report
from .feature_builder import FeatureBuilder
from .train import load_models


def compare_saved_models(
    baseline_path: str | Path | None = None,
    new_path: str | Path | None = None,
    comparison_name: str = "comparison",
) -> dict | None:
    """Compare two saved model versions.

    Args:
        baseline_path: Path to baseline models directory (default: backend/ml/models)
        new_path: Path to new models directory (default: backend/ml/models with _new suffix)
        comparison_name: Name for the comparison report

    Returns:
        Comparison report dict or None if comparison failed

    Example:
        >>> report = compare_saved_models()
        >>> print(report['conclusion'])
    """
    if baseline_path is None:
        baseline_path = Path(__file__).parent / "models"
    else:
        baseline_path = Path(baseline_path)

    if new_path is None:
        new_path = baseline_path
    else:
        new_path = Path(new_path)

    # Load baseline models
    model_names = ["rf", "gb", "ridge", "xgb", "lgbm"]
    models_a = {}
    for name in model_names:
        path = baseline_path / f"{name}_model.pkl"
        try:
            models_a[name] = joblib.load(path)
        except (OSError, FileNotFoundError):
            print(f"Warning: Could not load baseline model {name} from {path}")
            return None

    # Load new models
    models_b = {}
    for name in model_names:
        path = new_path / f"{name}_model_new.pkl"
        try:
            models_b[name] = joblib.load(path)
        except (OSError, FileNotFoundError):
            print(f"Warning: Could not load new model {name} from {path}")
            return None

    if not models_a or not models_b:
        print("Error: Could not load models")
        return None

    # Get validation data
    builder = FeatureBuilder()
    sources = builder.load_sources()
    full_df = builder.build_training_frame(sources)

    val_years = builder.config.validation_years
    splits = builder.split_by_years(full_df, [], val_years)

    if not splits.get("val"):
        print("Error: No validation data available")
        return None

    val_dfs = list(splits["val"].values())
    if not val_dfs:
        print("Error: No validation data available")
        return None

    val_df = pd.concat(val_dfs, ignore_index=True)

    if val_df.empty or "punteggio_reale" not in val_df.columns:
        print("Error: Invalid validation data")
        return None

    # Get feature columns from metadata
    _, meta = load_models()
    if meta and meta.get("selected_features"):
        feature_cols = [c for c in meta["selected_features"] if c in val_df.columns]
    else:
        feature_cols = builder.get_feature_columns(val_df)

    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["punteggio_reale"].values

    # Generate predictions
    weights_a = meta.get("ensemble_weights", {}) if meta else {}
    if not weights_a:
        weights_a = {name: 1.0 / len(models_a) for name in models_a}

    weights_b = weights_a

    pred_a = np.zeros(len(X_val))
    for name, model in models_a.items():
        if name in weights_a:
            pred_a += weights_a[name] * model.predict(X_val)

    pred_b = np.zeros(len(X_val))
    for name, model in models_b.items():
        if name in weights_b:
            pred_b += weights_b[name] * model.predict(X_val)

    # Generate comparison report
    report = generate_comparison_report(
        y_val,
        pred_a,
        pred_b,
        model_a_name="Baseline Model",
        model_b_name="New Model",
        n_bootstrap=1000,
        random_state=42,
    )

    # Save report
    output_path = Path(__file__).parent / "models" / f"{comparison_name}_report.json"
    report_path = save_comparison_report(report, output_path)

    # Print results
    separator = "=" * 70
    print(f"\n{separator}")
    print("MODEL COMPARISON (A/B Testing)")
    print(separator)
    print(f"\nBaseline vs New Model on validation data ({len(y_val)} samples)")
    print(f"Report saved to: {report_path}")

    print("\nStatistical Tests:")
    wilcoxon = report["statistical_tests"]["wilcoxon"]
    print(f"  Wilcoxon p-value: {wilcoxon['p_value']:.4f}")
    print(f"  Significant: {wilcoxon['significant']}")
    print(f"  Winner: {wilcoxon['winner']}")

    print("\nBootstrap Analysis:")
    bootstrap = report["bootstrap"]
    print(f"  Mean MAE difference: {bootstrap['mean_mae_difference']:.2f}")
    print(f"  95% CI: [{bootstrap['ci_95_lower']:.2f}, {bootstrap['ci_95_upper']:.2f}]")
    print(f"  Improvement probability: {bootstrap['improvement_probability']:.1%}")

    print(f"\nConclusion: {report['conclusion']}")
    print(separator)

    return report


def compare_with_predictions(
    y_true: np.ndarray,
    pred_baseline: np.ndarray,
    pred_new: np.ndarray,
    comparison_name: str = "comparison",
) -> dict:
    """Compare models using pre-computed predictions.

    Args:
        y_true: True target values
        pred_baseline: Predictions from baseline model
        pred_new: Predictions from new model
        comparison_name: Name for the comparison report

    Returns:
        Comparison report dict

    Example:
        >>> import numpy as np
        >>> y_true = np.array([100, 200, 150, 180, 220])
        >>> pred_baseline = np.array([110, 190, 160, 170, 210])
        >>> pred_new = np.array([105, 195, 155, 175, 215])
        >>> report = compare_with_predictions(y_true, pred_baseline, pred_new)
    """
    report = generate_comparison_report(
        y_true,
        pred_baseline,
        pred_new,
        model_a_name="Baseline Model",
        model_b_name="New Model",
        n_bootstrap=1000,
        random_state=42,
    )

    # Save report
    output_path = Path(__file__).parent / "models" / f"{comparison_name}_report.json"
    save_comparison_report(report, output_path)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare ML model versions")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline models directory",
    )
    parser.add_argument(
        "--new",
        type=str,
        default=None,
        help="Path to new models directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="comparison",
        help="Name for the comparison report",
    )
    args = parser.parse_args()

    report = compare_saved_models(
        baseline_path=args.baseline,
        new_path=args.new,
        comparison_name=args.name,
    )

    if report is None:
        print("\nComparison failed. Check that models are trained and available.")
        exit(1)
