"""A/B testing framework for model comparison.

Provides statistical tests to compare model versions, metrics, and predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two model versions."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    winner: str | None  # "A", "B", or "tie"
    effect_size: float | None = None
    interpretation: str = ""


@dataclass
class MetricsComparison:
    """Comparison of metrics between two model versions."""

    metrics_a: dict[str, float]
    metrics_b: dict[str, float]
    differences: dict[str, float] = field(default_factory=dict)
    relative_differences: dict[str, float] = field(default_factory=dict)
    improvements: dict[str, str] = field(default_factory=dict)  # "A", "B", or "tie"


@dataclass
class BootstrapResult:
    """Result of bootstrap comparison."""

    mae_difference_samples: np.ndarray
    mean_difference: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    zero_in_ci: bool
    improvement_probability: float


def compare_model_versions(
    y_true: np.ndarray | pd.Series,
    pred_a: np.ndarray | pd.Series,
    pred_b: np.ndarray | pd.Series,
    test: str = "wilcoxon",
    alpha: float = 0.05,
) -> ComparisonResult:
    """
    Statistical comparison of two model versions.

    Performs paired statistical tests to determine if model B is significantly
    different from model A.

    Args:
        y_true: True target values
        pred_a: Predictions from model A (baseline)
        pred_b: Predictions from model B (challenger)
        test: Statistical test to use ("wilcoxon", "ttest", "sign")
        alpha: Significance level (default: 0.05)

    Returns:
        ComparisonResult with test statistics and winner

    Raises:
        ValueError: If arrays have different lengths or unknown test specified

    Examples:
        >>> y_true = np.array([100, 200, 150, 180, 220])
        >>> pred_a = np.array([110, 190, 160, 170, 210])
        >>> pred_b = np.array([105, 195, 155, 175, 215])
        >>> result = compare_model_versions(y_true, pred_a, pred_b)
        >>> print(f"Winner: {result.winner}, p-value: {result.p_value:.4f}")
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    # Validate inputs
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("All input arrays must have the same length")

    if len(y_true) < 2:
        raise ValueError("Need at least 2 samples for comparison")

    # Calculate errors
    errors_a = np.abs(y_true - pred_a)
    errors_b = np.abs(y_true - pred_b)

    # Perform statistical test
    if test == "wilcoxon":
        # Wilcoxon signed-rank test (non-parametric)
        statistic, p_value = stats.wilcoxon(errors_a, errors_b)
        test_name = "Wilcoxon signed-rank test"
    elif test == "ttest":
        # Paired t-test (parametric, assumes normal distribution)
        statistic, p_value = stats.ttest_rel(errors_a, errors_b)
        test_name = "Paired t-test"
    elif test == "sign":
        # Sign test (simpler non-parametric)
        n_pos = np.sum(errors_a > errors_b)
        n_neg = np.sum(errors_a < errors_b)
        n = n_pos + n_neg
        # Binomial test against null of equal probability
        p_value = 2 * min(stats.binom.cdf(n_pos, n, 0.5), stats.binom.cdf(n_neg, n, 0.5))
        statistic = n_pos
        test_name = "Sign test"
    else:
        raise ValueError(f"Unknown test: {test}. Use 'wilcoxon', 'ttest', or 'sign'")

    # Calculate effect size (Cohen's d for paired samples)
    diff = errors_a - errors_b
    effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0

    # Determine winner and significance
    significant = bool(p_value < alpha)
    mean_error_a = np.mean(errors_a)
    mean_error_b = np.mean(errors_b)

    if significant:
        if mean_error_a < mean_error_b:
            winner = "A"
        elif mean_error_b < mean_error_a:
            winner = "B"
        else:
            winner = "tie"
    else:
        winner = None

    # Interpret results
    if significant:
        if winner == "B":
            interpretation = (
                f"Model B significantly outperforms Model A "
                f"(p={p_value:.4f}, effect size={effect_size:.3f})"
            )
        else:
            interpretation = (
                f"Model A significantly outperforms Model B "
                f"(p={p_value:.4f}, effect size={effect_size:.3f})"
            )
    else:
        interpretation = (
            f"No significant difference found (p={p_value:.4f} > {alpha}). "
            f"Effect size: {effect_size:.3f}"
        )

    return ComparisonResult(
        test_name=test_name,
        statistic=float(statistic),
        p_value=float(p_value),
        significant=significant,
        winner=winner,
        effect_size=float(effect_size),
        interpretation=interpretation,
    )


def compare_metrics(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    threshold: float = 0.01,
) -> MetricsComparison:
    """
    Compare metrics between two model versions.

    Calculates absolute and relative differences between metrics and determines
    which model performs better for each metric.

    Args:
        metrics_a: Metrics from model A (e.g., {"mae": 50.0, "r2": 0.8})
        metrics_b: Metrics from model B
        threshold: Minimum relative difference to consider as improvement (default: 1%)

    Returns:
        MetricsComparison with differences and improvements

    Examples:
        >>> metrics_a = {"mae": 50.0, "r2": 0.8, "rmse": 70.0}
        >>> metrics_b = {"mae": 45.0, "r2": 0.85, "rmse": 65.0}
        >>> comparison = compare_metrics(metrics_a, metrics_b)
        >>> print(comparison.improvements["mae"])  # "B" (model B is better)
    """
    # Get common metrics
    common_keys = set(metrics_a.keys()) & set(metrics_b.keys())

    differences: dict[str, float] = {}
    relative_differences: dict[str, float] = {}
    improvements: dict[str, str] = {}

    for key in common_keys:
        val_a = metrics_a[key]
        val_b = metrics_b[key]

        # Skip if values are identical or zero
        if val_a == val_b:
            differences[key] = 0.0
            relative_differences[key] = 0.0
            improvements[key] = "tie"
            continue

        # Calculate absolute and relative differences
        diff = val_b - val_a
        rel_diff = diff / abs(val_a) if val_a != 0 else float("inf" if diff != 0 else 0)

        differences[key] = diff
        relative_differences[key] = rel_diff

        # Determine winner (lower is better for errors, higher for scores)
        if key.lower() in ["mae", "mse", "rmse", "mape"]:
            # Error metrics: lower is better
            if rel_diff < -threshold:
                improvements[key] = "B"
            elif rel_diff > threshold:
                improvements[key] = "A"
            else:
                improvements[key] = "tie"
        else:
            # Score metrics (R², accuracy, etc.): higher is better
            if rel_diff > threshold:
                improvements[key] = "B"
            elif rel_diff < -threshold:
                improvements[key] = "A"
            else:
                improvements[key] = "tie"

    return MetricsComparison(
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        differences=differences,
        relative_differences=relative_differences,
        improvements=improvements,
    )


def bootstrap_comparison(
    y_true: np.ndarray | pd.Series,
    pred_a: np.ndarray | pd.Series,
    pred_b: np.ndarray | pd.Series,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int | None = None,
) -> BootstrapResult:
    """
    Bootstrap comparison of predictions.

    Uses bootstrap resampling to estimate confidence intervals for the difference
    in MAE between two models. This provides a robust way to assess model
    performance without distributional assumptions.

    Args:
        y_true: True target values
        pred_a: Predictions from model A (baseline)
        pred_b: Predictions from model B (challenger)
        n_bootstrap: Number of bootstrap iterations (default: 1000)
        ci_level: Confidence interval level (default: 0.95)
        random_state: Random seed for reproducibility

    Returns:
        BootstrapResult with confidence intervals and improvement probability

    Examples:
        >>> result = bootstrap_comparison(y_true, pred_a, pred_b, n_bootstrap=10000)
        >>> print(f"95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
        >>> print(f"Improvement prob: {result.improvement_probability:.1%}")
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    n_samples = len(y_true)

    if random_state is not None:
        np.random.seed(random_state)

    # Bootstrap samples
    mae_diff_samples = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Calculate MAE for this bootstrap sample
        mae_a = np.mean(np.abs(y_true[indices] - pred_a[indices]))
        mae_b = np.mean(np.abs(y_true[indices] - pred_b[indices]))

        # Store difference (positive = B is better, lower MAE)
        mae_diff_samples[i] = mae_a - mae_b

    # Calculate statistics
    mean_diff = float(np.mean(mae_diff_samples))
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(mae_diff_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(mae_diff_samples, 100 * (1 - alpha / 2)))
    ci_width = ci_upper - ci_lower
    zero_in_ci = ci_lower <= 0 <= ci_upper

    # Probability that B is better (positive difference)
    improvement_probability = float(np.mean(mae_diff_samples > 0))

    return BootstrapResult(
        mae_difference_samples=mae_diff_samples,
        mean_difference=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_width,
        zero_in_ci=zero_in_ci,
        improvement_probability=improvement_probability,
    )


def calculate_prediction_intervals(
    y_true: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals using bootstrap.

    Args:
        y_true: True target values
        predictions: Model predictions
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence interval level
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (lower_bounds, upper_bounds) arrays
    """
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    residuals = y_true - predictions

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(y_true)
    predictions_boot = np.zeros((n_bootstrap, n_samples))

    for i in range(n_bootstrap):
        # Bootstrap residuals
        boot_residuals = np.random.choice(residuals, size=n_samples, replace=True)
        predictions_boot[i] = predictions + boot_residuals

    alpha = 1 - ci_level
    lower = np.percentile(predictions_boot, 100 * alpha / 2, axis=0)
    upper = np.percentile(predictions_boot, 100 * (1 - alpha / 2), axis=0)

    return lower, upper


def generate_comparison_report(
    y_true: np.ndarray | pd.Series,
    pred_a: np.ndarray | pd.Series,
    pred_b: np.ndarray | pd.Series,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    n_bootstrap: int = 1000,
    random_state: int | None = 42,
) -> dict[str, Any]:
    """
    Generate comprehensive A/B testing comparison report.

    Runs multiple statistical tests and generates a complete comparison report
    including metrics, statistical significance, and bootstrap confidence intervals.

    Args:
        y_true: True target values
        pred_a: Predictions from model A
        pred_b: Predictions from model B
        model_a_name: Name for model A in report
        model_b_name: Name for model B in report
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing complete comparison report

    Examples:
        >>> report = generate_comparison_report(y_true, pred_a, pred_b,
        ...                                      model_a_name="Baseline",
        ...                                      model_b_name="Improved")
        >>> print(json.dumps(report, indent=2))
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Calculate metrics for both models
    metrics_a = {
        "mae": mean_absolute_error(y_true, pred_a),
        "rmse": np.sqrt(mean_squared_error(y_true, pred_a)),
        "r2": r2_score(y_true, pred_a),
    }

    metrics_b = {
        "mae": mean_absolute_error(y_true, pred_b),
        "rmse": np.sqrt(mean_squared_error(y_true, pred_b)),
        "r2": r2_score(y_true, pred_b),
    }

    # Metrics comparison
    metrics_comp = compare_metrics(metrics_a, metrics_b)

    # Statistical tests (Wilcoxon and t-test)
    wilcoxon_result = compare_model_versions(y_true, pred_a, pred_b, test="wilcoxon")
    ttest_result = compare_model_versions(y_true, pred_a, pred_b, test="ttest")

    # Bootstrap comparison
    bootstrap_result = bootstrap_comparison(
        y_true, pred_a, pred_b, n_bootstrap=n_bootstrap, random_state=random_state
    )

    # Compile report
    report = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "metrics": {
            model_a_name: metrics_a,
            model_b_name: metrics_b,
        },
        "metrics_comparison": {
            "improvements": metrics_comp.improvements,
            "absolute_differences": metrics_comp.differences,
            "relative_differences": metrics_comp.relative_differences,
        },
        "statistical_tests": {
            "wilcoxon": {
                "test_name": wilcoxon_result.test_name,
                "statistic": wilcoxon_result.statistic,
                "p_value": wilcoxon_result.p_value,
                "significant": wilcoxon_result.significant,
                "winner": wilcoxon_result.winner,
                "effect_size": wilcoxon_result.effect_size,
                "interpretation": wilcoxon_result.interpretation,
            },
            "ttest": {
                "test_name": ttest_result.test_name,
                "statistic": ttest_result.statistic,
                "p_value": ttest_result.p_value,
                "significant": ttest_result.significant,
                "winner": ttest_result.winner,
                "effect_size": ttest_result.effect_size,
                "interpretation": ttest_result.interpretation,
            },
        },
        "bootstrap": {
            "mean_mae_difference": bootstrap_result.mean_difference,
            "ci_95_lower": bootstrap_result.ci_lower,
            "ci_95_upper": bootstrap_result.ci_upper,
            "ci_width": bootstrap_result.ci_width,
            "zero_in_ci": bootstrap_result.zero_in_ci,
            "improvement_probability": bootstrap_result.improvement_probability,
        },
        "conclusion": _generate_conclusion(
            wilcoxon_result, ttest_result, bootstrap_result, model_a_name, model_b_name
        ),
    }

    return report


def _generate_conclusion(
    wilcoxon: ComparisonResult,
    ttest: ComparisonResult,
    bootstrap: BootstrapResult,
    model_a_name: str,
    model_b_name: str,
) -> str:
    """Generate conclusion text from comparison results."""
    lines = []

    # Check consistency of results
    significant_count = sum([wilcoxon.significant, ttest.significant])

    if significant_count >= 2:
        # Both tests agree
        if wilcoxon.winner == "B":
            lines.append(
                f"{model_b_name} significantly outperforms {model_a_name} "
                f"based on both Wilcoxon and t-test."
            )
        elif wilcoxon.winner == "A":
            lines.append(
                f"{model_a_name} significantly outperforms {model_b_name} "
                f"based on both Wilcoxon and t-test."
            )
        else:
            lines.append("Results are inconsistent between tests.")
    elif significant_count == 1:
        lines.append("Results are borderline - only one test shows significance.")
    else:
        lines.append("No significant difference found between models.")

    # Bootstrap interpretation
    if not bootstrap.zero_in_ci:
        direction = "lower" if bootstrap.mean_difference > 0 else "higher"
        lines.append(
            f"Bootstrap analysis confirms {model_b_name} has {direction} MAE "
            f"than {model_a_name} with 95% confidence."
        )
    else:
        lines.append(
            "Bootstrap analysis cannot conclude significant difference (95% CI includes zero)."
        )

    # Improvement probability
    if bootstrap.improvement_probability > 0.95:
        lines.append(
            f"There is a {bootstrap.improvement_probability:.1%} probability "
            f"that {model_b_name} outperforms {model_a_name}."
        )

    return " ".join(lines)


def save_comparison_report(
    report: dict[str, Any],
    output_path: Path | str | None = None,
) -> Path:
    """
    Save comparison report to JSON file.

    Args:
        report: Comparison report dictionary
        output_path: Path to save report (default: models/comparison_report.json)

    Returns:
        Path to saved report file
    """
    if output_path is None:
        output_path = Path(__file__).parent / "models" / "comparison_report.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path


def load_comparison_report(
    input_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """
    Load comparison report from JSON file.

    Args:
        input_path: Path to report file (default: models/comparison_report.json)

    Returns:
        Report dictionary or None if file not found
    """
    if input_path is None:
        input_path = Path(__file__).parent / "models" / "comparison_report.json"
    else:
        input_path = Path(input_path)

    try:
        with open(input_path) as f:
            return json.load(f)
    except (OSError, FileNotFoundError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    # Example usage and testing
    print("A/B Testing Framework for Model Comparison")
    print("=" * 60)

    # Generate synthetic data for demonstration
    np.random.seed(42)
    n = 100

    # True values
    y_true = np.random.normal(200, 50, n)

    # Model A predictions (baseline - more error)
    pred_a = y_true + np.random.normal(0, 30, n)

    # Model B predictions (improved - less error)
    pred_b = y_true + np.random.normal(0, 20, n)

    print(f"\nGenerated {n} samples for testing")
    print(f"True value range: [{y_true.min():.1f}, {y_true.max():.1f}]")

    # Run statistical comparison
    print("\n" + "=" * 60)
    print("Statistical Tests")
    print("=" * 60)

    wilcoxon_result = compare_model_versions(y_true, pred_a, pred_b, test="wilcoxon")
    print("\nWilcoxon signed-rank test:")
    print(f"  Statistic: {wilcoxon_result.statistic:.2f}")
    print(f"  P-value: {wilcoxon_result.p_value:.4f}")
    print(f"  Significant: {wilcoxon_result.significant}")
    print(f"  Winner: {wilcoxon_result.winner}")
    print(f"  Effect size: {wilcoxon_result.effect_size:.3f}")

    # Bootstrap comparison
    print("\n" + "=" * 60)
    print("Bootstrap Analysis (1000 iterations)")
    print("=" * 60)

    bootstrap_result = bootstrap_comparison(y_true, pred_a, pred_b, n_bootstrap=1000)
    print(f"\nMean MAE difference (A - B): {bootstrap_result.mean_difference:.2f}")
    print(f"95% CI: [{bootstrap_result.ci_lower:.2f}, {bootstrap_result.ci_upper:.2f}]")
    print(f"CI width: {bootstrap_result.ci_width:.2f}")
    print(f"Zero in CI: {bootstrap_result.zero_in_ci}")
    print(f"Improvement probability: {bootstrap_result.improvement_probability:.1%}")

    # Generate full report
    print("\n" + "=" * 60)
    print("Full Comparison Report")
    print("=" * 60)

    report = generate_comparison_report(
        y_true,
        pred_a,
        pred_b,
        model_a_name="Baseline",
        model_b_name="Improved",
        n_bootstrap=1000,
    )

    print(f"\nConclusion: {report['conclusion']}")

    # Save report
    report_path = save_comparison_report(report)
    print(f"\nReport saved to: {report_path}")
