from backend.ml.benchmark import evaluate_go_no_go, summarize_metrics


def test_go_no_go_approves_when_rules_are_met():
    baseline = {
        "regression": {"best_mae_cv": 0.60, "ensemble_rmse": 90.0},
        "classification": {"macro_f1": 0.45, "balanced_accuracy": 0.48},
    }
    candidate = {
        "regression": {"best_mae_cv": 0.57, "ensemble_rmse": 84.0},
        "classification": {"macro_f1": 0.50, "balanced_accuracy": 0.47},
    }

    out = evaluate_go_no_go(baseline, candidate)

    assert out["approved"] is True
    assert out["rules"]["regression_pass"] is True
    assert out["rules"]["classification_pass"] is True
    assert out["rules"]["stability_pass"] is True


def test_go_no_go_rejects_when_macro_f1_does_not_improve():
    baseline = {
        "regression": {"best_mae_cv": 0.60, "ensemble_rmse": 90.0},
        "classification": {"macro_f1": 0.45, "balanced_accuracy": 0.48},
    }
    candidate = {
        "regression": {"best_mae_cv": 0.56, "ensemble_rmse": 85.0},
        "classification": {"macro_f1": 0.46, "balanced_accuracy": 0.48},
    }

    out = evaluate_go_no_go(baseline, candidate)

    assert out["approved"] is False
    assert out["rules"]["classification_pass"] is False


def test_summarize_metrics_picks_dynamic_best_regressor():
    metrics = {
        "random_forest": {"mae_cv": 0.62, "r2": 0.1},
        "tabicl_v2": {"mae_cv": 0.55, "r2": 0.2},
        "ensemble": {"rmse": 81.0, "r2": 0.3, "mape": 12.0, "category_macro_f1": 0.5},
        "category_classifier": {
            "model_name": "rf_classifier",
            "category_macro_f1": 0.5,
            "category_balanced_accuracy": 0.51,
            "category_accuracy": 0.52,
            "features": 12,
        },
        "selected_features": ["f1", "f2"],
    }

    summary = summarize_metrics(metrics, label="candidate-tabular")

    assert summary["regression"]["best_model"] == "tabicl_v2"
    assert summary["regression"]["best_mae_cv"] == 0.55
