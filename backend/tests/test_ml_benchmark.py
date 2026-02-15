from backend.ml.benchmark import evaluate_go_no_go


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
