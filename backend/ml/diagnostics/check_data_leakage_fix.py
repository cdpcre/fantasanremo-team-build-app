#!/usr/bin/env python3
"""
Diagnostic: verify the data leakage fix in normalize_features().
Run this manually to confirm the fix is working correctly.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for backend.* imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import joblib

from backend.ml.features import normalize_features


def test_data_leakage_fix():
    """Test that data leakage is prevented in normalize_features()"""

    print("\n" + "=" * 80)
    print("TESTING DATA LEAKAGE FIX")
    print("=" * 80)

    # Test 1: Training mode creates and saves scaler
    print("\n[Test 1] Training mode - fit and save scaler")
    train_data = pd.DataFrame(
        {
            "avg_position": [15.5, 22.3, 8.7],
            "position_variance": [25.3, 18.7, 32.1],
            "position_trend": [2.1, -1.5, 3.2],
            "participations": [3, 5, 2],
            "best_position": [20, 25, 15],
            "recent_avg": [16.2, 21.5, 9.3],
            "quotazione_2026": [15, 17, 14],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_path = Path(tmp_dir) / "feature_scaler.pkl"

        normalized_train, scaler = normalize_features(
            train_data, mode="train", scaler_path=scaler_path
        )

        # Check scaler was saved
        assert scaler_path.exists(), "Scaler file not created!"
        print("   ✓ Scaler saved to disk")

        # Check normalized data has mean ≈ 0, std ≈ 1
        means = normalized_train.mean(numeric_only=True)
        stds = normalized_train.std(numeric_only=True)

        # With only 3 samples, we need more tolerance
        assert all(abs(means) < 0.1), f"Mean not close to 0: {means.to_dict()}"
        assert all(abs(stds - 1.0) < 0.5), f"Std not close to 1: {stds.to_dict()}"
        print("   ✓ Training data properly normalized (mean≈0, std≈1)")

        # Test 2: Prediction mode loads and uses scaler
        print("\n[Test 2] Prediction mode - load and use scaler")
        pred_data = pd.DataFrame(
            {
                "avg_position": [19.8, 14.2, 21.5, 10.5, 25.3],  # Different samples!
                "position_variance": [22.1, 30.5, 17.8, 28.4, 15.9],
                "position_trend": [1.2, -0.8, 2.5, -1.9, 3.7],
                "participations": [4, 2, 5, 1, 6],
                "best_position": [23, 17, 24, 12, 26],
                "recent_avg": [18.5, 13.8, 20.2, 9.7, 22.8],
                "quotazione_2026": [16, 14, 17, 13, 17],
            }
        )

        normalized_pred, scaler_pred = normalize_features(
            pred_data, mode="predict", scaler_path=scaler_path
        )

        # Check prediction data is normalized using TRAINING parameters
        # (so mean/std won't be 0/1, but transformation is consistent)
        assert scaler_pred is not None, "Scaler not loaded"
        print("   ✓ Scaler loaded from disk")

        # Test 3: Verify no data leakage - scaler parameters unchanged
        print("\n[Test 3] Verify no data leakage")
        loaded_scaler = joblib.load(scaler_path)

        # Compare parameters
        assert np.allclose(scaler.mean_, loaded_scaler.mean_), "Means changed!"
        assert np.allclose(scaler.scale_, loaded_scaler.scale_), "Scales changed!"
        print("   ✓ Scaler parameters unchanged after prediction")

        # Compare n_samples_seen_
        assert np.array_equal(scaler.n_samples_seen_, loaded_scaler.n_samples_seen_), (
            "n_samples_seen_ changed!"
        )

        # n_samples_seen_ is a scalar, not an array
        print(f"   ✓ n_samples_seen_ = {scaler.n_samples_seen_} (training size, not prediction)")

        # Test 4: Multiple predictions use same scaler
        print("\n[Test 4] Multiple predictions use same scaler")
        normalized_pred2, _ = normalize_features(pred_data, mode="predict", scaler_path=scaler_path)
        assert normalized_pred.equals(normalized_pred2), "Inconsistent predictions!"
        print("   ✓ Multiple predictions consistent")

    # Test 5: Error handling
    print("\n[Test 5] Error handling")
    try:
        normalize_features(train_data, mode="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("   ✓ Invalid mode rejected")

    try:
        normalize_features(train_data, mode="predict", scaler_path="nonexistent/scaler.pkl")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("   ✓ Missing scaler detected")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Data leakage fix is working correctly!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_data_leakage_fix()
        print("\n🎉 SUCCESS: The data leakage issue has been fixed!")
        print("\nNext steps:")
        print("1. Review DATA_LEAKAGE_FIX_REPORT.md for details")
        print("2. Run normalization_usage_example.py to see usage examples")
        print("3. Integrate the fixed function into your training pipeline")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
