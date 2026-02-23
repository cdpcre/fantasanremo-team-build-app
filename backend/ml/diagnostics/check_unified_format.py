"""
Test script to verify ML pipeline works with unified storico format.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import json

from backend.data_pipeline.config import get_config
from backend.ml.data_preparation import MLDataPreparation


def test_unified_format_loading():
    """Test that unified format loads correctly."""
    config = get_config()

    print("=" * 60)
    print("Testing Unified Storico Format Loading")
    print("=" * 60)

    # Test 1: Check file exists
    unified_path = config.data_dir / "storico_fantasanremo_unified.json"
    assert unified_path.exists(), f"Unified file not found at {unified_path}"
    print(f"✓ Unified file exists at {unified_path}")

    # Test 2: Load and verify structure
    with open(unified_path) as f:
        unified_data = json.load(f)

    expected_keys = ["festival_edizioni", "artisti_storici", "artisti_2026", "metadata"]
    for key in expected_keys:
        assert key in unified_data, f"Missing key: {key}"
        print(f"✓ Key '{key}' found in unified file")

    # Test 3: Verify data counts
    festival_count = len(unified_data.get("festival_edizioni", {}))
    artisti_storici_count = len(unified_data.get("artisti_storici", {}))
    artisti_2026_count = len(unified_data.get("artisti_2026", []))

    print(f"\n✓ Festival editions: {festival_count}")
    print(f"✓ Storici artists: {artisti_storici_count}")
    print(f"✓ 2026 artists: {artisti_2026_count}")

    # Test 4: Test MLDataPreparation
    print("\n" + "=" * 60)
    print("Testing MLDataPreparation")
    print("=" * 60)

    preparator = MLDataPreparation(config)
    sources = preparator.load_all_sources()

    print(f"\n✓ Loaded {len(sources)} data sources:")
    for source_name in sources.keys():
        print(f"  - {source_name}")

    # Test 5: Check for unified format in sources
    if "storico_unified" in sources:
        print("\n✓ Unified format loaded from storico_unified")
    else:
        print("\n⚠ Unified format not loaded (storico_unified missing)")

    # Test 6: Verify no KeyError on legacy keys
    print("\n" + "=" * 60)
    print("Testing No KeyError on Legacy Keys")
    print("=" * 60)

    try:
        # This should NOT raise KeyError for "statistiche_artisti_2026" or "albo_oro"
        storico = sources.get("storico", {})
        if isinstance(storico, dict):
            # Check if it has legacy structure
            has_legacy_key = "statistiche_artisti_2026" in storico
            print(f"✓ 'statistiche_artisti_2026' key present: {has_legacy_key}")

        # Check for unified file usage
        unified_path = config.data_dir / "storico_fantasanremo_unified.json"
        if unified_path.exists():
            print("✓ Unified file exists, should be used for loading")

        # Test that we can build historical dataframe without errors
        historical_df = preparator._build_historical_dataframe(sources)
        print(f"\n✓ Built historical DataFrame: {len(historical_df)} records")
        if not historical_df.empty:
            print(f"  Columns: {list(historical_df.columns)}")
            print(f"  Years: {sorted(historical_df['anno'].unique())}")

    except KeyError as e:
        print(f"\n✗ KeyError encountered: {e}")
        print("  This indicates the code is looking for legacy keys that don't exist")
        raise AssertionError("KeyError encountered while building historical data") from e
    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
        raise AssertionError("Unexpected error while building historical data") from e

    # Test 7: Create training dataset
    print("\n" + "=" * 60)
    print("Testing Training Dataset Creation")
    print("=" * 60)

    try:
        X_train, y_train, X_val_2023, y_val_2023, X_val_2025, y_val_2025, X_2026 = (
            preparator.create_training_dataset(sources)
        )

        print(f"\n✓ Training samples: {len(X_train)}")
        print(f"✓ Validation 2023 samples: {len(X_val_2023)}")
        print(f"✓ Validation 2025 samples: {len(X_val_2025)}")
        print(f"✓ 2026 prediction samples: {len(X_2026)}")

        if not X_train.empty:
            print(f"\n✓ Feature columns: {list(X_train.columns)}")

    except Exception as e:
        print(f"\n✗ Error creating training dataset: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise AssertionError("Error creating training dataset") from e

    print("\n" + "=" * 60)
    print("All Tests Passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_unified_format_loading()
        sys.exit(0)
    except AssertionError as exc:
        print(f"\n❌ TEST FAILED: {exc}")
        sys.exit(1)
