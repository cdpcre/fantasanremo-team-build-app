#!/usr/bin/env python
"""
Fantasanremo Data Pipeline CLI

Interfaccia CLI per eseguire la data pipeline.
"""

import argparse
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from backend.data_pipeline.config import (
    get_development_config,
    get_production_config,
    get_test_config,
    set_config,
)
from backend.data_pipeline.pipeline import run_pipeline
from backend.ml.predict import predict_2026, save_predictions
from backend.ml.train import save_models, train_models


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fantasanremo Data Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to check data
  uv run python scripts/run_pipeline.py --dry-run

  # Run full pipeline
  uv run python scripts/run_pipeline.py

  # Run specific steps only
  uv run python scripts/run_pipeline.py --steps fetch,validate

  # Run with verbose output
  uv run python scripts/run_pipeline.py --verbose

  # Run ML training + predictions
  uv run python scripts/run_pipeline.py --ml-training

  # Production mode
  uv run python scripts/run_pipeline.py --config prod
        """,
    )

    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated list of steps to run (fetch,validate,transform,load,generate_report)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Run pipeline without saving anything"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--config",
        choices=["dev", "prod", "test"],
        default="dev",
        help="Configuration environment (default: dev)",
    )

    parser.add_argument(
        "--ml-training",
        action="store_true",
        help="Run ML training + predictions instead of data pipeline",
    )

    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate data, don't load anything"
    )

    args = parser.parse_args()

    # Set configuration
    if args.config == "prod":
        config = get_production_config()
    elif args.config == "test":
        config = get_test_config()
    else:
        config = get_development_config()

    set_config(config)

    # Parse steps
    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]

    if args.validate_only:
        steps = ["fetch", "validate", "generate_report"]

    # Run appropriate pipeline
    if args.ml_training:
        print("🤖 Running ML Training + Prediction...")
        print("=" * 60)

        models, metrics = train_models()
        save_models(models, metrics)

        # Generate predictions for 2026
        try:
            with open(config.artisti_2026_path) as f:
                artisti_data = json.load(f).get("artisti", [])

            predictions = predict_2026(pd.DataFrame(artisti_data), pd.DataFrame())
            if predictions:
                pred_path = save_predictions(
                    predictions, output_path=config.models_dir / "predictions_2026.json"
                )
                print(f"\n✓ Predictions saved to: {pred_path}")
            else:
                print("\n⚠ WARNING: Prediction generation returned empty results")
        except Exception as exc:
            print(f"\n⚠ WARNING: Failed to generate predictions: {type(exc).__name__} - {exc}")

        # Print training summary
        print("\n" + "=" * 60)
        print("ML Training Results")
        print("=" * 60)
        print(f"Samples: {metrics['random_forest']['samples']}")
        print(f"Features: {metrics['random_forest']['features']}")
        print(
            f"MAE (RF): {metrics['random_forest']['mae_cv']:.2f} ± "
            f"{metrics['random_forest']['mae_std']:.2f}"
        )
        print(
            f"MAE (GB): {metrics['gradient_boosting']['mae_cv']:.2f} ± "
            f"{metrics['gradient_boosting']['mae_std']:.2f}"
        )
        results = {"steps_failed": []}

    else:
        print("📊 Running Fantasanremo Data Pipeline...")
        print("=" * 60)

        results = run_pipeline(
            steps=steps, dry_run=args.dry_run, verbose=args.verbose, config=config
        )

        # Print results
        print("\n" + "=" * 60)
        print("Pipeline Results")
        print("=" * 60)
        print(f"Steps completed: {', '.join(results.get('steps_completed', []))}")

        if results.get("steps_failed"):
            print(f"Steps failed: {', '.join(results['steps_failed'])}")

        if results.get("validation"):
            val = results["validation"]
            if "quality" in val:
                quality = val["quality"]
                print(f"\nData Quality Score: {quality.get('overall_score', 0)}/100")

        if results.get("quality_report"):
            qr = results["quality_report"]
            print("\nQuality Report:")
            print(f"  Data sources: {qr.get('data_sources_count', 0)}")
            print(f"  Quality score: {qr.get('quality_score', 0)}")

    # Exit with appropriate code
    if results.get("steps_failed"):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
