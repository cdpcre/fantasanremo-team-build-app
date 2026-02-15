#!/usr/bin/env python
"""Run ML training benchmark and compare against an optional baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml.benchmark import evaluate_go_no_go, summarize_metrics
from backend.ml.train import save_models, train_models


def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and benchmark ML training")
    parser.add_argument(
        "--label",
        default="candidate",
        help="Label for this run in benchmark output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/ml/models/benchmark_latest.json"),
        help="Where to write benchmark output JSON",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline JSON (summary or benchmark report)",
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        default=None,
        help="Optional path where the current run summary is saved as baseline",
    )
    parser.add_argument(
        "--skip-save-models",
        action="store_true",
        help="Skip persisting model artifacts after training",
    )
    args = parser.parse_args()

    print("Running ML benchmark...")
    models, metrics = train_models()
    if not args.skip_save_models:
        save_models(models, metrics)

    summary = summarize_metrics(metrics, label=args.label)

    result = {"summary": summary}
    baseline_summary = None
    if args.baseline and args.baseline.exists():
        baseline_payload = _load_json(args.baseline)
        if baseline_payload:
            # Accept both raw summary and full benchmark report format.
            baseline_summary = baseline_payload.get("summary", baseline_payload)

    if baseline_summary:
        comparison = evaluate_go_no_go(baseline_summary, summary)
        result["baseline"] = baseline_summary
        result["comparison"] = comparison

    _dump_json(args.output, result)
    if args.write_baseline:
        _dump_json(args.write_baseline, summary)

    reg = summary["regression"]
    cls = summary["classification"]
    print(f"Best regressor: {reg['best_model']} (MAE CV={reg['best_mae_cv']:.3f})")
    print(
        "Category classifier: "
        f"{cls['model']} (macro-F1={cls['macro_f1']:.3f}, bal-acc={cls['balanced_accuracy']:.3f})"
    )
    print(f"Benchmark saved to: {args.output}")

    if "comparison" in result:
        approved = result["comparison"]["approved"]
        print(f"Go/No-Go: {'GO' if approved else 'NO-GO'}")
        return 0 if approved else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
