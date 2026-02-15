#!/usr/bin/env python3
"""Script to execute Jupyter notebooks programmatically using nbconvert."""

import subprocess
import sys
from pathlib import Path

def run_notebook(notebook_path: Path, timeout: int = 600) -> bool:
    """Execute a Jupyter notebook using nbconvert and return True if successful."""
    print(f"\n{'='*80}")
    print(f"Running notebook: {notebook_path.name}")
    print(f"{'='*80}\n")

    output_path = notebook_path.with_suffix('.executed.ipynb')

    try:
        # Run notebook with nbconvert
        result = subprocess.run(
            [
                'uv', 'run', 'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--ExecutePreprocessor.timeout=600',
                '--output', str(output_path.name),
                str(notebook_path)
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            print(f"❌ Error executing {notebook_path.name}:")
            print(result.stderr)
            return False

        print(f"✅ {notebook_path.name} completed successfully!")
        print(f"   Output saved to: {output_path}")

        # Clean up the executed notebook
        if output_path.exists():
            output_path.unlink()

        return True

    except subprocess.TimeoutExpired:
        print(f"❌ Timeout executing {notebook_path.name}")
        return False
    except Exception as e:
        print(f"❌ Exception executing {notebook_path.name}: {e}")
        return False

def main():
    """Run all notebooks in sequence."""
    notebooks_dir = Path('backend/ml/notebooks')

    notebooks = [
        'exploratory_data_analysis.ipynb',
        'feature_engineering.ipynb',
        'model_training.ipynb'
    ]

    for notebook in notebooks:
        notebook_path = notebooks_dir / notebook
        if not notebook_path.exists():
            print(f"❌ Notebook not found: {notebook_path}")
            return False

        if not run_notebook(notebook_path):
            print(f"\n❌ Failed to run {notebook}")
            return False

    print(f"\n{'='*80}")
    print("🎉 All notebooks executed successfully!")
    print(f"{'='*80}\n")

    # Show generated files
    processed_dir = Path('backend/ml/data/processed')
    models_dir = Path('backend/ml/models')

    print("\n📂 Generated files:")
    if processed_dir.exists():
        print(f"\n📊 Processed data ({processed_dir}):")
        for f in sorted(processed_dir.glob('*')):
            print(f"  • {f.name}")

    if models_dir.exists():
        print(f"\n🤖 Models ({models_dir}):")
        for f in sorted(models_dir.glob('*')):
            print(f"  • {f.name}")

    return True

if __name__ == '__main__':
    sys.exit(0 if main() else 1)
