"""
Test suite for FantaSanremo Team Builder backend.

This package contains comprehensive unit tests for:
- API endpoints (test_api.py)
- Database models (test_models.py)
- ML features and predictions (test_ml.py)
"""

import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
