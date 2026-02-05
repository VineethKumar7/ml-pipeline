"""
Pytest Configuration
====================

This module configures pytest for the ML Pipeline test suite.

Configuration includes:
- Adding project root to Python path for imports
- Setting working directory for config file loading
- Shared fixtures (defined in individual test files)

Usage:
    This file is automatically loaded by pytest.
    Run tests with: pytest tests/ -v

See Also:
    - tests/test_model.py: Model unit tests
    - tests/test_api.py: API integration tests
"""

import os
import sys
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Get the project root directory (parent of tests/)
project_root = Path(__file__).parent.parent

# Add project root to Python path for imports
# This allows: from src.config import get_config
sys.path.insert(0, str(project_root))

# Change working directory to project root
# This is required for config.yaml loading to work correctly
os.chdir(project_root)
