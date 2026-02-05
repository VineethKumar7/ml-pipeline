"""
Pytest configuration and fixtures.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set working directory to project root for config loading
os.chdir(project_root)
