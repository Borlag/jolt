#!/usr/bin/env python
"""
JOLT Verification Suite Entry Point.

Run the joint solver verification against Boeing JOLT reference results.

Usage:
    python run_verification.py                     # Run all tests
    python run_verification.py --model D06         # Run specific model
    python run_verification.py --formula boeing    # Run with Boeing formula only
    python run_verification.py --list              # List available tests
    python run_verification.py --create-sample D07 # Create sample reference file

For more options:
    python run_verification.py --help
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verification.cli import main

if __name__ == "__main__":
    sys.exit(main())
