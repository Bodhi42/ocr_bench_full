#!/usr/bin/env python3
"""Generate dashboard charts from computed metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard import generate_all

if __name__ == "__main__":
    generate_all()
