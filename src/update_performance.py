#!/usr/bin/env python
# SHIM — lógica movida para src/manager/_sync.py.
# Uso: python src/update_performance.py [--dry-run] [--filter exp9] [--update-meta "msg"]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from manager._sync import _cli_sync_performance
if __name__ == "__main__":
    _cli_sync_performance()
    sys.exit(0)
