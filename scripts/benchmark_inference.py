#!/usr/bin/env python3
"""Shim de compatibilidade.

A implementação canônica do benchmark formal vive em src/benchmark_inference.py.
Este arquivo existe apenas para preservar comandos legados em scripts/.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "src" / "benchmark_inference.py"
    runpy.run_path(str(target), run_name="__main__")
