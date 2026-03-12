#!/usr/bin/env python
"""
inference_minimal.py — Interface mínima para usar FG2P.

Dependência: torch (nada mais)

Uso Python:
    from src.inference_minimal import G2PPredictor

    predictor = G2PPredictor.load("best_per")
    print(predictor.predict("computador"))

Ou CLI:
    python -m src.inference_minimal
    python src/inference_minimal.py
"""

import sys
from pathlib import Path

# Se rodado como script direto, adiciona parent ao path para imports relativos funcionarem
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from .inference_light import G2PPredictor

__all__ = ["G2PPredictor"]


if __name__ == "__main__":
    predictor = G2PPredictor.load("best_per")

    test_words = ["computador", "português", "inteligência"]
    for word in test_words:
        phonemes = predictor.predict(word)
        print(f"{word:20} → {phonemes}")
