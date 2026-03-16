#!/usr/bin/env python
"""
inference_minimal.py — Interface mínima para usar FG2P.

Dependência: torch (nada mais)

Uso Python — palavra única (TTS, consulta):
    from src.inference_minimal import G2PPredictor

    p = G2PPredictor.load("best_per")   # melhor PER — TTS, fonética
    p = G2PPredictor.load("best_wer")   # melhor WER — NLP, busca
    p.predict("computador")             # → "k õ p u . t a . ˈ d o x"

Uso Python — batch (corpus, pipeline — 6–32× mais rápido):
    results = p.predict_batch_native(lista_de_palavras, batch_size=32)
    # CPU: batch=32 → ~155 w/s | batch=128 → ~190 w/s (pico)
    # GPU: batch=32 → ~406 w/s | batch=512 → ~1.106 w/s (pico)

CLI (completo com batch, neologismos e mais):
    python src/inference_light.py --alias best_per --word computador
    python src/inference_light.py --alias best_per --file corpus.txt --batch-size 128
    python src/inference_light.py --alias best_per --neologisms docs/data/generalization_test.tsv
    python src/inference_light.py --help
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

    # Palavra única — TTS, consulta interativa (p50: ~42ms CPU / ~28ms GPU)
    test_words = ["computador", "português", "inteligência"]
    for word in test_words:
        phonemes = predictor.predict(word)
        print(f"{word:20} → {phonemes}")

    # Batch — corpus, pipeline (CPU: ~155 w/s em batch=32; GPU: ~406 w/s)
    # results = predictor.predict_batch_native(test_words, batch_size=32)
