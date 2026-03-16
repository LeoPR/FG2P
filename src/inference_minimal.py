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
    # ─────────────────────────────────────────────────────────────────────────
    # PASSO 1 — Carregar o modelo (operação cara: ~1–2s, feita UMA VEZ)
    #
    # load() faz: lê dicionário (95k palavras), constrói vocabulários,
    # carrega pesos (~9–17M params) e move para o device (CPU/GPU).
    #
    # ❌ Errado — chamar load() antes de cada palavra:
    #       for word in words:
    #           predictor = G2PPredictor.load(...)  # ~1s de overhead por palavra!
    #           predictor.predict(word)
    #
    # ✅ Correto — carregar UMA VEZ, reutilizar o predictor:
    # ─────────────────────────────────────────────────────────────────────────
    predictor = G2PPredictor.load("best_per")   # best_per = Exp104d (TTS, fonética)
    # predictor = G2PPredictor.load("best_wer") # best_wer = Exp9    (NLP, busca)

    words = ["computador", "português", "inteligência", "extraordinariamente"]

    # ─────────────────────────────────────────────────────────────────────────
    # PASSO 2a — Palavra por vez (TTS em tempo real, consulta interativa)
    #
    # predict() processa uma palavra: encoder → decoder autoregressivo.
    # Latência p50: ~42ms CPU / ~28ms GPU — adequado para resposta em tempo real.
    # Throughput: ~24 w/s CPU / ~34 w/s GPU (limitado pelo overhead por chamada).
    # ─────────────────────────────────────────────────────────────────────────
    print("── Palavra por vez ──")
    for word in words:
        print(f"  {word:24} → {predictor.predict(word)}")

    # ─────────────────────────────────────────────────────────────────────────
    # PASSO 2b — Batch (corpus, pipeline, pré-processamento de texto)
    #
    # predict_batch_native() agrupa N palavras em uma única chamada ao modelo:
    # o encoder processa o batch de uma vez e o decoder roda em paralelo.
    # O resultado é idêntico ao predict() chamado individualmente.
    #
    # Throughput (Exp104d, sweep formal):
    #   CPU batch=32  → ~155 w/s   (+6.5× vs palavra por vez)
    #   CPU batch=128 → ~190 w/s   (pico CPU)
    #   GPU batch=32  → ~406 w/s   (+11.8×)
    #   GPU batch=512 → ~1.106 w/s (pico GPU)
    #
    # Quando usar: qualquer lista de palavras conhecida de antemão —
    # pré-processamento de corpus, normalização de texto, geração de legendas.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Batch (mesma saída, maior throughput) ──")
    results = predictor.predict_batch_native(words, batch_size=32)
    for word, phonemes in zip(words, results):
        print(f"  {word:24} → {phonemes}")
