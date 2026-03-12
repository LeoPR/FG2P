#!/usr/bin/env python
"""
scripts/cross_eval.py — Avaliacao cross-split

Avalia qualquer modelo FG2P em um test set gerado por parametros de split
DIFERENTES dos usados no treinamento do modelo.

Permite comparacao direta entre modelos treinados com splits distintos
(estratificado vs nao-estratificado, seeds diferentes).

Uso:
    # Avaliar modelo 11 (exp0_legacy) no split estratificado padrao:
    python scripts/cross_eval.py --index 11 --stratify --seed 42 --test-ratio 0.2

    # Avaliar modelo 8 (exp0_baseline) no split NAO estratificado do legacy:
    python scripts/cross_eval.py --index 8 --no-stratify --seed 42 --test-ratio 0.2

    # Listar modelos disponiveis:
    python scripts/cross_eval.py --list

Saida:
    PER, WER, Accuracy, distribuicao de erros A/B/C/D (se PanPhon disponivel)

Requisitos: torch (mesmo que inference_light.py)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.inference_light import G2PPredictor  # noqa: E402
from src.g2p import G2PCorpus  # noqa: E402


# ---------------------------------------------------------------------------
# PER / WER helpers (sem dependencias pesadas)
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _per(pred_seq: list[str], ref_seq: list[str]) -> tuple[int, int]:
    """Returns (edit_distance, ref_len) for one word."""
    return _edit_distance(pred_seq, ref_seq), len(ref_seq)


def evaluate(
    predictor: G2PPredictor,
    test_pairs: list[tuple[str, str]],
    verbose: bool = False,
) -> dict:
    """
    Evaluate predictor on test_pairs = [(grapheme, ipa_reference), ...].
    Returns dict with per, wer, accuracy and counts.
    """
    total_phone_errors = 0
    total_phones = 0
    word_errors = 0
    total_words = len(test_pairs)
    skipped = 0

    for grapheme, reference in test_pairs:
        try:
            prediction = predictor.predict(grapheme)
        except Exception:
            skipped += 1
            continue

        pred_tokens = prediction.split()
        ref_tokens = reference.split()

        ed, ref_len = _per(pred_tokens, ref_tokens)
        total_phone_errors += ed
        total_phones += max(ref_len, 1)

        if prediction.strip() != reference.strip():
            word_errors += 1

    per = (total_phone_errors / total_phones * 100) if total_phones else float("nan")
    wer = (word_errors / total_words * 100) if total_words else float("nan")
    accuracy = 100 - wer

    return {
        "per": per,
        "wer": wer,
        "accuracy": accuracy,
        "total_words": total_words,
        "word_errors": word_errors,
        "total_phones": total_phones,
        "phone_errors": total_phone_errors,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_models() -> None:
    from inference import _get_complete_model_files, load_model_metadata
    model_files = _get_complete_model_files()
    print(f"\n{'Idx':>4}  {'Experimento':<50}  {'Epochs':>8}")
    print("-" * 70)
    for i, pt in enumerate(model_files):
        meta = load_model_metadata(pt) or {}
        exp_name = meta.get("experiment_name", pt.stem)
        final_epoch = meta.get("final_epoch", meta.get("current_epoch", "?"))
        total_epochs = meta.get("total_epochs", "?")
        print(f"  {i:>3}  {exp_name:<50}  {str(final_epoch):>3}/{str(total_epochs):<4}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-split evaluation: avaliar modelo em split diferente do seu treino"
    )
    parser.add_argument("--index", "-i", type=int, default=None, help="Indice do modelo")
    parser.add_argument(
        "--alias",
        "-a",
        default=None,
        help="Alias do modelo (best_per, best_wer, etc.)",
    )
    parser.add_argument("--list", action="store_true", help="Listar modelos disponiveis")
    parser.add_argument(
        "--stratify",
        dest="stratify",
        action="store_true",
        default=True,
        help="Usar split estratificado (padrao)",
    )
    parser.add_argument(
        "--no-stratify",
        dest="stratify",
        action="store_false",
        help="Usar split NAO estratificado",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed do split (default: 42)")
    parser.add_argument(
        "--test-ratio", type=float, default=0.2, help="Fracao do test set (default: 0.2)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Fracao do val set (default: 0.1)"
    )
    parser.add_argument("--dict", default=None, help="Caminho para dicionario TSV (opcional)")
    parser.add_argument("--verbose", action="store_true", help="Mostrar erros por palavra")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.index is None and args.alias is None:
        parser.print_help()
        print("\nERRO: especifique --index N ou --alias <nome>")
        sys.exit(1)

    # --- Load model ---
    print("Carregando modelo...")
    if args.alias:
        predictor = G2PPredictor.load(args.alias)
    else:
        predictor = G2PPredictor.load(args.index)
    print(f"Modelo: {predictor.model_name}")

    # --- Load corpus ---
    dict_path = args.dict or str(ROOT / "dicts" / "pt-br.tsv")
    print(f"Corpus: {dict_path}")
    corpus = G2PCorpus(dict_path)

    # --- Build cross split ---
    strat_label = "estratificado" if args.stratify else "NAO-estratificado"
    print(
        f"Split: test={args.test_ratio:.0%}, seed={args.seed}, {strat_label}"
    )
    split = corpus.split(
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify=args.stratify,
    )

    test_pairs = split["test"]
    print(f"Test set: {len(test_pairs):,} palavras\n")

    # --- Evaluate ---
    print("Avaliando...")
    results = evaluate(predictor, test_pairs, verbose=args.verbose)

    # --- Report ---
    print(f"\n{'=' * 50}")
    print(f"Modelo:    {predictor.model_name}")
    print(f"Split:     seed={args.seed}, stratify={args.stratify}, test={args.test_ratio:.0%}")
    print(f"N_test:    {results['total_words']:,}")
    print(f"{'=' * 50}")
    print(f"PER:       {results['per']:.4f}%  ({results['phone_errors']:,}/{results['total_phones']:,} fonemas)")
    print(f"WER:       {results['wer']:.4f}%  ({results['word_errors']:,}/{results['total_words']:,} palavras)")
    print(f"Accuracy:  {results['accuracy']:.4f}%")
    if results["skipped"]:
        print(f"Skipped:   {results['skipped']} (erro de predicao)")
    print(f"{'=' * 50}")

    # Contexto de comparacao
    print("\nContexto:")
    print(f"  Performance.json (treino original):  ver docs/report/performance.json")
    print(f"  Se PER aqui ~= PER original: split nao introduziu bias no modelo")
    print(f"  Se PER aqui >> PER original: modelo overfittado ao seu split especifico")


if __name__ == "__main__":
    main()
