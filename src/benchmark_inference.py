"""
Benchmark de Velocidade de Inferência — Todos os Modelos G2P

Mede velocidade usando método robusto chunk-based com filtro IQR de Tukey,
tornando o resultado imune a picos de carga do sistema operacional.

Método:
  1. Warmup de N palavras (descartado)
  2. Divide as palavras de teste em chunks de tamanho C
  3. Mede throughput (palavras/s) de cada chunk individualmente
  4. Descarta chunks lentos: throughput < Q1 - iqr_factor × IQR  (Tukey's fences)
  5. Reporta mediana e média aparada 10% dos chunks limpos

Métricas por modelo:
  throughput_median_wps       — mediana robusta (recomendada para comparação)
  throughput_trimmed_mean_wps — média aparada 10% (estimativa central)
  latency_median_ms           — latência mediana por palavra
  chunks_total / chunks_discarded / jitter_pct — qualidade do benchmark

Saída: results/_reports/inference_benchmark.json

Uso:
  python src/benchmark_inference.py                          # GPU se disponível
  python src/benchmark_inference.py --device cpu             # força CPU
  python src/benchmark_inference.py --device cuda            # requer CUDA
  python src/benchmark_inference.py --dry-run 3              # últimos 3 modelos
  python src/benchmark_inference.py --words 2000 --chunk-size 50
  python src/benchmark_inference.py --iqr-factor 2.0         # filtro mais permissivo
"""

import json
import logging
import statistics
import time
import argparse
from pathlib import Path
from typing import List, Dict

import sys
import torch

sys.path.insert(0, str(Path(__file__).parent))

from inference_light import G2PPredictor
from utils import get_all_models_sorted

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s"
)
logger = logging.getLogger("benchmark_inference")

DICT_PATH       = Path("dicts/pt-br.tsv")
DEFAULT_OUTPUT  = Path("results/_reports/inference_benchmark.json")
DEFAULT_WORDS   = 5000
DEFAULT_WARMUP  = 100
DEFAULT_CHUNK   = 100
DEFAULT_IQR     = 1.5
DEFAULT_TRIM    = 0.10   # 10% de cada extremo para trimmed mean


# ── Device ────────────────────────────────────────────────────────────────────

def resolve_device(device_arg: str) -> torch.device:
    """Resolve --device {auto|cpu|cuda} para torch.device."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA solicitado mas não disponível — usando CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Palavras de teste ─────────────────────────────────────────────────────────

def load_test_words(n_words: int = DEFAULT_WORDS) -> List[str]:
    """
    Carrega palavras de teste do dicionário (determinístico).
    Sempre as mesmas N primeiras palavras → comparação cross-model consistente.
    """
    words = []
    if not DICT_PATH.exists():
        logger.warning(f"Dicionário não encontrado: {DICT_PATH}, usando dummy")
        return [f"word{i}" for i in range(min(n_words, 100))]

    with open(DICT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if len(words) >= n_words:
                break
            parts = line.strip().split("\t")
            if parts and parts[0].strip():
                words.append(parts[0].strip())

    logger.info(f"Carregadas {len(words)} palavras de teste")
    return words


# ── Estatística robusta ───────────────────────────────────────────────────────

def _iqr_filter(values: List[float], iqr_factor: float) -> List[float]:
    """
    Remove outliers inferiores pelo critério IQR de Tukey.
    Chunks lentos (sistema sobrecarregado) ficam abaixo de Q1 - factor*IQR.
    Não remove outliers superiores: throughput alto é bom, não deve ser descartado.
    """
    if len(values) < 4:
        return values
    qs = statistics.quantiles(values, n=4)
    q1, q3 = qs[0], qs[2]
    iqr = q3 - q1
    fence = q1 - iqr_factor * iqr
    return [v for v in values if v >= fence]


def _trimmed_mean(values: List[float], trim_frac: float = DEFAULT_TRIM) -> float:
    """Média aparada: remove trim_frac de cada extremo da lista ordenada."""
    if not values:
        return 0.0
    n = len(values)
    cut = int(n * trim_frac)
    trimmed = sorted(values)[cut: n - cut] if cut > 0 else list(values)
    return statistics.mean(trimmed) if trimmed else 0.0


# ── Benchmark de um modelo ────────────────────────────────────────────────────

def benchmark_model(
    model_path: Path,
    test_words: List[str],
    warmup_words: int = DEFAULT_WARMUP,
    chunk_size: int = DEFAULT_CHUNK,
    iqr_factor: float = DEFAULT_IQR,
    device: torch.device = None,
) -> Dict:
    """
    Benchmark robusto de um modelo com medição chunk-based e filtro IQR.

    Retorna dict com métricas de velocidade ou {"status": "error", "error": ...}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        predictor = G2PPredictor.load(model_path=model_path, device=device)
    except Exception as e:
        logger.error(f"  ERRO ao carregar modelo: {e}")
        return {"status": "error", "error": str(e)}

    # --- Warmup (GPU/CPU cold-start) ---
    for word in test_words[:warmup_words]:
        _ = predictor.predict(word)

    # --- Dividir em chunks ---
    chunks = [
        test_words[i: i + chunk_size]
        for i in range(0, len(test_words), chunk_size)
    ]
    # Descarta último chunk incompleto para evitar viés de fim de lista
    if len(chunks) > 1 and len(chunks[-1]) < chunk_size:
        chunks = chunks[:-1]

    # --- Medir throughput por chunk ---
    chunk_throughputs: List[float] = []
    for chunk in chunks:
        t0 = time.perf_counter()
        for word in chunk:
            _ = predictor.predict(word)
        elapsed = time.perf_counter() - t0
        chunk_throughputs.append(len(chunk) / elapsed if elapsed > 0 else 0.0)

    chunks_total = len(chunk_throughputs)

    # --- Filtro IQR ---
    clean = _iqr_filter(chunk_throughputs, iqr_factor)
    chunks_discarded = chunks_total - len(clean)
    jitter_pct = round(100 * chunks_discarded / chunks_total, 1) if chunks_total > 0 else 0.0

    if len(clean) < 2:
        logger.warning("  Poucos chunks limpos após IQR — usando todos para evitar perda de dados")
        clean = chunk_throughputs

    # --- Métricas finais ---
    throughput_median  = statistics.median(clean)
    throughput_trimmed = _trimmed_mean(clean)
    latency_median_ms  = (1000.0 / throughput_median) if throughput_median > 0 else 0.0

    return {
        "status":                      "success",
        "device":                      str(device),
        "throughput_median_wps":       round(throughput_median,  2),
        "throughput_trimmed_mean_wps": round(throughput_trimmed, 2),
        "latency_median_ms":           round(latency_median_ms,  3),
        "chunks_total":                chunks_total,
        "chunks_discarded":            chunks_discarded,
        "jitter_pct":                  jitter_pct,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de velocidade de inferência para modelos G2P",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python src/benchmark_inference.py
  python src/benchmark_inference.py --device cpu
  python src/benchmark_inference.py --dry-run 3
  python src/benchmark_inference.py --words 2000 --chunk-size 50 --iqr-factor 2.0
"""
    )
    parser.add_argument("--dry-run",    type=int,   default=0,
                        help="Testar apenas os últimos N modelos (0 = todos)")
    parser.add_argument("--output",     type=Path,  default=DEFAULT_OUTPUT,
                        help=f"Arquivo de saída JSON (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--words",      type=int,   default=DEFAULT_WORDS,
                        help=f"Palavras de teste por modelo (default: {DEFAULT_WORDS})")
    parser.add_argument("--warmup",     type=int,   default=DEFAULT_WARMUP,
                        help=f"Palavras de warmup descartadas (default: {DEFAULT_WARMUP})")
    parser.add_argument("--chunk-size", type=int,   default=DEFAULT_CHUNK,
                        help=f"Palavras por chunk de medição (default: {DEFAULT_CHUNK})")
    parser.add_argument("--iqr-factor", type=float, default=DEFAULT_IQR,
                        help=f"Fator IQR para descarte de chunks lentos (default: {DEFAULT_IQR})")
    parser.add_argument("--device",     choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device: auto (default) | cpu (força CPU) | cuda (requer GPU)")

    args = parser.parse_args()
    device = resolve_device(args.device)

    logger.info("=" * 70)
    logger.info("BENCHMARK DE VELOCIDADE DE INFERÊNCIA — G2P LSTM")
    logger.info("=" * 70)
    logger.info(f"Device     : {device}")
    logger.info(f"Metodo     : chunk-based (size={args.chunk_size}) + IQR filter (factor={args.iqr_factor})")
    logger.info(f"Palavras   : {args.words} | Warmup: {args.warmup}")

    test_words  = load_test_words(args.words)
    model_paths = list(get_all_models_sorted())

    if args.dry_run > 0:
        model_paths = model_paths[-args.dry_run:]
        logger.info(f"Modo DRY-RUN: testando {len(model_paths)} modelo(s)")
    else:
        logger.info(f"Testando {len(model_paths)} modelo(s)")

    if not model_paths:
        logger.error("Nenhum modelo encontrado em models/")
        return

    results = {
        "metadata": {
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device":         str(device),
            "test_words":     len(test_words),
            "warmup_words":   args.warmup,
            "chunk_size":     args.chunk_size,
            "iqr_factor":     args.iqr_factor,
            "models_tested":  len(model_paths),
            "method":         "chunk-iqr",
        },
        "results": []
    }

    for i, model_path in enumerate(model_paths, 1):
        model_name = model_path.stem
        logger.info(f"\n[{i}/{len(model_paths)}] {model_name}...")

        # Lê metadata para enriquecer o resultado (path correto: _metadata.json)
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        experiment = model_name
        params = 0
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    md = json.load(f)
                experiment = md.get("experiment_name", model_name)
                params     = md.get("total_params", 0)
            except Exception:
                pass

        metrics = benchmark_model(
            model_path, test_words,
            warmup_words=args.warmup,
            chunk_size=args.chunk_size,
            iqr_factor=args.iqr_factor,
            device=device,
        )

        entry = {"model_file": model_path.name, "experiment": experiment, "params": params}
        entry.update(metrics)
        results["results"].append(entry)

        if metrics["status"] == "success":
            logger.info(
                f"  > {metrics['throughput_median_wps']} w/s (median)"
                f" | {metrics['throughput_trimmed_mean_wps']} w/s (trimmed)"
                f" | {metrics['latency_median_ms']} ms/word"
                f" | jitter {metrics['jitter_pct']}%"
                f" ({metrics['chunks_discarded']}/{metrics['chunks_total']} chunks descartados)"
            )
        else:
            logger.warning(f"  FALHA: {metrics.get('error', 'desconhecido')}")

    # Salvar JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Resumo final
    successful = [r for r in results["results"] if r["status"] == "success"]
    failed     = [r for r in results["results"] if r["status"] == "error"]

    logger.info("\n" + "=" * 70)
    logger.info("RESUMO")
    logger.info("=" * 70)
    logger.info(f"Modelos testados com sucesso: {len(successful)}/{len(model_paths)}")
    if successful:
        medians = [r["throughput_median_wps"] for r in successful]
        jitters = [r["jitter_pct"] for r in successful]
        logger.info(f"  Throughput (median): {min(medians):.1f} — {max(medians):.1f} w/s")
        logger.info(f"  Media geral        : {sum(medians)/len(medians):.1f} w/s")
        logger.info(f"  Jitter medio       : {sum(jitters)/len(jitters):.1f}%")
    if failed:
        logger.warning(f"Modelos falhados: {len(failed)}")
        for r in failed:
            logger.warning(f"  - {r['experiment']}: {r.get('error', '?')}")

    logger.info(f"\nBenchmark concluido: {args.output}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
