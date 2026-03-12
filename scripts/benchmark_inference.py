#!/usr/bin/env python3
"""
benchmark_inference.py — Benchmark de performance (GPU vs CPU) para modelos G2P.

Mede throughput (words/sec) e latência para os melhores modelos em ambos devices.

Uso:
    python scripts/benchmark_inference.py --models best_per,best_wer,fast --warmup 10 --runs 100
    python scripts/benchmark_inference.py --help

Saída:
    Tabela comparativa GPU vs CPU com latência e throughput.
"""

import argparse
import time
import sys
import torch
from pathlib import Path
from typing import List, Tuple

# Setup paths
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from inference_light import G2PPredictor


def benchmark_model(
    predictor_alias: str,
    device: torch.device,
    test_words: List[str],
    warmup_runs: int = 10,
    benchmark_runs: int = 100
) -> dict:
    """
    Benchmark um modelo em um device específico.

    Args:
        predictor_alias: "best_per", "best_wer", "fast", etc.
        device: torch.device("cuda") ou torch.device("cpu")
        test_words: lista de palavras para teste
        warmup_runs: rodadas de aquecimento (descartadas)
        benchmark_runs: rodadas de benchmark (contadas)

    Returns:
        dict com latência média, desvio padrão, e throughput
    """
    print(f"\n  Loading '{predictor_alias}' on {device}...", end=" ", flush=True)
    try:
        predictor = G2PPredictor.load(alias=predictor_alias, device=device)
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")
        return None

    # Aquecimento (warm-up)
    print(f"  Warmup ({warmup_runs} runs)...", end=" ", flush=True)
    for _ in range(warmup_runs):
        for word in test_words:
            _ = predictor.predict(word)
    print("✓")

    # Benchmark
    print(f"  Benchmarking ({benchmark_runs} runs)...", end=" ", flush=True)
    latencies = []
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()

    for _ in range(benchmark_runs):
        for word in test_words:
            t0 = time.perf_counter()
            _ = predictor.predict(word)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.perf_counter()

    total_time = end - start
    total_words = benchmark_runs * len(test_words)

    # Calcular estatísticas
    latencies.sort()
    avg_latency = sum(latencies) / len(latencies)
    min_latency = latencies[0]
    max_latency = latencies[-1]
    p50_latency = latencies[len(latencies) // 2]
    p95_latency = latencies[int(len(latencies) * 0.95)]

    throughput = total_words / total_time  # words/sec

    print(f"✓ ({total_time:.2f}s)")

    return {
        "alias": predictor_alias,
        "device": str(device),
        "total_words": total_words,
        "total_time": total_time,
        "avg_latency_ms": avg_latency * 1000,
        "min_latency_ms": min_latency * 1000,
        "max_latency_ms": max_latency * 1000,
        "p50_latency_ms": p50_latency * 1000,
        "p95_latency_ms": p95_latency * 1000,
        "throughput_wps": throughput,  # words per second
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de inference GPU vs CPU para modelos G2P"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="best_per,best_wer,fast",
        help="Modelos a testar (separados por vírgula). Padrão: best_per,best_wer,fast"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Número de rodadas de aquecimento. Padrão: 10"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Número de rodadas de benchmark. Padrão: 100"
    )
    parser.add_argument(
        "--words",
        type=str,
        default="computador,português,inteligência,aplicação,desenvolvimento",
        help="Palavras para teste (separadas por vírgula)"
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    test_words = [w.strip() for w in args.words.split(",")]

    print("=" * 90)
    print(f"G2P Inference Benchmark: GPU vs CPU")
    print("=" * 90)
    print(f"Models: {', '.join(models)}")
    print(f"Test words ({len(test_words)}): {', '.join(test_words)}")
    print(f"Warmup runs: {args.warmup} | Benchmark runs: {args.runs}")
    print("=" * 90)

    # Determinar quais devices estão disponíveis
    devices = []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("\n✗ GPU not available (CUDA)")
    devices.append(torch.device("cpu"))

    # Executar benchmarks
    results = {}
    for model_alias in models:
        print(f"\n{'─' * 90}")
        print(f"Model: {model_alias}")
        print(f"{'─' * 90}")

        results[model_alias] = {}
        for device in devices:
            result = benchmark_model(
                model_alias,
                device,
                test_words,
                warmup_runs=args.warmup,
                benchmark_runs=args.runs
            )
            if result:
                results[model_alias][str(device)] = result

    # Imprimir tabela comparativa
    print(f"\n{'=' * 90}")
    print("RESULTADOS")
    print(f"{'=' * 90}\n")

    for model_alias in models:
        if model_alias not in results or not results[model_alias]:
            continue

        print(f"Model: {model_alias}")
        print(f"{'-' * 90}")
        print(
            f"{'Device':<12} {'Throughput':<15} {'Avg Latency':<15} "
            f"{'P50':<12} {'P95':<12} {'Total Time':<12}"
        )
        print(f"{'-' * 90}")

        gpu_result = results[model_alias].get("cuda")
        cpu_result = results[model_alias].get("cpu")

        if gpu_result:
            print(
                f"{'GPU':<12} "
                f"{gpu_result['throughput_wps']:>6.1f} w/s{'':<7} "
                f"{gpu_result['avg_latency_ms']:>6.2f} ms{'':<7} "
                f"{gpu_result['p50_latency_ms']:>6.2f} ms{'':<5} "
                f"{gpu_result['p95_latency_ms']:>6.2f} ms{'':<5} "
                f"{gpu_result['total_time']:>6.2f}s"
            )

        if cpu_result:
            print(
                f"{'CPU':<12} "
                f"{cpu_result['throughput_wps']:>6.1f} w/s{'':<7} "
                f"{cpu_result['avg_latency_ms']:>6.2f} ms{'':<7} "
                f"{cpu_result['p50_latency_ms']:>6.2f} ms{'':<5} "
                f"{cpu_result['p95_latency_ms']:>6.2f} ms{'':<5} "
                f"{cpu_result['total_time']:>6.2f}s"
            )

        if gpu_result and cpu_result:
            speedup = gpu_result['throughput_wps'] / cpu_result['throughput_wps']
            print(f"{'Speedup:':<12} {speedup:.1f}x faster on GPU")

        print()

    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
