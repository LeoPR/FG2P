#!/usr/bin/env python3
"""
benchmark_inference.py — Benchmark de performance (GPU vs CPU) para modelos G2P.

Mede throughput (words/sec, tokens/sec) e latência com detecção de instabilidade
de hardware (contention, throttling térmico) via análise post-hoc.

Princípios de medição:
  - Loop quente coleta apenas timestamps + contagem de tokens (overhead mínimo e determinístico)
  - Calibração inicial mede overhead do próprio loop (perf_counter + append) e subtrai das latências
  - Toda análise estatística (rolling window, CV, thermal check) é feita APÓS coleta, sem contaminar medições
  - Se o overhead do medidor for estável/determinístico, subtrair é suficiente — não precisa removê-lo

Uso:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --models best_per,best_wer --warmup 20 --runs 200
    python scripts/benchmark_inference.py --help

Saída:
    Tabela GPU vs CPU com throughput global, throughput estável (janela de menor CV),
    check térmico (primeiros vs últimos 20%), p50/p95/p99 e aviso de contention.

TODO (Tier 2 — não implementado):
  - Benchmark por grupo: palavras corretas vs incorretas (pré-classificadas offline via
    predictions.tsv existente). Hipótese: diferença de velocidade é efeito de comprimento
    de saída (alucinações=lento, truncamentos=rápido), não de confiança. Medir tokens/sec
    por grupo para isolar o efeito.
  - Monitoramento de temperatura GPU (pynvml): correlacionar drops de throughput com
    temperatura para distinguir throttling térmico de contention de software.
  - Sweep de tamanho de lote: throughput vs latência para encontrar ponto ótimo de batch.
"""

import argparse
import time
import sys
import math
import torch
from pathlib import Path
from typing import List, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from inference_light import G2PPredictor
from inference_light import _get_complete_model_files


# ---------------------------------------------------------------------------
# Calibração de overhead do loop
# ---------------------------------------------------------------------------

def _calibrate_loop_overhead(n: int = 2000) -> float:
    """
    Mede o overhead de um par perf_counter() + list.append() sem corpo útil.
    Retorna overhead médio em segundos por iteração.

    Esse valor é subtraído de cada latência medida no loop quente,
    eliminando o custo do instrumento sem precisar removê-lo.
    Válido desde que o overhead seja estável (determinístico) — o que
    perf_counter() garante em sistemas modernos (~50-300ns por chamada).
    """
    buf = []
    for _ in range(n):
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        buf.append(t1 - t0)
    # Descarta os 10% extremos para robustez
    buf.sort()
    trimmed = buf[n // 10: n - n // 10]
    return sum(trimmed) / len(trimmed)


# ---------------------------------------------------------------------------
# Análise post-hoc (opera sobre latências brutas, fora do loop quente)
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list, p: float) -> float:
    idx = int(len(sorted_data) * p)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def _cv(data: list) -> float:
    """Coeficiente de variação (std/mean). Alta CV → hardware instável."""
    if not data:
        return 0.0
    mean = sum(data) / len(data)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance) / mean


def _find_stable_window(latencies: list, window_frac: float = 0.20):
    """
    Varre janelas deslizantes de tamanho window_frac*N e retorna a janela
    com menor CV (mais estável). Representa o throughput real quando o
    hardware não está com contention ou throttling.

    Retorna: (mean_latency_stable, cv_stable, window_start_index)
    """
    n = len(latencies)
    w = max(10, int(n * window_frac))
    best_cv = float("inf")
    best_start = 0

    for i in range(n - w + 1):
        window = latencies[i : i + w]
        c = _cv(window)
        if c < best_cv:
            best_cv = c
            best_start = i

    stable = latencies[best_start : best_start + w]
    stable_mean = sum(stable) / len(stable)
    return stable_mean, best_cv, best_start


def _thermal_check(latencies: list, frac: float = 0.20):
    """
    Compara latência média dos primeiros vs últimos frac% das medições.
    - last/first > 1.10 → hardware aqueceu durante teste (throttling)
    - last/first < 0.90 → warmup insuficiente (hardware ainda esquentando no início)
    - ratio ≈ 1.0 → temperatura estável, medições confiáveis

    Retorna: (mean_first, mean_last, ratio)
    """
    n = len(latencies)
    w = max(5, int(n * frac))
    mean_first = sum(latencies[:w]) / w
    mean_last = sum(latencies[-w:]) / w
    ratio = mean_last / mean_first if mean_first > 0 else 1.0
    return mean_first, mean_last, ratio


def _analyze(
    latencies: list,
    token_counts: list,
    input_char_counts: list,
    output_char_counts: list,
    loop_overhead_s: float,
):
    """
    Análise completa post-hoc sobre latências brutas.
    Subtrai overhead de calibração antes de qualquer cálculo.
    """
    # Subtrai overhead do medidor (determinístico)
    corrected = [max(0.0, lat - loop_overhead_s) for lat in latencies]
    corrected_sorted = sorted(corrected)

    n = len(corrected)
    mean_lat = sum(corrected) / n
    variance = sum((x - mean_lat) ** 2 for x in corrected) / n if n > 1 else 0.0
    std_lat = math.sqrt(variance)
    sem_lat = std_lat / math.sqrt(n) if n > 1 else 0.0
    ci95_lat_low = max(0.0, mean_lat - 1.96 * sem_lat)
    ci95_lat_high = mean_lat + 1.96 * sem_lat
    total_time = sum(corrected)
    total_tokens = sum(token_counts)
    total_chars_in = sum(input_char_counts)
    total_chars_out = sum(output_char_counts)

    throughput_wps = n / total_time if total_time > 0 else 0.0
    throughput_tps = total_tokens / total_time if total_time > 0 else 0.0
    throughput_cps_in = total_chars_in / total_time if total_time > 0 else 0.0
    throughput_cps_out = total_chars_out / total_time if total_time > 0 else 0.0

    # CI95 aproximado para throughput via inversão do CI da latência média.
    # w/s = 1 / latência(s) por palavra.
    wps_ci95_low = (1.0 / ci95_lat_high) if ci95_lat_high > 0 else 0.0
    wps_ci95_high = (1.0 / ci95_lat_low) if ci95_lat_low > 0 else 0.0

    stable_lat, stable_cv, stable_start = _find_stable_window(corrected)
    stable_wps = 1.0 / stable_lat if stable_lat > 0 else 0.0

    mean_first, mean_last, thermal_ratio = _thermal_check(corrected)

    global_cv = _cv(corrected)

    return {
        "n":                   n,
        "total_time_s":        total_time,
        "mean_lat_ms":         mean_lat * 1000,
        "p50_ms":              _percentile(corrected_sorted, 0.50) * 1000,
        "p95_ms":              _percentile(corrected_sorted, 0.95) * 1000,
        "p99_ms":              _percentile(corrected_sorted, 0.99) * 1000,
        "throughput_wps":      throughput_wps,
        "throughput_tps":      throughput_tps,
        "throughput_cps_in":   throughput_cps_in,
        "throughput_cps_out":  throughput_cps_out,
        "tokens_per_word":     total_tokens / n if n > 0 else 0.0,
        "chars_in_per_word":   total_chars_in / n if n > 0 else 0.0,
        "chars_out_per_word":  total_chars_out / n if n > 0 else 0.0,
        "ci95_lat_low_ms":     ci95_lat_low * 1000,
        "ci95_lat_high_ms":    ci95_lat_high * 1000,
        "ci95_wps_low":        wps_ci95_low,
        "ci95_wps_high":       wps_ci95_high,
        "stable_wps":          stable_wps,
        "stable_cv":           stable_cv,
        "stable_window_start": stable_start,
        "global_cv":           global_cv,
        "thermal_ratio":       thermal_ratio,
        "mean_first_ms":       mean_first * 1000,
        "mean_last_ms":        mean_last * 1000,
        "loop_overhead_us":    loop_overhead_s * 1e6,
    }


# ---------------------------------------------------------------------------
# Loop quente de benchmark
# ---------------------------------------------------------------------------

def benchmark_model(
    model_ref: str,
    device: torch.device,
    test_words: List[str],
    warmup_runs: int = 20,
    benchmark_runs: int = 200,
    loop_overhead_s: float = 0.0,
) -> Optional[dict]:
    """
    Benchmark de um modelo em um device.

    Loop quente: apenas perf_counter() + predict() + append().
    Toda análise estatística é feita post-hoc por _analyze().
    """
    print(f"\n  Carregando '{model_ref}' em {device}...", end=" ", flush=True)
    try:
        model_id = model_ref
        if model_ref.startswith("index:"):
            idx = int(model_ref.split(":", 1)[1])
            predictor = G2PPredictor.load(index=idx, device=device)
            model_id = f"index:{idx}:{predictor.experiment_name}"
        elif model_ref.startswith("alias:"):
            alias = model_ref.split(":", 1)[1]
            predictor = G2PPredictor.load(alias=alias, device=device)
            model_id = alias
        else:
            predictor = G2PPredictor.load(alias=model_ref, device=device)
            model_id = model_ref
        print("OK")
    except Exception as e:
        print(f"ERRO ({e})")
        return None

    # Aquecimento — descartado
    print(f"  Warmup ({warmup_runs} passes)...", end=" ", flush=True)
    for _ in range(warmup_runs):
        for word in test_words:
            predictor.predict(word)
    print("OK")

    # Loop quente — mínimo dentro do loop para overhead determinístico
    print(f"  Benchmark ({benchmark_runs} passes × {len(test_words)} palavras)...",
          end=" ", flush=True)
    latencies: list = []
    token_counts: list = []
    input_char_counts: list = []
    output_char_counts: list = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(benchmark_runs):
        for word in test_words:
            t0 = time.perf_counter()
            result = predictor.predict(word)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)         # único append no loop quente
            token_counts.append(len(result.split()))  # custo: split() ~1µs, aceitável
            input_char_counts.append(len(word))
            output_char_counts.append(len(result.replace(" ", "")))

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("OK")

    stats = _analyze(
        latencies,
        token_counts,
        input_char_counts,
        output_char_counts,
        loop_overhead_s,
    )
    stats["alias"] = model_id
    stats["device"] = str(device)
    return stats


# ---------------------------------------------------------------------------
# Impressão dos resultados
# ---------------------------------------------------------------------------

_CV_WARN_THRESHOLD    = 0.15   # CV > 15% → aviso de contention
_THERMAL_WARN_HIGH    = 1.10   # last/first > 10% mais lento → throttling térmico
_THERMAL_WARN_LOW     = 0.90   # last/first > 10% mais rápido → warmup insuficiente


def _hardware_flags(stats: dict) -> list:
    flags = []
    if stats["global_cv"] > _CV_WARN_THRESHOLD:
        flags.append(
            f"[AVISO] CV global {stats['global_cv']*100:.1f}% > 15% — "
            "possível contention de recursos (GPU/CPU compartilhada durante teste)"
        )
    r = stats["thermal_ratio"]
    if r > _THERMAL_WARN_HIGH:
        flags.append(
            f"[AVISO] Thermal: últimas medições {(r-1)*100:.1f}% mais lentas — "
            "possível throttling térmico. Aumente --warmup ou aguarde hardware esfriar."
        )
    elif r < _THERMAL_WARN_LOW:
        flags.append(
            f"[AVISO] Thermal: últimas medições {(1-r)*100:.1f}% mais rápidas — "
            "warmup insuficiente (hardware ainda acelerando). Aumente --warmup."
        )
    return flags


def print_results(all_results: dict):
    sep = "=" * 100
    print(f"\n{sep}")
    print("RESULTADOS")
    print(sep)

    for alias, by_device in all_results.items():
        if not by_device:
            continue
        print(f"\nModelo: {alias}")
        print("-" * 100)

        # Cabeçalho
        print(f"{'Device':<8} {'Global w/s':>11} {'CI95 w/s':>19} {'Stable w/s':>11} {'Tok/s':>8} "
              f"{'CharIn/s':>10} {'CharOut/s':>11} {'p50 ms':>8} {'p95 ms':>8} {'CV%':>6}")
        print("-" * 100)

        gpu = by_device.get("cuda")
        cpu = by_device.get("cpu")

        for label, s in [("GPU", gpu), ("CPU", cpu)]:
            if s is None:
                continue
            print(
                f"{label:<8} "
                f"{s['throughput_wps']:>10.1f}w "
                f"[{s['ci95_wps_low']:.1f}, {s['ci95_wps_high']:.1f}] "
                f"{s['stable_wps']:>10.1f}w "
                f"{s['throughput_tps']:>7.0f}t "
                f"{s['throughput_cps_in']:>10.0f} "
                f"{s['throughput_cps_out']:>11.0f} "
                f"{s['p50_ms']:>7.2f}ms "
                f"{s['p95_ms']:>7.2f}ms "
                f"{s['global_cv']*100:>5.1f}%"
            )
            # Detalhe de estabilidade e overhead na linha seguinte
            print(
                f"{'':8} overhead calibração: {s['loop_overhead_us']:.2f}µs subtracted | "
                f"lat CI95=[{s['ci95_lat_low_ms']:.2f},{s['ci95_lat_high_ms']:.2f}]ms | "
                f"tok/word={s['tokens_per_word']:.1f} inChar/word={s['chars_in_per_word']:.1f} "
                f"outChar/word={s['chars_out_per_word']:.1f} | "
                f"stable window starts at run {s['stable_window_start']} "
                f"(CV {s['stable_cv']*100:.1f}%) | "
                f"thermal: first={s['mean_first_ms']:.2f}ms last={s['mean_last_ms']:.2f}ms "
                f"ratio={s['thermal_ratio']:.3f}"
            )

        if gpu and cpu:
            ratio_global = gpu["throughput_wps"] / cpu["throughput_wps"]
            ratio_stable = gpu["stable_wps"] / cpu["stable_wps"]
            print(f"{'Speedup':<8} global: {ratio_global:.2f}x | stable: {ratio_stable:.2f}x  "
                  f"(GPU vs CPU)")

        # Avisos de hardware
        for s in [gpu, cpu]:
            if s:
                for flag in _hardware_flags(s):
                    print(f"  {flag}")

        print()

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de inference GPU vs CPU para modelos G2P"
    )
    parser.add_argument("--models",  default="best_per,best_wer",
                        help="Aliases separados por vírgula (padrão: best_per,best_wer)")
    parser.add_argument("--indices", default="",
                        help="Índices de modelos separados por vírgula (ver src/inference_light.py --list)")
    parser.add_argument("--all-models", action="store_true",
                        help="Benchmark de todos os modelos completos disponíveis em models/")
    parser.add_argument("--warmup",  type=int, default=20,
                        help="Passes de warmup descartados (padrão: 20)")
    parser.add_argument("--runs",    type=int, default=200,
                        help="Passes de benchmark (padrão: 200)")
    parser.add_argument("--words",
                        default="computador,português,inteligência,aplicação,desenvolvimento",
                        help="Palavras de teste separadas por vírgula")
    args = parser.parse_args()

    model_refs: List[str] = []
    if args.all_models:
        all_models = _get_complete_model_files()
        model_refs = [f"index:{i}" for i in range(len(all_models))]
    elif args.indices.strip():
        idxs = [s.strip() for s in args.indices.split(",") if s.strip()]
        model_refs = [f"index:{int(i)}" for i in idxs]
    else:
        aliases = [m.strip() for m in args.models.split(",") if m.strip()]
        model_refs = [f"alias:{a}" for a in aliases]

    test_words = [w.strip() for w in args.words.split(",")]

    print("=" * 100)
    print("G2P Inference Benchmark — GPU vs CPU")
    print("=" * 100)
    print(f"Modelos : {', '.join(model_refs)}")
    print(f"Palavras: {', '.join(test_words)}")
    print(f"Warmup  : {args.warmup} passes | Benchmark: {args.runs} passes")

    # Calibração do overhead do loop (feita uma vez, fora dos benchmarks)
    print("\nCalibrando overhead do loop de medição...", end=" ", flush=True)
    overhead = _calibrate_loop_overhead(2000)
    print(f"{overhead*1e6:.2f}µs por iteração (será subtraído de cada latência)")

    # Devices disponíveis
    devices = []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: não disponível (CUDA)")
    devices.append(torch.device("cpu"))

    # Executar benchmarks
    all_results: dict = {}
    for model_ref in model_refs:
        print(f"\n{'─' * 100}")
        print(f"Modelo: {model_ref}")
        print("─" * 100)
        all_results[model_ref] = {}
        for device in devices:
            result = benchmark_model(
                model_ref, device, test_words,
                warmup_runs=args.warmup,
                benchmark_runs=args.runs,
                loop_overhead_s=overhead,
            )
            if result:
                all_results[model_ref][str(device)] = result

    print_results(all_results)


if __name__ == "__main__":
    main()
