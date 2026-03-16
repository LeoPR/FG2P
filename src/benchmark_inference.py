"""Benchmark formal de inferência para modelos G2P.

Concentra as melhores técnicas de benchmark desenvolvidas no projeto:
- warmup descartado
- calibração de overhead do loop de medição
- métricas globais + janela estável
- p50/p95/p99
- throughput em palavras/tokens/chars
- detecção de contention e drift térmico

O benchmark é parte do ecossistema do manager, mas roda de forma autônoma:
ele descobre modelos, conhece seus próprios artefatos, sincroniza performance.json
quando solicitado e expõe status/pêndencias via --list.
"""

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import sys
import torch

sys.path.insert(0, str(Path(__file__).parent))

from file_registry import FileRegistry, list_experiments, ExperimentRecord
from inference_light import G2PPredictor

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("benchmark_inference")

DEFAULT_WORDS = 200
DEFAULT_WARMUP = 20
DEFAULT_RUNS = 200
CV_WARN_THRESHOLD = 0.15
THERMAL_WARN_HIGH = 1.10
THERMAL_WARN_LOW = 0.90


def resolve_devices(device_arg: str) -> List[torch.device]:
    """Resolve devices alvo para benchmark."""
    cuda_available = torch.cuda.is_available()
    if device_arg == "cpu":
        return [torch.device("cpu")]
    if device_arg == "cuda":
        if not cuda_available:
            logger.warning("CUDA solicitado mas não disponível; usando CPU")
            return [torch.device("cpu")]
        return [torch.device("cuda")]
    if device_arg == "both":
        devices: List[torch.device] = []
        if cuda_available:
            devices.append(torch.device("cuda"))
        else:
            logger.warning("CUDA não disponível; benchmark 'both' será apenas CPU")
        devices.append(torch.device("cpu"))
        return devices
    return [torch.device("cuda" if cuda_available else "cpu")]


def get_complete_records() -> List[ExperimentRecord]:
    """Retorna runs completos na mesma ordem dos demais scripts do projeto."""
    return list_experiments(complete_only=True)



def _calibrate_loop_overhead(n: int = 2000) -> float:
    buf = []
    for _ in range(n):
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        buf.append(t1 - t0)
    buf.sort()
    trimmed = buf[n // 10: n - n // 10]
    return sum(trimmed) / len(trimmed)


def _percentile(sorted_data: List[float], p: float) -> float:
    idx = int(len(sorted_data) * p)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def _cv(data: List[float]) -> float:
    if not data:
        return 0.0
    mean = sum(data) / len(data)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance) / mean


def _iqr_filter(samples: List[float], k: float = 1.5):
    """Rejeição de outliers pelas cercas de Tukey (IQR × k).

    Padrão industrial usado em criterion (Rust), pyperf (Python) e JMH (Java).
    Com k=1.5 remove ~0,7% de uma distribuição Gaussiana — apenas extremos reais.
    Requer ≥4 amostras; retorna lista original se abaixo desse mínimo.
    """
    if len(samples) < 4:
        return list(samples)
    s = sorted(samples)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return [x for x in samples if lo <= x <= hi]


def _find_stable_window(latencies: List[float], window_frac: float = 0.20):
    n = len(latencies)
    w = max(10, int(n * window_frac))
    best_cv = float("inf")
    best_start = 0

    for i in range(n - w + 1):
        window = latencies[i: i + w]
        c = _cv(window)
        if c < best_cv:
            best_cv = c
            best_start = i

    stable = latencies[best_start: best_start + w]
    stable_mean = sum(stable) / len(stable)
    return stable_mean, best_cv, best_start


def _thermal_check(latencies: List[float], frac: float = 0.20):
    n = len(latencies)
    w = max(5, int(n * frac))
    mean_first = sum(latencies[:w]) / w
    mean_last = sum(latencies[-w:]) / w
    ratio = mean_last / mean_first if mean_first > 0 else 1.0
    return mean_first, mean_last, ratio


def _analyze(latencies: List[float], token_counts: List[int], input_char_counts: List[int],
             output_char_counts: List[int], loop_overhead_s: float) -> Dict:
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
    wps_ci95_low = (1.0 / ci95_lat_high) if ci95_lat_high > 0 else 0.0
    wps_ci95_high = (1.0 / ci95_lat_low) if ci95_lat_low > 0 else 0.0

    stable_lat, stable_cv, stable_start = _find_stable_window(corrected)
    stable_wps = 1.0 / stable_lat if stable_lat > 0 else 0.0
    mean_first, mean_last, thermal_ratio = _thermal_check(corrected)
    global_cv = _cv(corrected)

    # CV robusto: IQR/mediana — resistente a caudas longas e outliers.
    # Para batch=1, distingue variância de comprimento (genuína, alta)
    # de contention (inflaria também o robust_cv).
    corrected_sorted_q = sorted(corrected)
    nq = len(corrected_sorted_q)
    q1_val = corrected_sorted_q[nq // 4]
    q3_val = corrected_sorted_q[(3 * nq) // 4]
    p50_val = corrected_sorted_q[nq // 2]
    iqr_val = q3_val - q1_val
    robust_cv = iqr_val / p50_val if p50_val > 0 else 0.0

    return {
        "n": n,
        "total_time_s": total_time,
        "mean_lat_ms": mean_lat * 1000,
        "p50_ms": _percentile(corrected_sorted, 0.50) * 1000,
        "p95_ms": _percentile(corrected_sorted, 0.95) * 1000,
        "p99_ms": _percentile(corrected_sorted, 0.99) * 1000,
        "throughput_wps": throughput_wps,
        "throughput_tps": throughput_tps,
        "throughput_cps_in": throughput_cps_in,
        "throughput_cps_out": throughput_cps_out,
        "tokens_per_word": total_tokens / n if n > 0 else 0.0,
        "chars_in_per_word": total_chars_in / n if n > 0 else 0.0,
        "chars_out_per_word": total_chars_out / n if n > 0 else 0.0,
        "ci95_lat_low_ms": ci95_lat_low * 1000,
        "ci95_lat_high_ms": ci95_lat_high * 1000,
        "ci95_wps_low": wps_ci95_low,
        "ci95_wps_high": wps_ci95_high,
        "stable_wps": stable_wps,
        "stable_cv": stable_cv,
        "stable_window_start": stable_start,
        "global_cv": global_cv,
        "robust_cv": robust_cv,
        "thermal_ratio": thermal_ratio,
        "mean_first_ms": mean_first * 1000,
        "mean_last_ms": mean_last * 1000,
        "loop_overhead_us": loop_overhead_s * 1e6,
    }


def _hardware_flags(stats: Dict,
                    cv_threshold: float = CV_WARN_THRESHOLD,
                    thermal_warn_high: float = THERMAL_WARN_HIGH,
                    thermal_warn_low: float = THERMAL_WARN_LOW) -> List[str]:
    """Detecta anomalias de hardware e distingue causas de CV alto.

    CV alto (> 15%) tem três causas distintas:
    1. Variância de comprimento de entrada (batch=1): o LSTM autoregressive
       leva mais passos para palavras longas → CV estrutural inevitável.
       Sinal: robust_cv também alto, mas ausência de outliers extremos.
    2. Amostra insuficiente (batch grande, poucas palavras): CV instável
       porque há poucos chunks por run. Sinal: n / batch_size < 20.
    3. Contention real (OS scheduler, outro processo): CV alto mesmo com
       robust_cv baixo, picos extremos (>5×p50).
    """
    flags = []
    gcv = stats["global_cv"]
    rcv = stats.get("robust_cv", gcv)
    batch_size = stats.get("batch_size", 1)
    n = stats.get("n", 1)
    mean_lat_ms = stats.get("mean_lat_ms", 999.0)

    if gcv > cv_threshold:
        if batch_size == 1:
            # Causa provável: heterogeneidade de comprimento de entrada.
            # O decoder autoregressive processa mais passos para palavras longas.
            # Isso é variância estrutural, não contention — o robust_cv confirma.
            flags.append(
                f"CV {gcv*100:.1f}% (robust_cv={rcv*100:.1f}%) — variância de comprimento de "
                f"entrada: LSTM processa mais passos para palavras longas. "
                f"Use p50 e robust_cv como referência; stable_wps ≈ throughput em palavras típicas."
            )
        else:
            # batch > 1: verificar se há amostras suficientes
            n_chunks = n // batch_size  # chunks independentes (aprox)
            if n_chunks < 100:
                flags.append(
                    f"CV {gcv*100:.1f}% (robust_cv={rcv*100:.1f}%) — amostra insuficiente: "
                    f"~{n_chunks} chunks independentes. Use mais palavras (--words {batch_size*20}) "
                    f"para estatísticas estáveis. robust_cv={rcv*100:.1f}% é mais confiável."
                )
            elif mean_lat_ms < 5.0:
                # Modelo rápido: overhead Python proporcional à latência
                flags.append(
                    f"CV {gcv*100:.1f}% (robust_cv={rcv*100:.1f}%) — overhead Python "
                    f"(~0,1–0,3 ms) representa {0.2/mean_lat_ms*100:.0f}% da latência média "
                    f"({mean_lat_ms:.2f} ms). robust_cv={rcv*100:.1f}% indica dispersão real do hardware."
                )
            else:
                flags.append(
                    f"CV {gcv*100:.1f}% (robust_cv={rcv*100:.1f}%) — dispersão elevada; "
                    f"verificar processos competindo por recursos."
                )

    ratio = stats["thermal_ratio"]
    if ratio > thermal_warn_high:
        flags.append(f"thermal drift +{(ratio - 1) * 100:.1f}% (possível throttling)")
    elif ratio < thermal_warn_low:
        flags.append(f"thermal drift -{(1 - ratio) * 100:.1f}% (warmup insuficiente)")
    return flags


def load_benchmark_artifact(path: Optional[Path]) -> Optional[Dict]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _format_benchmark_status(rec: ExperimentRecord, device: str) -> str:
    artifact = load_benchmark_artifact(rec.latest_benchmark_path(device=device))
    if not artifact:
        return "pendente"
    result = artifact.get("result", {})
    meta = artifact.get("benchmark_metadata", {})
    return (
        f"{result.get('throughput_wps', 0):.1f} w/s "
        f"| p50 {result.get('p50_ms', 0):.2f} ms "
        f"| {meta.get('benchmark_date', '?')}"
    )


def list_models(show_devices: Optional[List[str]] = None) -> List[ExperimentRecord]:
    records = get_complete_records()
    if not records:
        print("Nenhum modelo completo encontrado em models/")
        return []

    devices = show_devices or ["cuda", "cpu"]
    print(f"\n{'='*100}")
    print("MODELOS DISPONÍVEIS PARA BENCHMARK")
    print(f"{'='*100}")
    for i, rec in enumerate(records):
        meta = rec.metadata or {}
        name = meta.get("experiment_name", rec.base_name)
        epoch = meta.get("current_epoch", "?")
        best = meta.get("best_loss", 0)
        params = meta.get("total_params", 0)
        print(f"  [{i:2d}] {name}")
        print(f"       epoch={epoch} | loss={best:.4f} | {params:,}p | run={rec.base_name}")
        for device in devices:
            print(f"       benchmark[{device}]: {_format_benchmark_status(rec, device)}")
    print(f"{'='*100}")
    print(f"Mais recente: [{len(records) - 1}]")
    print("Use --index N para rodar um modelo, ou sem --index para todos os completos.\n")
    return records


def benchmark_record(rec: ExperimentRecord, device: torch.device, n_words: int,
                     warmup_runs: int, benchmark_runs: int,
                     loop_overhead_s: float, quantize: bool = False,
                     num_threads: int = None, batch_size: int = 1,
                     adaptive: bool = False, min_runs: int = 10,
                     max_runs: int = 80, cv_target: float = 0.03) -> tuple:
    """Executa benchmark e retorna (stats_dict, raw_records).

    Amostra palavras exclusivamente do split de teste do próprio experimento,
    respeitando os mesmos parâmetros (test_ratio, val_ratio, split_seed) usados
    no treino — idêntico ao que inference.py e eval fazem.

    batch_size=1: modo unitário (baseline) — predict() por palavra.
    batch_size>1: modo batch nativo — predict_batch_native() por chunk.
      Latência registrada por item = tempo_do_chunk / n_items (throughput médio por palavra).

    raw_records — lista de dicts com uma entrada por (run_idx × word_idx):
        run_idx, word_idx, word, latency_s, tokens, chars_in, chars_out
    """
    try:
        predictor = G2PPredictor.load(model_path=rec.pt_path, device=device,
                                      quantize=quantize, num_threads=num_threads)
    except Exception as exc:
        return {"status": "error", "device": str(device), "error": str(exc)}, []

    # Reconstrói o split do experimento — mesmo test_ratio/val_ratio/seed do treino
    data_cfg   = (predictor.metadata or {}).get("config", {}).get("data", {})
    test_ratio = float(data_cfg.get("test_ratio", 0.2))
    val_ratio  = float(data_cfg.get("val_ratio", 0.1))
    seed       = int(data_cfg.get("split_seed", 42))
    split      = predictor.corpus.split(test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)
    test_words_all, _ = split.test_pairs()
    rng        = random.Random(seed)
    test_words = rng.sample(test_words_all, min(n_words, len(test_words_all)))
    logger.info(
        f"  split: test_ratio={test_ratio} val_ratio={val_ratio} seed={seed} "
        f"| test={len(test_words_all)} palavras | amostra={len(test_words)}"
        f" | batch_size={batch_size}"
    )

    # Warmup
    if batch_size > 1:
        for _ in range(warmup_runs):
            predictor.predict_batch_native(test_words, batch_size=batch_size)
    else:
        for _ in range(warmup_runs):
            for word in test_words:
                predictor.predict(word)

    raw_records: List[Dict] = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Modo adaptivo: para quando o CV das medianas por run converge.
    # Modo fixo: roda exatamente benchmark_runs passes (comportamento original).
    n_runs_limit = max_runs if adaptive else benchmark_runs
    run_medians: List[float] = []   # uma mediana por run — para convergência
    converged = False
    runs_executed = 0

    for run_idx in range(n_runs_limit):
        runs_executed += 1
        run_latencies: List[float] = []

        if batch_size > 1:
            for chunk_start in range(0, len(test_words), batch_size):
                chunk = test_words[chunk_start:chunk_start + batch_size]
                t0 = time.perf_counter()
                results = predictor.predict_batch_native(chunk, batch_size=batch_size)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                per_item_s = (t1 - t0) / len(chunk)
                run_latencies.append(per_item_s)
                for word_idx, (word, result) in enumerate(zip(chunk, results)):
                    raw_records.append({
                        "run_idx":   run_idx,
                        "word_idx":  chunk_start + word_idx,
                        "word":      word,
                        "latency_s": per_item_s,
                        "tokens":    len(result.split()),
                        "chars_in":  len(word),
                        "chars_out": len(result.replace(" ", "")),
                    })
        else:
            for word_idx, word in enumerate(test_words):
                t0 = time.perf_counter()
                result = predictor.predict(word)
                t1 = time.perf_counter()
                lat = t1 - t0
                run_latencies.append(lat)
                raw_records.append({
                    "run_idx":   run_idx,
                    "word_idx":  word_idx,
                    "word":      word,
                    "latency_s": lat,
                    "tokens":    len(result.split()),
                    "chars_in":  len(word),
                    "chars_out": len(result.replace(" ", "")),
                })

        if adaptive and run_latencies:
            # Mediana do run como representante (robusto a outliers intra-run)
            s = sorted(run_latencies)
            run_medians.append(s[len(s) // 2])

            if len(run_medians) >= min_runs:
                clean = _iqr_filter(run_medians)
                if len(clean) >= max(4, min_runs // 2) and _cv(clean) < cv_target:
                    converged = True
                    break

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Rejeição de outliers a nível de amostra (aplica sempre em modo adaptivo,
    # opcional em modo fixo — desativado para não alterar comportamento histórico)
    n_raw = len(raw_records)
    if adaptive:
        all_lats = [r["latency_s"] for r in raw_records]
        if len(all_lats) >= 4:
            s = sorted(all_lats)
            n = len(s)
            q1, q3 = s[n // 4], s[(3 * n) // 4]
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            raw_records = [r for r in raw_records if lo <= r["latency_s"] <= hi]

    n_outliers = n_raw - len(raw_records)

    latencies          = [r["latency_s"]  for r in raw_records]
    token_counts       = [r["tokens"]     for r in raw_records]
    input_char_counts  = [r["chars_in"]   for r in raw_records]
    output_char_counts = [r["chars_out"]  for r in raw_records]

    stats = _analyze(latencies, token_counts, input_char_counts, output_char_counts, loop_overhead_s)
    stats.update({"status": "success", "device": str(device), "batch_size": batch_size})

    if adaptive:
        final_cv = _cv(_iqr_filter(run_medians)) if len(run_medians) >= 2 else 0.0
        stats.update({
            "adaptive": True,
            "runs_executed": runs_executed,
            "max_runs": max_runs,
            "min_runs": min_runs,
            "cv_target": cv_target,
            "converged": converged,
            "final_run_cv": final_cv,
            "n_samples_raw": n_raw,
            "n_samples_clean": len(raw_records),
            "n_outliers_removed": n_outliers,
        })

    return stats, raw_records


def _write_raw_csv(path: Path, raw_records: List[Dict], loop_overhead_s: float) -> None:
    """Grava medições brutas por palavra em CSV para reanálise offline.

    Colunas:
        run_idx, word_idx, word, latency_s, latency_corrected_s,
        tokens, chars_in, chars_out
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "run_idx", "word_idx", "word",
            "latency_s", "latency_corrected_s",
            "tokens", "chars_in", "chars_out",
        ])
        for row in raw_records:
            corrected = max(0.0, row["latency_s"] - loop_overhead_s)
            writer.writerow([
                row["run_idx"], row["word_idx"], row["word"],
                f"{row['latency_s']:.9f}", f"{corrected:.9f}",
                row["tokens"], row["chars_in"], row["chars_out"],
            ])


def _write_benchmark_artifact(rec: ExperimentRecord, entry: Dict,
                               raw_records: List[Dict], benchmark_meta: Dict,
                               batch_size: int = 1) -> Path:
    """Grava JSON do artefato + sidecar CSV com medições brutas.

    Para batch_size=1 (modo baseline): usa o caminho padrão (compatível com --list e display).
    Para batch_size>1 (modo sweep): inclui sufixo '_bN' no nome do arquivo para não sobrescrever
    o artefato base e permitir múltiplos batch sizes por modelo/device.
    """
    device_tag = entry["device"]
    base_path     = rec._reg.get_benchmark_path(device_tag)
    base_csv_path = rec._reg.get_benchmark_raw_csv_path(device_tag)
    if batch_size > 1:
        path     = base_path.parent / f"{base_path.stem}_b{batch_size}{base_path.suffix}"
        csv_path = base_csv_path.parent / f"{base_csv_path.stem}_b{batch_size}{base_csv_path.suffix}"
    else:
        path     = base_path
        csv_path = base_csv_path
    loop_overhead_s = benchmark_meta.get("loop_overhead_us", 0.0) / 1e6
    _write_raw_csv(csv_path, raw_records, loop_overhead_s)
    payload = {
        "artifact_type": "inference_benchmark",
        "generated_by": "python src/benchmark_inference.py",
        "base_name": rec.base_name,
        "experiment": rec.exp_name,
        "raw_csv_path": csv_path.as_posix(),
        "benchmark_metadata": benchmark_meta,
        "result": entry,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _build_run_output_path(args, selected_records: List[ExperimentRecord]) -> Path:
    if args.output is not None:
        return args.output
    if args.index is not None and selected_records:
        label = f"index_{args.index}_{selected_records[0].exp_name}"
    elif args.dry_run > 0:
        label = f"dryrun_{args.dry_run}"
    else:
        label = "all"
    return FileRegistry.get_benchmark_run_path(run_label=label)


def _sync_performance_with_benchmark(entries: List[Dict], benchmark_meta: Dict) -> None:
    from manager._constants import PERFORMANCE_PATH

    perf_path = PERFORMANCE_PATH
    if not perf_path.exists():
        logger.warning("performance.json não encontrado; sync ignorado")
        return

    from manager._mappings import map_experiment_to_name

    try:
        perf = json.loads(perf_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Falha ao ler performance.json: {exc}")
        return

    models = {m.get("name"): m for m in perf.get("fg2p_models", []) if m.get("name")}
    updates = 0
    for entry in entries:
        if entry.get("status") != "success":
            continue
        perf_name = map_experiment_to_name(entry.get("experiment", ""))
        if not perf_name or perf_name not in models:
            continue

        target = models[perf_name]
        summary = target.get("benchmark_summary") or {"source": "src/benchmark_inference.py", "devices": {}}
        if not isinstance(summary.get("devices"), dict):
            summary = {"source": "src/benchmark_inference.py", "devices": {}}

        device_key = entry["device"]
        summary["last_updated"] = benchmark_meta.get("benchmark_date")
        summary["devices"][device_key] = {
            "artifact_path": entry.get("artifact_path"),
            "throughput_wps": entry.get("throughput_wps"),
            "stable_wps": entry.get("stable_wps"),
            "p50_ms": entry.get("p50_ms"),
            "p95_ms": entry.get("p95_ms"),
            "global_cv": entry.get("global_cv"),
            "thermal_ratio": entry.get("thermal_ratio"),
            "test_words": benchmark_meta.get("test_words"),
            "warmup_runs": benchmark_meta.get("warmup_runs"),
            "benchmark_runs": benchmark_meta.get("benchmark_runs"),
        }

        primary_device = summary.get("primary_device")
        if primary_device is None or device_key == "cuda":
            summary["primary_device"] = device_key
            target["inference_speed_wps"] = round(entry.get("throughput_wps", 0.0), 2)
            target["latency_avg_ms"] = round(entry.get("mean_lat_ms", 0.0), 2)
            target["total_time_s"] = round(entry.get("total_time_s", 0.0), 2)

        if target.get("benchmark_summary") != summary:
            target["benchmark_summary"] = summary
            updates += 1

    if updates:
        perf_path.write_text(json.dumps(perf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info(f"performance.json atualizado com {updates} benchmark(s)")


def analyze_from_artifact(artifact_path: Path) -> None:
    """Reanalisar benchmark salvo sem recarregar o modelo.

    Lê o JSON do artefato → localiza o CSV bruto via 'raw_csv_path' → recalcula
    todas as estatísticas. Útil para aplicar thresholds diferentes, verificar os
    dados brutos ou diagnosticar avisos de contention sem nova execução.

    Uso:
        python src/benchmark_inference.py --analyze results/exp104d_.../benchmark_..._cuda.json
    """
    if not artifact_path.exists():
        logger.error(f"Artefato não encontrado: {artifact_path}")
        return

    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error(f"Falha ao ler artefato: {exc}")
        return

    # Aceita também artefato agregado de execução (results/benchmarks/benchmark_run_*.json)
    # e redireciona para o artefato por modelo quando houver um único resultado.
    if "raw_csv_path" not in artifact and "results" in artifact:
        results = artifact.get("results", [])
        if len(results) == 1 and isinstance(results[0], dict):
            nested_path = results[0].get("artifact_path")
            if nested_path:
                nested_artifact = Path(nested_path)
                if not nested_artifact.exists():
                    logger.error(
                        f"Artefato agregado aponta para arquivo inexistente: {nested_artifact}"
                    )
                    return
                logger.info(
                    "Artefato agregado detectado; analisando artefato por modelo/device: "
                    f"{nested_artifact.name}"
                )
                artifact_path = nested_artifact
                try:
                    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.error(f"Falha ao ler artefato referenciado: {exc}")
                    return
        elif len(results) > 1:
            logger.error(
                "Artefato agregado com múltiplos resultados. Use o campo 'artifact_path' "
                "de cada entrada para analisar um modelo/device específico."
            )
            return

    raw_csv_str = artifact.get("raw_csv_path")
    if not raw_csv_str:
        logger.error(
            "Artefato não contém 'raw_csv_path'. "
            "Rode o benchmark novamente para gerar o CSV bruto de medições."
        )
        return

    raw_csv_path = Path(raw_csv_str)
    if not raw_csv_path.exists():
        logger.error(f"CSV bruto não encontrado: {raw_csv_path}")
        return

    meta = artifact.get("benchmark_metadata", {})
    loop_overhead_s = meta.get("loop_overhead_us", 0.0) / 1e6

    with open(raw_csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        logger.error("CSV bruto está vazio.")
        return

    latencies          = [float(r["latency_s"])  for r in rows]
    token_counts       = [int(r["tokens"])        for r in rows]
    input_char_counts  = [int(r["chars_in"])      for r in rows]
    output_char_counts = [int(r["chars_out"])     for r in rows]

    stats = _analyze(latencies, token_counts, input_char_counts,
                     output_char_counts, loop_overhead_s)
    flags = _hardware_flags(stats)

    device = artifact.get("result", {}).get("device", "?")
    base   = artifact.get("base_name", "?")
    exp    = artifact.get("experiment", "?")
    sep    = "=" * 80

    print(f"\n{sep}")
    print(f"ANÁLISE OFFLINE  —  {artifact_path.name}")
    print(sep)
    print(f"  Modelo          : {base}")
    print(f"  Experimento     : {exp}")
    print(f"  Device          : {device}")
    print(f"  N medições      : {stats['n']:,}")
    print(f"  Overhead calib  : {stats['loop_overhead_us']:.2f} µs")
    print(f"  CSV bruto       : {raw_csv_path}")
    print()
    print(f"  Throughput      : {stats['throughput_wps']:.1f} w/s  (global)")
    print(f"                  : {stats['stable_wps']:.1f} w/s  "
          f"(janela estável, CV={stats['stable_cv']*100:.1f}%)")
    print(f"  Latência média  : {stats['mean_lat_ms']:.3f} ms")
    print(f"  p50 / p95 / p99 : {stats['p50_ms']:.2f} / {stats['p95_ms']:.2f} "
          f"/ {stats['p99_ms']:.2f} ms")
    print(f"  IC95 latência   : [{stats['ci95_lat_low_ms']:.3f}, "
          f"{stats['ci95_lat_high_ms']:.3f}] ms")
    print(f"  IC95 throughput : [{stats['ci95_wps_low']:.1f}, "
          f"{stats['ci95_wps_high']:.1f}] w/s")
    print(f"  Chars/s entrada : {stats['throughput_cps_in']:.1f} c/s")
    print(f"  Chars/s saída   : {stats['throughput_cps_out']:.1f} c/s")
    print(f"  Global CV       : {stats['global_cv']*100:.1f}%")
    print(f"  Thermal ratio   : {stats['thermal_ratio']:.3f}  "
          f"(início / fim 20%: {stats['mean_first_ms']:.2f} / {stats['mean_last_ms']:.2f} ms)")
    print()
    if flags:
        print("  Avisos:")
        for flag in flags:
            print(f"    [!] {flag}")
    else:
        print("  Sem avisos de hardware.")
    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark formal de inferência para modelos G2P",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python src/benchmark_inference.py --list
  python src/benchmark_inference.py --index 18
  python src/benchmark_inference.py --device both --index 18 --force
  python src/benchmark_inference.py --device auto --force --update-performance
""",
    )
    parser.add_argument("--analyze", type=Path, default=None, metavar="ARTIFACT_JSON",
                        help="Reanalisar artefato JSON já gravado sem carregar o modelo "
                             "(requer CSV bruto sidecar no mesmo diretório)")
    parser.add_argument("--list", action="store_true", help="Lista modelos e status de benchmark")
    parser.add_argument("--index", type=int, default=None, help="Índice do modelo (mesma ordem de --list)")
    parser.add_argument("--dry-run", type=int, default=0, help="Seleciona apenas os últimos N modelos")
    parser.add_argument("--force", action="store_true", help="Regrava benchmark mesmo se já existir")
    parser.add_argument("--output", type=Path, default=None, help="JSON agregado da execução atual")
    parser.add_argument("--words", type=int, default=DEFAULT_WORDS, help="Quantidade de palavras do dicionário")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Passes de warmup")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Passes de benchmark")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "both"], default="auto",
                        help="Device alvo do benchmark")
    parser.add_argument("--update-performance", action="store_true",
                        help="Sincroniza speed/resumo do benchmark em performance.json")
    parser.add_argument("--quantize", action="store_true",
                        help="Ativa INT8 dynamic quantization no CPU. "
                             "ATENCAO: testado em Xeon+Windows, resultou em regressao de 38%%. "
                             "Util apenas para teste em ARM ou hardware diferente — ver doc 016.")
    parser.add_argument("--threads", type=int, default=None, metavar="N",
                        help="Threads intra-op do PyTorch para CPU (default: sistema/MKL). "
                             "ATENCAO: em Xeon, threads=1 causou regressao de 61%% — MKL "
                             "multi-thread ja e otimo para LSTM hidden=384. Ver doc 016.")
    parser.add_argument("--batch-size", type=int, default=1, metavar="N",
                        help="Batch size para inferencia nativa (default: 1 = unitario/baseline). "
                             "Valores >1 usam predict_batch_native(): 1 chamada ao encoder "
                             "+ max_len passos do decoder em paralelo por chunk. "
                             "Testar: 4, 8, 16, 32, 64. Latencia reportada = tempo_chunk/n_items.")
    parser.add_argument("--batch-sizes", type=str, default=None, metavar="B1,B2,...",
                        help="Sweep de batch sizes separados por virgula (ex: '1,4,8,16,32,64,128'). "
                             "Roda o benchmark completo para cada batch e salva artefatos separados "
                             "(sufixo _bN no arquivo). Incompativel com --batch-size.")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep padrao de batch sizes: 1, 4, 8, 16, 32, 64, 128. "
                             "Equivalente a --batch-sizes 1,4,8,16,32,64,128. "
                             "Gera sweep_summary.csv com todos os resultados agrupados.")
    parser.add_argument("--sweep-gpu", action="store_true",
                        help="Sweep estendido para GPU: 1, 4, 8, 16, 32, 64, 128, 256, 512. "
                             "Inclui range alem da saturacao de CPU (~batch=64) ate a saturacao "
                             "de GPU (~batch=512 para RTX 3060 + LSTM hidden=384). "
                             "Nao recomendado para CPU pois 256+ nao agrega informacao nova.")
    # --- Controle de runs adaptivo ---
    parser.add_argument("--adaptive", action="store_true",
                        help="Ativa modo adaptivo: para quando o CV das medianas por run converge "
                             "(cv_target) ou o limite max-runs é atingido. "
                             "Usa rejeicao de outliers por IQR (cercas de Tukey k=1.5) "
                             "a nivel de run e de amostra individual. "
                             "Substitui --runs pelo par --min-runs / --max-runs. "
                             "Tecnica padrao de criterion (Rust), pyperf (Python) e JMH (Java).")
    parser.add_argument("--min-runs", type=int, default=10, metavar="N",
                        help="Minimo de runs antes de checar convergencia (default: 10). "
                             "So tem efeito com --adaptive.")
    parser.add_argument("--max-runs", type=int, default=80, metavar="N",
                        help="Maximo de runs no modo adaptivo (default: 80). "
                             "So tem efeito com --adaptive; com --runs fixo e ignorado.")
    parser.add_argument("--cv-target", type=float, default=0.03, metavar="FLOAT",
                        help="CV alvo para convergencia no modo adaptivo (default: 0.03 = 3%%). "
                             "CV = std/mean das medianas por run apos filtragem IQR. "
                             "Valores tipicos: 0.02 (rigoroso), 0.03 (padrao), 0.05 (rapido).")
    args = parser.parse_args()

    selected_devices = [str(d) for d in resolve_devices(args.device)]

    if args.analyze is not None:
        analyze_from_artifact(args.analyze)
        return

    if args.list:
        list_models(show_devices=selected_devices if args.device != "auto" else ["cuda", "cpu"])
        return

    # Determina lista de batch sizes para o run
    if args.sweep_gpu:
        batch_sizes_list = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    elif args.sweep:
        batch_sizes_list = [1, 4, 8, 16, 32, 64, 128]
    elif args.batch_sizes is not None:
        try:
            batch_sizes_list = [int(x.strip()) for x in args.batch_sizes.split(",")]
        except ValueError:
            logger.error("--batch-sizes deve ser uma lista de inteiros separados por virgula (ex: '1,4,8,16,32,64,128')")
            return
    else:
        batch_sizes_list = [args.batch_size]

    is_sweep = len(batch_sizes_list) > 1

    records = get_complete_records()
    if not records:
        logger.error("Nenhum modelo completo disponível para benchmark")
        return

    selected = list(enumerate(records))
    if args.index is not None:
        if args.index < 0 or args.index >= len(records):
            logger.error(f"Índice {args.index} inválido. Use --list para ver 0-{len(records) - 1}.")
            return
        selected = [(args.index, records[args.index])]
    elif args.dry_run > 0:
        selected = selected[-args.dry_run:]

    pending: List[tuple[int, ExperimentRecord]] = []
    skipped: List[Dict] = []
    for idx, rec in selected:
        missing = [device for device in selected_devices if not rec.has_benchmark(device)]
        if args.force or missing:
            pending.append((idx, rec))
        else:
            skipped.append({"index": idx, "base_name": rec.base_name, "reason": "já benchmarkado"})

    logger.info("=" * 100)
    logger.info("BENCHMARK FORMAL DE INFERÊNCIA")
    logger.info("=" * 100)
    logger.info(f"Dispositivos   : {', '.join(selected_devices)}")
    if args.adaptive:
        logger.info(f"Warmup         : {args.warmup} | Modo: ADAPTIVO "
                    f"(min={args.min_runs} max={args.max_runs} runs, cv_target={args.cv_target*100:.1f}%)")
    else:
        logger.info(f"Warmup/Runs    : {args.warmup}/{args.runs}")
    logger.info(f"Palavras teste : {args.words}")
    if is_sweep:
        logger.info(f"Batch sizes    : {batch_sizes_list} (modo sweep)")
    else:
        logger.info(f"Batch size     : {batch_sizes_list[0]}")
    logger.info(f"Selecionados   : {len(selected)} | Pendentes: {len(pending)} | Pulados: {len(skipped)}")

    if not pending and not args.force:
        logger.info("Nenhum benchmark pendente. Use --force para regravar.")
        return

    overhead = _calibrate_loop_overhead(2000)
    logger.info(f"Overhead calibrado: {overhead * 1e6:.2f} µs por iteração")

    # CSV incremental — aberto antes do loop para gravar linha a linha (crash-safe)
    sweep_csv_path: Optional[Path] = None
    sweep_csv_cols = ["experiment", "model_index", "device", "batch_size",
                      "stable_wps", "throughput_wps", "p50_ms", "p95_ms",
                      "global_cv", "thermal_ratio", "ci95_wps_low", "ci95_wps_high"]
    if is_sweep:
        sweep_ts = time.strftime("%Y%m%d_%H%M%S")
        sweep_csv_name = f"sweep_summary_{sweep_ts}.csv"
        if args.index is not None and pending:
            sweep_csv_name = f"sweep_summary_{pending[0][1].exp_name}_{sweep_ts}.csv"
        sweep_csv_path = FileRegistry.get_benchmark_run_path(run_label="__sweep_tmp__").parent / sweep_csv_name
        sweep_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(sweep_csv_cols)
        logger.info(f"Sweep CSV (incremental): {sweep_csv_path}")

    # Aviso de chunks insuficientes para batch sizes grandes
    max_batch = max(batch_sizes_list)
    if max_batch > args.words:
        n_chunks = max(1, args.words // max_batch)
        logger.warning(
            f"batch_size={max_batch} > words={args.words}: apenas ~{n_chunks} chunk(s) por run. "
            f"Estatísticas podem ser instáveis. Considere --words {max_batch * 10} para ≥10 chunks."
        )

    all_results_entries: List[Dict] = []

    for batch_idx, current_batch_size in enumerate(batch_sizes_list):
        if is_sweep:
            logger.info(f"\n{'='*60}")
            logger.info(f"SWEEP [{batch_idx+1}/{len(batch_sizes_list)}] batch_size={current_batch_size}")
            logger.info(f"{'='*60}")

        run_metadata = {
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "devices": selected_devices,
            "test_words": args.words,
            "warmup_runs": args.warmup,
            "benchmark_runs": args.runs,
            "adaptive": args.adaptive,
            "adaptive_min_runs": args.min_runs if args.adaptive else None,
            "adaptive_max_runs": args.max_runs if args.adaptive else None,
            "adaptive_cv_target": args.cv_target if args.adaptive else None,
            "loop_overhead_us": overhead * 1e6,
            "force": args.force,
            "selected_models": len(pending),
            "cpu_quantize": args.quantize,
            "cpu_threads": args.threads,
            "batch_size": current_batch_size,
            "sweep_mode": is_sweep,
            "sweep_batch_sizes": batch_sizes_list if is_sweep else None,
        }

        batch_results_entries: List[Dict] = []

        for ordinal, (model_index, rec) in enumerate(pending, 1):
            meta = rec.metadata or {}
            experiment_name = meta.get("experiment_name", rec.exp_name)
            params = meta.get("total_params", 0)
            logger.info(f"\n[{ordinal}/{len(pending)}] [{model_index}] {experiment_name}")
            for device in resolve_devices(args.device):
                device_key = str(device)
                if not args.force and not is_sweep and rec.has_benchmark(device_key):
                    logger.info(f"  {device_key}: pulado (artefato já existe)")
                    continue

                stats, raw_records_list = benchmark_record(
                    rec,
                    device=device,
                    n_words=args.words,
                    warmup_runs=args.warmup,
                    benchmark_runs=args.runs,
                    loop_overhead_s=overhead,
                    quantize=args.quantize,
                    num_threads=args.threads,
                    batch_size=current_batch_size,
                    adaptive=args.adaptive,
                    min_runs=args.min_runs,
                    max_runs=args.max_runs,
                    cv_target=args.cv_target,
                )

                entry = {
                    "model_index": model_index,
                    "base_name": rec.base_name,
                    "experiment": experiment_name,
                    "params": params,
                }
                entry.update(stats)

                if entry["status"] == "success":
                    artifact_path = _write_benchmark_artifact(
                        rec, entry, raw_records_list, run_metadata,
                        batch_size=current_batch_size
                    )
                    entry["artifact_path"] = artifact_path.as_posix()
                    flags = _hardware_flags(entry)
                    logger.info(
                        f"  {device_key}: {entry['throughput_wps']:.1f} w/s "
                        f"| stable {entry['stable_wps']:.1f} w/s "
                        f"| p50 {entry['p50_ms']:.2f} ms | p95 {entry['p95_ms']:.2f} ms"
                    )
                    if entry.get("adaptive"):
                        conv_mark = "✓ convergiu" if entry.get("converged") else "✗ limite atingido"
                        logger.info(
                            f"  {device_key}: [adaptivo] {entry['runs_executed']}/{entry['max_runs']} runs "
                            f"| cv_runs={entry['final_run_cv']*100:.1f}% {conv_mark} "
                            f"| {entry['n_outliers_removed']} outliers removidos "
                            f"({entry['n_samples_clean']}/{entry['n_samples_raw']} amostras)"
                        )
                    if flags:
                        logger.info(f"  {device_key}: avisos -> {'; '.join(flags)}")

                    if is_sweep and sweep_csv_path is not None:
                        # Grava linha imediatamente — crash-safe
                        row = {
                            "experiment": experiment_name,
                            "model_index": model_index,
                            "device": device_key,
                            "batch_size": current_batch_size,
                            "stable_wps": round(entry["stable_wps"], 2),
                            "throughput_wps": round(entry["throughput_wps"], 2),
                            "p50_ms": round(entry["p50_ms"], 3),
                            "p95_ms": round(entry["p95_ms"], 3),
                            "global_cv": round(entry["global_cv"], 4),
                            "thermal_ratio": round(entry["thermal_ratio"], 4),
                            "ci95_wps_low": round(entry["ci95_wps_low"], 2),
                            "ci95_wps_high": round(entry["ci95_wps_high"], 2),
                        }
                        with open(sweep_csv_path, "a", newline="", encoding="utf-8") as fh:
                            csv.writer(fh).writerow([row[c] for c in sweep_csv_cols])
                else:
                    logger.warning(f"  {device_key}: falha -> {entry.get('error', '?')}")

                batch_results_entries.append(entry)
                all_results_entries.append(entry)

        # Salva artefato agregado por batch_size (em sweep) ou único (normal)
        if is_sweep:
            batch_label = f"sweep_b{current_batch_size}"
            if args.index is not None:
                batch_label = f"index_{args.index}_{pending[0][1].exp_name}_b{current_batch_size}"
            batch_output = FileRegistry.get_benchmark_run_path(run_label=batch_label)
        else:
            batch_output = _build_run_output_path(args, [rec for _, rec in pending])

        batch_run_result = {
            "metadata": run_metadata,
            "results": batch_results_entries,
            "skipped": skipped if not is_sweep else [],
        }
        batch_output.parent.mkdir(parents=True, exist_ok=True)
        batch_output.write_text(json.dumps(batch_run_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if is_sweep:
            logger.info(f"  Artefato batch={current_batch_size}: {batch_output}")

    if is_sweep and sweep_csv_path is not None:
        logger.info(f"\nSweep summary CSV (completo): {sweep_csv_path}")

    if args.update_performance and not is_sweep:
        _sync_performance_with_benchmark(all_results_entries, {
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_words": args.words,
            "warmup_runs": args.warmup,
            "benchmark_runs": args.runs,
        })

    successful = [r for r in all_results_entries if r.get("status") == "success"]
    failed = [r for r in all_results_entries if r.get("status") == "error"]
    logger.info("\n" + "=" * 100)
    logger.info("RESUMO")
    logger.info("=" * 100)
    logger.info(f"Sucessos: {len(successful)} | Falhas: {len(failed)} | Pulados: {len(skipped)}")
    if successful and not is_sweep:
        wps = [r["throughput_wps"] for r in successful]
        logger.info(f"Throughput global: {min(wps):.1f} — {max(wps):.1f} w/s | média {sum(wps)/len(wps):.1f} w/s")
        output_path = _build_run_output_path(args, [rec for _, rec in pending])
        logger.info(f"Artefato agregado: {output_path}")
    elif is_sweep:
        logger.info(f"Sweep: {len(batch_sizes_list)} batch sizes × {len(pending)} modelos × {len(selected_devices)} devices")
        if sweep_csv_path is not None:
            logger.info(f"Sweep summary CSV: {sweep_csv_path}")


if __name__ == "__main__":
    main()
