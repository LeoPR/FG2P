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
        "thermal_ratio": thermal_ratio,
        "mean_first_ms": mean_first * 1000,
        "mean_last_ms": mean_last * 1000,
        "loop_overhead_us": loop_overhead_s * 1e6,
    }


def _hardware_flags(stats: Dict,
                    cv_threshold: float = CV_WARN_THRESHOLD,
                    thermal_warn_high: float = THERMAL_WARN_HIGH,
                    thermal_warn_low: float = THERMAL_WARN_LOW) -> List[str]:
    flags = []
    if stats["global_cv"] > cv_threshold:
        # Modelos rápidos: overhead Python (encode/decode/tensor-create) ~0,1–0,3 ms
        # já contribui para CV > 15% mesmo sem carga externa, pois a latência total
        # é pequena e o ruído relativo é proporcionalmente maior.
        note = ""
        if stats.get("mean_lat_ms", 999.0) < 5.0:
            note = (
                " — modelo rápido: overhead Python (~0,1–0,3 ms por predição) "
                "é suficiente para elevar CV; contention externa pode coexistir mas "
                "não é a única explicação"
            )
        flags.append(
            f"CV {stats['global_cv']*100:.1f}% > {cv_threshold*100:.0f}%"
            f" (possível contention{note})"
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
                     num_threads: int = 1) -> tuple:
    """Executa benchmark e retorna (stats_dict, raw_records).

    Amostra palavras exclusivamente do split de teste do próprio experimento,
    respeitando os mesmos parâmetros (test_ratio, val_ratio, split_seed) usados
    no treino — idêntico ao que inference.py e eval fazem.

    raw_records — lista de dicts com uma entrada por (run_idx × word_idx):
        run_idx, word_idx, word, latency_s, tokens, chars_in, chars_out
    Ao preservar os registros brutos, toda a análise pode ser refeita
    offline com --analyze sem precisar recarregar o modelo.
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
    )

    for _ in range(warmup_runs):
        for word in test_words:
            predictor.predict(word)

    raw_records: List[Dict] = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    for run_idx in range(benchmark_runs):
        for word_idx, word in enumerate(test_words):
            t0 = time.perf_counter()
            result = predictor.predict(word)
            t1 = time.perf_counter()
            raw_records.append({
                "run_idx":   run_idx,
                "word_idx":  word_idx,
                "word":      word,
                "latency_s": t1 - t0,
                "tokens":    len(result.split()),
                "chars_in":  len(word),
                "chars_out": len(result.replace(" ", "")),
            })

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies          = [r["latency_s"]  for r in raw_records]
    token_counts       = [r["tokens"]     for r in raw_records]
    input_char_counts  = [r["chars_in"]   for r in raw_records]
    output_char_counts = [r["chars_out"]  for r in raw_records]

    stats = _analyze(latencies, token_counts, input_char_counts, output_char_counts, loop_overhead_s)
    stats.update({"status": "success", "device": str(device)})
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
                               raw_records: List[Dict], benchmark_meta: Dict) -> Path:
    """Grava JSON do artefato + sidecar CSV com medições brutas."""
    device_tag = entry["device"]
    path     = rec._reg.get_benchmark_path(device_tag)
    csv_path = rec._reg.get_benchmark_raw_csv_path(device_tag)
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
    parser.add_argument("--threads", type=int, default=1, metavar="N",
                        help="Threads intra-op do PyTorch para CPU (default: 1). "
                             "Para inferencia unitaria sequencial, 1 thread minimiza overhead. "
                             "Use valores maiores para medir impacto em throughput por batch.")
    args = parser.parse_args()

    selected_devices = [str(d) for d in resolve_devices(args.device)]

    if args.analyze is not None:
        analyze_from_artifact(args.analyze)
        return

    if args.list:
        list_models(show_devices=selected_devices if args.device != "auto" else ["cuda", "cpu"])
        return

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
    logger.info(f"Warmup/Runs    : {args.warmup}/{args.runs}")
    logger.info(f"Palavras teste : {args.words}")
    logger.info(f"Selecionados   : {len(selected)} | Pendentes: {len(pending)} | Pulados: {len(skipped)}")

    if not pending and not args.force:
        logger.info("Nenhum benchmark pendente. Use --force para regravar.")
        return

    overhead = _calibrate_loop_overhead(2000)
    logger.info(f"Overhead calibrado: {overhead * 1e6:.2f} µs por iteração")

    results = {
        "metadata": {
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "devices": selected_devices,
            "test_words": args.words,
            "warmup_runs": args.warmup,
            "benchmark_runs": args.runs,
            "loop_overhead_us": overhead * 1e6,
            "force": args.force,
            "selected_models": len(pending),
            "cpu_quantize": args.quantize,
            "cpu_threads": args.threads,
        },
        "results": [],
        "skipped": skipped,
    }

    for ordinal, (model_index, rec) in enumerate(pending, 1):
        meta = rec.metadata or {}
        experiment_name = meta.get("experiment_name", rec.exp_name)
        params = meta.get("total_params", 0)
        logger.info(f"\n[{ordinal}/{len(pending)}] [{model_index}] {experiment_name}")
        for device in resolve_devices(args.device):
            device_key = str(device)
            if not args.force and rec.has_benchmark(device_key):
                logger.info(f"  {device_key}: pulado (artefato já existe)")
                continue

            stats, raw_records = benchmark_record(
                rec,
                device=device,
                n_words=args.words,
                warmup_runs=args.warmup,
                benchmark_runs=args.runs,
                loop_overhead_s=overhead,
                quantize=args.quantize,
                num_threads=args.threads,
            )

            entry = {
                "model_index": model_index,
                "base_name": rec.base_name,
                "experiment": experiment_name,
                "params": params,
            }
            entry.update(stats)

            if entry["status"] == "success":
                artifact_path = _write_benchmark_artifact(rec, entry, raw_records, results["metadata"])
                entry["artifact_path"] = artifact_path.as_posix()
                flags = _hardware_flags(entry)
                logger.info(
                    f"  {device_key}: {entry['throughput_wps']:.1f} w/s "
                    f"| stable {entry['stable_wps']:.1f} w/s "
                    f"| p50 {entry['p50_ms']:.2f} ms | p95 {entry['p95_ms']:.2f} ms"
                )
                if flags:
                    logger.info(f"  {device_key}: avisos -> {'; '.join(flags)}")
            else:
                logger.warning(f"  {device_key}: falha -> {entry.get('error', '?')}")

            results["results"].append(entry)

    output_path = _build_run_output_path(args, [rec for _, rec in pending])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.update_performance:
        _sync_performance_with_benchmark(results["results"], results["metadata"])

    successful = [r for r in results["results"] if r.get("status") == "success"]
    failed = [r for r in results["results"] if r.get("status") == "error"]
    logger.info("\n" + "=" * 100)
    logger.info("RESUMO")
    logger.info("=" * 100)
    logger.info(f"Sucessos: {len(successful)} | Falhas: {len(failed)} | Pulados: {len(skipped)}")
    if successful:
        wps = [r["throughput_wps"] for r in successful]
        logger.info(f"Throughput global: {min(wps):.1f} — {max(wps):.1f} w/s | média {sum(wps)/len(wps):.1f} w/s")
    logger.info(f"Artefato agregado: {output_path}")


if __name__ == "__main__":
    main()
