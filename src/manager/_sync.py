"""
Sincronização de dados derivados.

sync_performance()  — lê evaluation_*.txt → atualiza performance.json
                      (absorve src/update_performance.py)
rebuild_registry()  — lê metadata.json + evaluation_*.txt → gera model_registry.json
"""
import json
from datetime import datetime

from ._constants import PERFORMANCE_PATH, REGISTRY_PATH, RESULTS_DIR, MODELS_DIR
from ._mappings  import (map_experiment_to_name,
                         parse_eval_metrics, parse_eval_experiment, parse_eval_timestamp)


# =============================================================================
# sync_performance — atualiza docs/report/performance.json
# =============================================================================

def sync_performance(filter_exp: str = None, dry_run: bool = False,
                     update_meta: str = None, include_timestamp: bool = False):
    """
    Lê todos os evaluation_*.txt e sincroniza PER/WER/Accuracy em performance.json.

    Args:
        filter_exp:       substring para filtrar arquivos de avaliação
        dry_run:          mostra diff sem gravar
        update_meta:      atualiza last_updated/revision com essa string
        include_timestamp: atualiza inference_completed mesmo sem mudança de métricas
    """
    if not PERFORMANCE_PATH.exists():
        print(f"[ERRO] Nao encontrado: {PERFORMANCE_PATH}")
        return

    eval_files = sorted(RESULTS_DIR.glob("**/evaluation_*.txt"))
    if filter_exp:
        eval_files = [p for p in eval_files if filter_exp in p.name]
    if not eval_files:
        print("Nenhum evaluation_*.txt encontrado.")
        return

    perf   = json.loads(PERFORMANCE_PATH.read_text(encoding="utf-8"))
    models = perf.get("fg2p_models", [])
    updates, meta_updated = [], False

    for eval_path in eval_files:
        exp_name  = parse_eval_experiment(eval_path)
        if not exp_name:
            print(f"[SKIP] {eval_path.name}: campo Experiment nao encontrado")
            continue
        perf_name = map_experiment_to_name(exp_name)
        if not perf_name:
            print(f"[SKIP] {eval_path.name}: sem mapeamento (adicionar a EXPERIMENT_TO_NAME)")
            continue
        metrics   = parse_eval_metrics(eval_path)
        if not metrics:
            print(f"[SKIP] {eval_path.name}: metricas nao encontradas")
            continue
        timestamp = parse_eval_timestamp(eval_path)

        for entry in models:
            if entry.get("name") != perf_name:
                continue
            current = {k: entry.get(k) for k in ("per", "wer", "accuracy")}
            desired = {k: metrics.get(k) for k in ("per", "wer", "accuracy")}
            needs_update = current != desired
            ts_update    = include_timestamp and timestamp and entry.get("inference_completed") != timestamp
            if needs_update or ts_update:
                entry.update(desired)
                if timestamp:
                    entry["inference_completed"] = timestamp
                updates.append((eval_path.name, perf_name, current, desired))
            break

    if update_meta:
        perf["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        perf["revision"]     = update_meta
        meta_updated         = True

    if not updates and not meta_updated:
        print("[SYNCED] performance.json ja esta sincronizado.")
        return

    if dry_run:
        print("[DIFF] Alteracoes sugeridas:")
        for fname, name, _before, after in updates:
            print(f"  {fname} -> {name}: {after}")
        if meta_updated:
            print("  metadata: last_updated/revision seriam atualizados")
        return

    PERFORMANCE_PATH.write_text(
        json.dumps(perf, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print(f"[OK] performance.json: {len(updates)} entrada(s) sincronizada(s)")


# =============================================================================
# rebuild_registry — gera models/model_registry.json
# =============================================================================

def rebuild_registry(manager):
    """
    Reconstrói model_registry.json a partir dos metadados e arquivos de avaliação.

    Regra de promoção: N_test >= 5000 para claims best_per / best_wer.
    exp107 → comparison_latphon (N < 5000, não elegível).
    """
    MIN_TEST = 5000

    candidates = []
    for exp in manager.experiments:
        if "model_checkpoint" not in exp.artifacts:
            continue
        if "evaluation_txt" not in exp.artifacts:
            continue
        meta    = exp.get_metadata() or {}
        metrics = parse_eval_metrics(exp.artifacts["evaluation_txt"]["path"])
        if not metrics:
            continue
        n_test   = meta.get("dataset", {}).get("test_size", 0)
        exp_name = meta.get("experiment_name", exp.base_name.split("__")[0])
        run_id   = meta.get("run_id", "")
        uses_sep = bool(meta.get("config", {}).get("data", {})
                        .get("keep_syllable_separators", False))
        candidates.append({
            "experiment": exp_name, "run_id": run_id,
            "per": metrics.get("per", 999) / 100,
            "wer": metrics.get("wer", 999) / 100,
            "n_test": n_test, "uses_separators": uses_sep,
            "eligible": n_test >= MIN_TEST,
        })

    eligible = [c for c in candidates if c["eligible"]]
    best_per = min(eligible, key=lambda c: c["per"])  if eligible else None
    best_wer = min(eligible, key=lambda c: c["wer"])  if eligible else None

    def _alias(entry, desc, notes=""):
        if entry is None:
            return {"status": "no_eligible_model", "experiment": None}
        return {
            "experiment": entry["experiment"], "run_id": entry["run_id"],
            "per": round(entry["per"], 6), "wer": round(entry["wer"], 6),
            "n_test": entry["n_test"], "uses_separators": entry["uses_separators"],
            "description": desc, "notes": notes,
        }

    # fast alias: exp106 se tiver checkpoint, senão pending
    exp106_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and "exp106" in d.name]
    exp106_pt   = any(list(d.glob("*.pt")) for d in exp106_dirs) if exp106_dirs else False
    fast_entry  = next((c for c in eligible if "exp106" in c["experiment"]), None)
    fast_alias  = (_alias(fast_entry, "Fastest inference (exp106, no-hyphen, ~2.58x faster).",
                          "exp106 no-hyphen variant.")
                   if exp106_pt and fast_entry else
                   {"status": "pending_retrain", "experiment": None, "run_id": None,
                    "per": None, "wer": None, "n_test": None, "uses_separators": None,
                    "description": "Fastest model (exp106) blocked: no .pt checkpoint.",
                    "notes": "Retrain exp106 to activate."})

    # comparison_latphon: exp107 (fixo, N < 5000)
    exp107 = next((c for c in candidates if "exp107" in c["experiment"]), None)
    latphon = {
        "experiment": exp107["experiment"] if exp107 else "exp107_maxdata_95train",
        "run_id":     exp107["run_id"]     if exp107 else None,
        "per":        round(exp107["per"], 6) if exp107 else 0.0046,
        "wer":        round(exp107["wer"], 6) if exp107 else 0.0556,
        "n_test":     exp107["n_test"]    if exp107 else 960,
        "uses_separators": exp107["uses_separators"] if exp107 else True,
        "description": "Methodological comparison with LatPhon 2025 (N=960). NOT eligible for best_per (N < 5000).",
        "notes": "Small N_test intentional: reproduces LatPhon evaluation protocol.",
    }

    registry = {
        "_version":        "1.0",
        "_generated_by":   "python src/manage_experiments.py --registry",
        "_generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "_promotion_rules": {"min_test_words": MIN_TEST,
                             "note": f"N_test >= {MIN_TEST} required for best_per/best_wer."},
        "aliases": {
            "best_per": _alias(best_per,
                               "Best phoneme accuracy. Recommended for TTS and phonetic alignment.",
                               "DA Loss + syllable separators + structural distance correction."),
            "best_wer": _alias(best_wer,
                               "Best word accuracy. Recommended for NLP, search, word-level tasks.",
                               "DA Loss lambda=0.2, intermediate capacity 9.7M params."),
            "fast":              fast_alias,
            "comparison_latphon": latphon,
        },
    }

    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n[OK] Registry salvo: {REGISTRY_PATH}")
    print(f"Modelos elegíveis (N >= {MIN_TEST}): {len(eligible)}")
    for c in sorted(eligible, key=lambda x: x["per"]):
        tags = ""
        if best_per and c["experiment"] == best_per["experiment"]:
            tags += " [best_per]"
        if best_wer and c["experiment"] == best_wer["experiment"]:
            tags += " [best_wer]"
        print(f"  {c['experiment']}: PER={c['per']:.4%} WER={c['wer']:.4%} N={c['n_test']}{tags}")
    if best_per:
        print(f"\nbest_per → {best_per['experiment']} (PER={best_per['per']:.4%})")
    if best_wer:
        print(f"best_wer → {best_wer['experiment']} (WER={best_wer['wer']:.4%})")
    print(f"fast     → {'PENDENTE' if not (exp106_pt and fast_entry) else 'OK'}")


# =============================================================================
# CLI standalone (para uso direto via python src/manager/_sync.py)
# =============================================================================

def _cli_sync_performance():
    """Entry point para uso como script autônomo (compatibilidade com update_performance.py)."""
    import argparse
    parser = argparse.ArgumentParser(description="Sincroniza performance.json a partir de evaluation_*.txt")
    parser.add_argument("--dry-run",          action="store_true")
    parser.add_argument("--filter",           type=str, default=None)
    parser.add_argument("--update-meta",      type=str, default=None)
    parser.add_argument("--include-timestamp",action="store_true")
    args = parser.parse_args()
    sync_performance(filter_exp=args.filter, dry_run=args.dry_run,
                     update_meta=args.update_meta, include_timestamp=args.include_timestamp)
