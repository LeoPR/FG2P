"""
Relatórios de cobertura e orientação de fluxo.

show_missing()       — tabela compacta de gaps (eval / error_analysis / plot)
check_consistency()  — verifica consistência entre disco, performance.json e registry
guide()              — orientação detalhada por experimento
compare_experiment() — compara múltiplos runs do mesmo experimento
show_status()        — detecta experimentos em execução
"""
from datetime import datetime

from ._artifacts import ExperimentStatus
from ._constants import MODELS_DIR, RESULTS_DIR
from ._mappings  import parse_eval_metrics, map_experiment_to_name, load_performance_index


# =============================================================================
# show_missing
# =============================================================================

def show_missing(manager):
    """Tabela compacta: quais experimentos faltam eval / error_analysis / plot."""
    if not manager.experiments:
        print("\nNenhum experimento encontrado.")
        return

    manager.index_map = manager._build_index_map()
    COLS = [
        ("eval",   ("evaluation_txt", "predictions_tsv")),
        ("err_an", ("error_analysis_txt",)),
        ("plot",   ("convergence_plot",)),
        ("bench",  ("benchmark_json",)),
    ]

    complete     = [(idx, exp) for idx, exp in manager.index_map.items()
                    if exp.classify_status() == ExperimentStatus.COMPLETE]
    non_complete = [(idx, exp, exp.classify_status()) for idx, exp in manager.index_map.items()
                    if exp.classify_status() != ExperimentStatus.COMPLETE]

    max_name = max((len(e.base_name) for _, e in complete), default=40)
    max_name = max(max_name, 40)
    header   = f"{'idx':>4}  {'experimento':<{max_name}}  {'eval':^6}  {'err_an':^6}  {'plot':^6}  {'bench':^6}"
    sep      = "-" * len(header)

    print(f"\n{'='*len(header)}\nCOBERTURA DE ARTEFATOS — EXPERIMENTOS COMPLETOS\n{'='*len(header)}")
    print(header)
    print(sep)

    ok_list  = []
    gap_lists = {label: [] for label, _ in COLS}

    for idx, exp in complete:
        arts  = exp.artifacts
        cells = []
        has_gap = False
        for label, keys in COLS:
            present = all(k in arts for k in keys)
            cells.append("OK" if present else "--")
            if not present:
                has_gap = True
                gap_lists[label].append((idx, exp.base_name))
        row    = f"{idx:>4}  {exp.base_name:<{max_name}}  {cells[0]:^6}  {cells[1]:^6}  {cells[2]:^6}  {cells[3]:^6}"
        marker = " <" if has_gap else ""
        print(row + marker)
        if not has_gap:
            ok_list.append(exp.base_name)

    print(sep)

    # Experimentos com resultados mas sem .pt
    results_only = []
    for subdir in sorted(RESULTS_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        if any(subdir.glob("evaluation_*.txt")) and not any((MODELS_DIR / subdir.name).glob("*.pt")):
            results_only.append(subdir.name)

    if non_complete:
        print(f"\n(Nao listados: {len(non_complete)} nao-completos — use --list)")
    if results_only:
        print(f"\nATENCAO: {len(results_only)} experimento(s) com resultados sem .pt:")
        for name in results_only:
            print(f"  {name}")

    print("\nRESUMO DE GAPS\n" + "-" * 50)
    any_gap = False
    for label, _ in COLS:
        gaps = gap_lists[label]
        if gaps:
            any_gap = True
            names   = ", ".join(f"[{i}]" for i, _ in gaps)
            print(f"  {label:8s}: {len(gaps)} exp {names}")
    if not any_gap:
        print("  Todos completamente processados.")

    print("\nCOMANDOS SUGERIDOS\n" + "-" * 50)
    needs_inf  = gap_lists["eval"]
    needs_err  = gap_lists["err_an"]
    needs_plot = gap_lists["plot"]
    if needs_inf:
        for idx, _ in needs_inf:
            print(f"  python src/manage_experiments.py --run {idx} --force")
    if needs_err or needs_plot:
        cheap = {}
        for idx, name in needs_err + needs_plot:
            cheap.setdefault(idx, name)
        for idx in sorted(cheap):
            print(f"  python src/manage_experiments.py --run {idx}")
    if not needs_inf and not needs_err and not needs_plot:
        print("  Nenhuma acao necessaria.")
    print()


# =============================================================================
# guide
# =============================================================================

def guide(manager):
    """Orientação de próximo passo por experimento (verbose)."""
    if not manager.experiments:
        print("\nNenhum experimento encontrado.")
        return

    manager.index_map = manager._build_index_map()
    perf_index = load_performance_index()
    to_infer, to_analyze, to_plot, perf_outdated = [], [], [], []
    latest_eval_mtime = None

    print(f"\n{'='*100}\nORIENTACAO DE FLUXO (NEXT STEPS)\n{'='*100}")

    for idx, exp in manager.index_map.items():
        status    = exp.classify_status()
        artifacts = exp.artifacts
        print(f"\n[{idx}] {exp.base_name}")
        print(f"    Status: {status.upper()}")

        if status == ExperimentStatus.RUNNING:
            print("    → Em execucao. Aguarde o treino terminar.")
            continue
        if status == ExperimentStatus.ORPHAN:
            print("    → Orfao: sem metadados/checkpoint. Sugestao: --clean N.")
            continue
        if status == ExperimentStatus.INCOMPLETE:
            print("    → Incompleto. Sugestao: --clean N ou retreinar.")
            continue

        missing = [k for k in ("evaluation_txt", "predictions_tsv", "error_analysis_txt")
                   if k not in artifacts]
        if not missing:
            print("    → Completo: treino + avaliacao OK.")
        else:
            print(f"    → Faltam: {', '.join(missing)}")
            if any(k in missing for k in ("evaluation_txt", "predictions_tsv")):
                print(f"      Sugestao: python src/inference.py --model {exp.base_name}")
                to_infer.append(exp.base_name)
            if "error_analysis_txt" in missing:
                print(f"      Sugestao: python src/analyze_errors.py --model {exp.base_name}")
                to_analyze.append(exp.base_name)

        if "history_csv" in artifacts and "convergence_plot" not in artifacts:
            print("    → Falta grafico de convergencia")
            print(f"      Sugestao: python src/analysis.py --model-name {exp.base_name}")
            to_plot.append(exp.base_name)

        if "evaluation_txt" in artifacts:
            mtime = artifacts["evaluation_txt"].get("mtime")
            if mtime:
                latest_eval_mtime = max(latest_eval_mtime or 0, mtime)
            meta      = exp.get_metadata() or {}
            exp_name  = meta.get("experiment_name", "")
            perf_name = map_experiment_to_name(exp_name) if exp_name else None
            perf_entry = perf_index.get(perf_name) if perf_name else None
            eval_m    = parse_eval_metrics(artifacts["evaluation_txt"]["path"])
            if eval_m and perf_entry:
                perf_m = {k: perf_entry.get(k) for k in ("per", "wer", "accuracy")}
                if eval_m != perf_m:
                    perf_outdated.append(perf_name or exp.base_name)
            elif eval_m and not perf_entry:
                perf_outdated.append(perf_name or exp.base_name)

    from ._constants import REPORT_PATH
    report_mtime    = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else 0
    report_outdated = latest_eval_mtime and latest_eval_mtime > report_mtime

    print("\nResumo de proximo passo:")
    if to_infer:
        print("  1) Inference:       " + ", ".join(sorted(set(to_infer))))
    if to_analyze:
        print("  2) Error analysis:  " + ", ".join(sorted(set(to_analyze))))
    if to_plot:
        print("  3) Plots:           " + ", ".join(sorted(set(to_plot))))
    if perf_outdated:
        print("  4) performance.json desatualizado: " + ", ".join(sorted(set(perf_outdated))))
        print("     → python src/manage_experiments.py --registry")
    if report_outdated:
        print("  5) Report HTML desatualizado")
        print("     → python src/reporting/report_generator.py")
    if not any([to_infer, to_analyze, to_plot, perf_outdated, report_outdated]):
        print("  Nenhuma acao pendente.")
    print(f"\n{'='*100}")


# =============================================================================
# compare_experiment
# =============================================================================

def compare_experiment(manager, exp_name: str):
    """Compara todos os runs do mesmo experimento (útil após re-treino)."""
    runs = [e for e in manager.experiments if e.base_name.split("__")[0] == exp_name]
    if not runs:
        runs = [e for e in manager.experiments if exp_name in e.base_name]
    if not runs:
        print(f"Nenhum run encontrado para '{exp_name}'. Use --missing para ver opções.")
        return
    if len(runs) == 1:
        print(f"Apenas um run de '{exp_name}'. Nada a comparar.")
        idx = next((i for i, e in manager.index_map.items() if e is runs[0]), None)
        if idx is not None:
            manager.show_experiment(idx)
        return

    print(f"\n{'='*80}\nCOMPARACAO: {exp_name} ({len(runs)} runs)\n{'='*80}")

    rows = []
    for exp in sorted(runs, key=lambda e: e.base_name):
        meta      = exp.get_metadata() or {}
        run_id    = meta.get("run_id", "?")
        best_loss = meta.get("best_loss")
        epoch     = meta.get("final_epoch") or meta.get("current_epoch", "?")
        has_pt    = "model_checkpoint" in exp.artifacts
        metrics   = (parse_eval_metrics(exp.artifacts["evaluation_txt"]["path"])
                     if "evaluation_txt" in exp.artifacts else None)
        rows.append({"run_id": run_id, "epoch": epoch, "best_loss": best_loss,
                     "per": metrics.get("per") if metrics else None,
                     "wer": metrics.get("wer") if metrics else None,
                     "has_pt": has_pt, "base_name": exp.base_name})

    print(f"\n{'Run ID':<22} {'Epoch':>6} {'BestLoss':>10} {'PER%':>7} {'WER%':>7}  Checkpoint")
    print("-" * 70)
    for r in rows:
        per_s  = f"{r['per']:.2f}"  if r["per"]       is not None else "    -"
        wer_s  = f"{r['wer']:.2f}"  if r["wer"]       is not None else "    -"
        loss_s = f"{r['best_loss']:.6f}" if r["best_loss"] is not None else "       -"
        pt_s   = "[OK]" if r["has_pt"] else "[sem .pt]"
        print(f"{r['run_id']:<22} {r['epoch']:>6} {loss_s:>10} {per_s:>7} {wer_s:>7}  {pt_s}")

    with_per  = [r for r in rows if r["per"]  is not None]
    with_wer  = [r for r in rows if r["wer"]  is not None]
    with_loss = [r for r in rows if r["best_loss"] is not None]
    print()
    if with_per:
        b = min(with_per,  key=lambda r: r["per"])
        print(f"Melhor PER : run {b['run_id']} ({b['per']:.2f}%)")
    if with_wer:
        b = min(with_wer,  key=lambda r: r["wer"])
        print(f"Melhor WER : run {b['run_id']} ({b['wer']:.2f}%)")
    if with_loss:
        b = min(with_loss, key=lambda r: r["best_loss"])
        print(f"Menor loss : run {b['run_id']} ({b['best_loss']:.6f})")

    print()
    for r in rows:
        idx = next((i for i, e in manager.index_map.items() if e.base_name == r["base_name"]), None)
        if idx is not None:
            print(f"  Remover run {r['run_id']}: python src/manage_experiments.py --clean {idx}")


# =============================================================================
# check_consistency
# =============================================================================

def check_consistency(manager):
    """
    Verifica se todos os dados de publicação estão consistentes.

        Checa quatro camadas:
      1. Disco     — experimentos completos têm evaluation_*.txt
      2. Sync      — evaluation_*.txt bate com performance.json (mesmas métricas)
      3. Registry  — model_registry.json reflete o estado atual
            4. Benchmark — benchmark formal existe e não aponta para artefato quebrado

    Se tudo passar: os dados em performance.json são publicáveis.
    """
    from ._constants import REGISTRY_PATH
    import json as _json

    manager.index_map = manager._build_index_map()
    perf_index        = load_performance_index()
    W = 90
    print(f"\n{'='*W}\nVERIFICACAO DE CONSISTENCIA — PUBLICACAO\n{'='*W}")

    issues     = []
    ok_list    = []
    orphaned   = []   # entries em performance.json sem checkpoint ativo

    # --- Camada 1 & 2: disco vs performance.json ---
    complete_exps = [(idx, exp) for idx, exp in manager.index_map.items()
                     if exp.classify_status() == ExperimentStatus.COMPLETE]

    print(f"\n[1/3] EXPERIMENTOS COM CHECKPOINT ({len(complete_exps)})")
    print(f"  {'idx':>4}  {'experimento':<50}  {'eval':^5}  {'perf.json':^9}  {'delta'}")
    print(f"  {'-'*80}")

    for idx, exp in complete_exps:
        meta     = exp.get_metadata() or {}
        exp_name = meta.get("experiment_name", "")
        arts     = exp.artifacts

        has_eval = "evaluation_txt" in arts
        perf_name = map_experiment_to_name(exp_name) if exp_name else None
        perf_entry = perf_index.get(perf_name) if perf_name else None

        eval_m  = parse_eval_metrics(arts["evaluation_txt"]["path"]) if has_eval else None
        perf_m  = {k: perf_entry.get(k) for k in ("per","wer","accuracy")} if perf_entry else None

        eval_ok  = "[OK]" if has_eval   else "[--]"
        perf_ok  = "[OK]" if perf_entry else "[sem entrada]"

        delta = ""
        if eval_m and perf_m:
            diffs = {k: abs((eval_m.get(k) or 0) - (perf_m.get(k) or 0))
                     for k in ("per","wer") if eval_m.get(k) is not None}
            if any(v > 0.005 for v in diffs.values()):
                delta = f"DIVERGE per={eval_m.get('per')} vs {perf_m.get('per')}"
                issues.append(f"[{idx}] {exp.base_name}: {delta}")
            else:
                delta = "match"
                ok_list.append(exp.base_name)
        elif has_eval and not perf_entry:
            delta = "sem mapeamento em performance.json"
            issues.append(f"[{idx}] {exp.base_name}: nao tem entrada em performance.json")
        elif not has_eval:
            issues.append(f"[{idx}] {exp.base_name}: sem evaluation_*.txt — rode --run {idx}")

        print(f"  {idx:>4}  {exp.base_name:<50}  {eval_ok:^5}  {perf_ok:^9}  {delta}")

    # --- Camada 3: orphaned entries em performance.json ---
    active_perf_names = set()
    for _, exp in complete_exps:
        meta = exp.get_metadata() or {}
        name = map_experiment_to_name(meta.get("experiment_name",""))
        if name:
            active_perf_names.add(name)

    print("\n[2/3] ENTRADAS ORFAS EM performance.json (sem checkpoint ativo)")
    orphaned_entries = [n for n in perf_index if n not in active_perf_names
                        and n.startswith("FG2P")]
    if orphaned_entries:
        for n in orphaned_entries:
            e = perf_index[n]
            src = "inference_completed" in e
            tag = "(sincronizado em execucao anterior)" if src else "(entrada manual)"
            print(f"  {n}  {tag}")
            orphaned.append(n)
    else:
        print("  Nenhuma.")

    # --- Camada 4: registry ---
    print("\n[3/3] MODEL REGISTRY")
    if REGISTRY_PATH.exists():
        reg      = _json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        aliases  = reg.get("aliases", {})
        bp       = aliases.get("best_per", {})
        bw       = aliases.get("best_wer", {})
        fast_a   = aliases.get("fast",     {})
        per_val  = f"{bp.get('per',0)*100:.2f}%" if bp.get("per") else "?"
        wer_val  = f"{bw.get('wer',0)*100:.2f}%" if bw.get("wer") else "?"
        fast_exp = fast_a.get("experiment") or fast_a.get("status","PENDENTE")
        print(f"  best_per → {bp.get('experiment','?')} (PER={per_val})")
        print(f"  best_wer → {bw.get('experiment','?')} (WER={wer_val})")
        print(f"  fast     → {fast_exp}")
    else:
        print("  [!] model_registry.json nao encontrado — rode --registry")
        issues.append("model_registry.json ausente")

    # --- Camada 4: benchmark formal ---
    print("\n[4/4] BENCHMARK FORMAL")
    print(f"  {'idx':>4}  {'experimento':<50}  {'artifact':^10}  {'perf.json':^10}  {'delta'}")
    print(f"  {'-'*90}")
    for idx, exp in complete_exps:
        meta = exp.get_metadata() or {}
        exp_name = meta.get("experiment_name", "")
        perf_name = map_experiment_to_name(exp_name) if exp_name else None
        perf_entry = perf_index.get(perf_name) if perf_name else None
        bench_info = exp.artifacts.get("benchmark_json")
        has_artifact = bench_info is not None
        summary = perf_entry.get("benchmark_summary") if perf_entry else None
        has_perf_summary = bool(summary)
        delta = ""
        if not has_artifact:
            delta = "pendente"
            issues.append(f"[{idx}] {exp.base_name}: sem benchmark formal — rode --benchmark {idx}")
        elif has_artifact and not has_perf_summary:
            delta = "artifact existe, sem sync em performance.json"
            issues.append(f"[{idx}] {exp.base_name}: benchmark sem resumo em performance.json")
        elif has_artifact and has_perf_summary:
            artifact_path = bench_info["path"].as_posix()
            devices = (summary.get("devices") or {}) if isinstance(summary, dict) else {}
            known_paths = {str(v.get("artifact_path")) for v in devices.values() if isinstance(v, dict)}
            delta = "match" if artifact_path in known_paths else "artifact fora do summary"
            if delta != "match":
                issues.append(f"[{idx}] {exp.base_name}: benchmark artifact não referenciado em performance.json")

        artifact_ok = "[OK]" if has_artifact else "[--]"
        perf_ok = "[OK]" if has_perf_summary else "[--]"
        print(f"  {idx:>4}  {exp.base_name:<50}  {artifact_ok:^10}  {perf_ok:^10}  {delta}")

    # --- Resumo ---
    print(f"\n{'='*W}")
    n_ok  = len(ok_list)
    n_exp = len(complete_exps)
    n_orf = len(orphaned)
    if not issues:
        print(f"[PUBLICAVEL] {n_ok}/{n_exp} experimentos sincronizados | "
              f"{n_orf} entradas historicas em performance.json (validas)")
        print("  performance.json esta em sincronia com os dados no disco.")
        print("  Os dados podem ser usados diretamente para publicacao.")
    else:
        print(f"[ATENCAO] {len(issues)} problema(s) encontrado(s) — corrigir antes de publicar:")
        for iss in issues:
            print(f"  ! {iss}")
    print(f"{'='*W}\n")
    return len(issues) == 0


# =============================================================================
# show_status
# =============================================================================

def show_status(manager):
    """Detecta experimentos em execução (checkpoint < 30 min)."""
    import json as _json
    running = [(idx, exp, (datetime.now().timestamp() - exp.artifacts["model_checkpoint"]["mtime"]) / 60)
               for idx, exp in manager.index_map.items()
               if "model_checkpoint" in exp.artifacts
               and (datetime.now().timestamp() - exp.artifacts["model_checkpoint"]["mtime"]) / 60 <= 30]

    if not running:
        print("\nNenhum experimento em execucao (checkpoint mais recente > 30 min).")
        return

    print(f"\n{'='*70}\nEXPERIMENTOS EM EXECUCAO ({len(running)})\n{'='*70}")
    for idx, exp, age in running:
        meta     = exp.get_metadata() or {}
        exp_name = meta.get("experiment_name", exp.base_name)
        epoch    = meta.get("current_epoch", "?")
        total_ep = meta.get("total_epochs", "?")
        print(f"\n[{idx}] {exp_name}")
        print(f"    Checkpoint: {age:.1f} min atras | Epoch: {epoch}/{total_ep}")

        r_dir     = RESULTS_DIR / exp_name
        prog_files = sorted(r_dir.glob("training_progress*.json"), key=lambda p: p.stat().st_mtime) if r_dir.exists() else []
        if prog_files:
            try:
                prog = _json.loads(prog_files[-1].read_text(encoding="utf-8"))
                print(f"    Progress: epoch={prog.get('epoch','?')} "
                      f"train_loss={prog.get('train_loss','?')} "
                      f"val_loss={prog.get('val_loss','?')} "
                      f"time={prog.get('epoch_time_s','?')}s")
            except Exception:
                pass
