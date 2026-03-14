"""
Orquestração do pipeline de pós-processamento.

Sequência após um treino concluído:
  inference  →  error_analysis  →  plots
  →  sync_performance  →  rebuild_registry  →  report HTML
"""
import subprocess
import sys
from pathlib import Path

from ._artifacts import ExperimentStatus
from ._constants import REPORT_PATH


def process_all_pending(manager, dry_run=False, force=False,
                        force_inference=False, verify_subprocess_dry_run=True,
                        index=None):
    """
    Executa automaticamente todos os passos pendentes para cada experimento
    completo (com checkpoint).

    Args:
        manager:                  ExperimentManager
        dry_run:                  mostra comandos sem executar
        force:                    re-executa etapas leves mesmo se artefatos existem
        force_inference:          força re-execução de inference
        verify_subprocess_dry_run: valida inference.py --dry-run antes de executar
        index:                    processa apenas o experimento com esse índice
    """
    if not manager.experiments:
        print("\nNenhum experimento encontrado.")
        return

    manager.index_map = manager._build_index_map()

    if index is not None:
        if index not in manager.index_map:
            print(f"\nIndice {index} nao encontrado. Use --missing para ver indices.")
            return
        manager.index_map = {index: manager.index_map[index]}

    print("\n" + "=" * 100)
    label = f"EXPERIMENTO [{index}]" if index is not None else "TODOS OS EXPERIMENTOS"
    print(f"PIPELINE AUTOMATICO — {label}")
    print("=" * 100)

    tasks = {"inference": [], "error_analysis": [], "plots": []}
    report_mtimes = []

    for idx, exp in manager.index_map.items():
        if exp.classify_status() != ExperimentStatus.COMPLETE:
            continue
        arts = exp.artifacts

        if force_inference or "predictions_tsv" not in arts:
            if "model_checkpoint" in arts:
                tasks["inference"].append((exp.base_name, arts["model_checkpoint"]["path"]))

        if "predictions_tsv" in arts and (
            force or "error_analysis_txt" not in arts or "evaluation_txt" not in arts
        ):
            tasks["error_analysis"].append((exp.base_name, arts["predictions_tsv"]["path"]))

        if "history_csv" in arts and (force or "convergence_plot" not in arts):
            tasks["plots"].append(exp.base_name)

        for key in ("evaluation_txt", "error_analysis_txt", "convergence_plot"):
            if key in arts and arts[key].get("mtime"):
                report_mtimes.append(arts[key]["mtime"])

    report_mtime    = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else 0
    needs_report    = force or (report_mtimes and max(report_mtimes) > report_mtime)
    total_tasks     = sum(len(v) for v in tasks.values()) + (1 if needs_report else 0)

    if total_tasks == 0:
        print("\nTodos os experimentos estao completos. Nenhuma tarefa pendente.\n")
        return

    print(f"\nTarefas pendentes: {total_tasks}")
    print(f"  Inference:      {len(tasks['inference'])}")
    print(f"  Error Analysis: {len(tasks['error_analysis'])}")
    print(f"  Plots:          {len(tasks['plots'])}")
    print(f"  Report HTML:    {'1' if needs_report else '0'}")

    if dry_run:
        _dry_run_summary(tasks, needs_report, force_inference, verify_subprocess_dry_run)
        return

    confirm = input("\nContinuar? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelado.")
        return

    ok = err = 0

    # 1 — Inference
    if tasks["inference"]:
        print(f"\n{'='*100}\nINFERENCE ({len(tasks['inference'])} modelo(s))\n{'='*100}")
        for base_name, _ in tasks["inference"]:
            cmd = [sys.executable, "src/inference.py", "--model", base_name]
            if force_inference:
                cmd.append("--force")
            ok, err = _run(cmd, base_name, ok, err, timeout=600)

    # 2 — Error Analysis
    if tasks["error_analysis"]:
        print(f"\n{'='*100}\nERROR ANALYSIS ({len(tasks['error_analysis'])} exp)\n{'='*100}")
        for base_name, _ in tasks["error_analysis"]:
            cmd = [sys.executable, "src/analyze_errors.py", "--model", base_name]
            ok, err = _run(cmd, base_name, ok, err, timeout=300)

    # 3 — Plots
    if tasks["plots"]:
        print(f"\n{'='*100}\nPLOTS ({len(tasks['plots'])} exp)\n{'='*100}")
        for base_name in tasks["plots"]:
            cmd = [sys.executable, "src/analysis.py", "--model-name", base_name]
            ok, err = _run(cmd, base_name, ok, err, timeout=120)

    # 4 — Sync performance.json (direto, sem subprocess)
    if tasks["error_analysis"] or needs_report:
        print(f"\n{'='*100}\nSINCRONIZANDO performance.json\n{'='*100}")
        try:
            from ._sync import sync_performance
            sync_performance()
            print("  OK")
            ok += 1
        except Exception as e:
            print(f"  Erro: {e}")
            err += 1

    # 5 — Rebuild registry (direto)
    if tasks["inference"] or tasks["error_analysis"]:
        print(f"\n{'='*100}\nATUALIZANDO model_registry.json\n{'='*100}")
        try:
            manager.rebuild_registry()
            ok += 1
        except Exception as e:
            print(f"  Erro: {e}")
            err += 1

    # 6 — Report HTML
    if needs_report:
        print(f"\n{'='*100}\nREPORT HTML\n{'='*100}")
        cmd = [sys.executable, "src/reporting/report_generator.py"]
        ok, err = _run(cmd, "report_generator", ok, err, timeout=300)

    print(f"\n{'='*100}\nRESUMO: {ok+err} tarefas | {ok} OK | {err} erros\n{'='*100}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, label, ok, err, timeout=300):
    print(f"\n→ {label}")
    try:
        result = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True,
                                text=True, encoding='utf-8', errors='replace',
                                timeout=timeout)
        if result.returncode == 0:
            print("  OK")
            return ok + 1, err
        else:
            print(f"  Erro (code={result.returncode}): {(result.stderr or '')[:200]}")
            return ok, err + 1
    except subprocess.TimeoutExpired:
        print(f"  Timeout (>{timeout}s)")
        return ok, err + 1
    except Exception as e:
        print(f"  Erro: {e}")
        return ok, err + 1


def train_experiment(manager, config_name: str, dry_run: bool = False):
    """
    Treina um experimento a partir de conf/<config_name>.json e,
    ao terminar com sucesso, roda automaticamente o pipeline de pós-processamento.

    Args:
        config_name: nome do arquivo de config sem extensão, ex:
                     "config_exp0_baseline_70split" ou apenas "exp0_baseline_70split"
    """
    from ._constants import MODELS_DIR
    conf_dir = MODELS_DIR.parent / "conf"

    # Aceita com ou sem extensão / prefixo
    if not config_name.endswith(".json"):
        config_name += ".json"
    if not config_name.startswith("config_"):
        # tenta primeiro sem prefixo, depois com
        candidates = [conf_dir / config_name,
                      conf_dir / f"config_{config_name}"]
    else:
        candidates = [conf_dir / config_name]

    config_path = next((p for p in candidates if p.exists()), None)
    if config_path is None:
        tried = [str(c) for c in candidates]
        print(f"[ERRO] Config nao encontrado. Tentativas: {tried}")
        return

    print(f"\n{'='*100}\nTREINANDO: {config_path.name}\n{'='*100}")
    if dry_run:
        print(f"[DRY RUN] python src/train.py --config {config_path}")
        return

    ok, err = _run([sys.executable, "src/train.py", "--config", str(config_path)],
                   config_path.stem, 0, 0, timeout=7200)
    if err:
        print("[ERRO] Treino falhou. Pipeline nao sera executado.")
        return

    # Treino OK — recarregar experimentos e rodar pipeline para o novo run
    print("\nTreino concluido. Executando pipeline de pos-processamento...")
    manager.experiments = manager._discover()
    process_all_pending(manager, dry_run=False, force=False)


def train_all(manager, dry_run: bool = False):
    """
    Lista os configs em conf/ sem checkpoint correspondente e treina cada um.
    """
    from pathlib import Path
    from file_registry import list_experiments

    conf_dir = Path("conf")
    if not conf_dir.exists():
        print("[ERRO] Pasta conf/ nao encontrada.")
        return

    existing = {r.exp_name for r in list_experiments()}
    configs  = sorted(conf_dir.glob("config_*.json"))

    pending = []
    for cfg in configs:
        # extrai exp_name do nome do arquivo: config_<exp_name>.json
        exp_name = cfg.stem[len("config_"):]
        if exp_name not in existing:
            pending.append(cfg)

    if not pending:
        print("\nTodos os experimentos ja possuem checkpoint. Nada a treinar.")
        return

    print(f"\nExperimentos sem checkpoint: {len(pending)}")
    for cfg in pending:
        print(f"  {cfg.name}")

    if dry_run:
        print("[DRY RUN] Nenhuma acao executada.")
        return

    confirm = input(f"\nTreinar {len(pending)} experimento(s)? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelado.")
        return

    for cfg in pending:
        train_experiment(manager, cfg.stem, dry_run=False)


def run_benchmark(manager, index=None, dry_run: bool = False,
                  device: str = "auto", force: bool = False,
                  update_performance: bool = True):
    """Executa benchmark formal de inferência como etapa opcional do manager."""
    manager.index_map = manager._build_index_map(filter_status=ExperimentStatus.COMPLETE)

    if not manager.index_map:
        print("\nNenhum experimento completo disponível para benchmark.")
        return

    cmd = [sys.executable, "src/benchmark_inference.py", "--device", device]
    label = "todos os experimentos completos"

    if index is not None:
        if index not in manager.index_map:
            print(f"\nIndice {index} nao encontrado. Use --list para ver indices.")
            return
        cmd.extend(["--index", str(index)])
        label = f"experimento [{index}]"

    if force:
        cmd.append("--force")

    if update_performance:
        cmd.append("--update-performance")

    print("\n" + "=" * 100)
    print(f"BENCHMARK FORMAL — {label.upper()}")
    print("=" * 100)

    if dry_run:
        print("[DRY RUN] Comando que seria executado:")
        print("  " + " ".join(cmd))
        return

    confirm = input("\nContinuar? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelado.")
        return

    ok, err = _run(cmd, "benchmark_inference", 0, 0, timeout=3600)
    print(f"\nRESUMO BENCHMARK: {ok} OK | {err} erro(s)\n")


def _dry_run_summary(tasks, needs_report, force_inference, verify):
    print("\n[DRY RUN] Comandos que seriam executados:\n")
    if tasks["inference"]:
        print("1. INFERENCE:")
        for base_name, _ in tasks["inference"]:
            flag = " --force" if force_inference else ""
            print(f"   python src/inference.py --model {base_name}{flag}")
    if tasks["error_analysis"]:
        print("\n2. ERROR ANALYSIS:")
        for base_name, _ in tasks["error_analysis"]:
            print(f"   python src/analyze_errors.py --model {base_name}")
    if tasks["plots"]:
        print("\n3. PLOTS:")
        for base_name in tasks["plots"]:
            print(f"   python src/analysis.py --model-name {base_name}")
    print("\n4. sync_performance()       # interno, sem subprocess")
    print("5. rebuild_registry()       # interno, sem subprocess")
    if needs_report:
        print("6. python src/reporting/report_generator.py")
    print("\nUse sem --dry-run para executar.")
