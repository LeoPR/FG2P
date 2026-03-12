"""
ExperimentManager — descoberta de experimentos e operações básicas de CRUD.

Operações pesadas (pipeline, sync, reports) são delegadas aos submódulos
correspondentes e expostas como métodos finos aqui.
"""
from typing import Optional

from ._artifacts import ExperimentArtifacts, ExperimentStatus


class ExperimentManager:
    """Ponto central de acesso aos experimentos."""

    def __init__(self):
        self.experiments: list[ExperimentArtifacts] = self._discover()
        self.index_map:   dict[int, ExperimentArtifacts] = {}

    # ------------------------------------------------------------------
    # Descoberta
    # ------------------------------------------------------------------

    def _discover(self) -> list[ExperimentArtifacts]:
        from file_registry import list_experiments as _list_exp
        return [ExperimentArtifacts(r.base_name) for r in _list_exp()]

    def _build_index_map(self, filter_status: Optional[str] = None) -> dict:
        order = [ExperimentStatus.COMPLETE, ExperimentStatus.RUNNING,
                 ExperimentStatus.INCOMPLETE, ExperimentStatus.ORPHAN]
        by_status: dict[str, list] = {s: [] for s in order}
        for exp in self.experiments:
            by_status[exp.classify_status()].append(exp)
        index_map, idx = {}, 0
        for status in order:
            if filter_status and status != filter_status:
                continue
            for exp in by_status[status]:
                index_map[idx] = exp
                idx += 1
        return index_map

    def _require_index_map(self):
        if not self.index_map:
            self.index_map = self._build_index_map()

    # ------------------------------------------------------------------
    # Listagem e inspecção
    # ------------------------------------------------------------------

    def list_experiments(self, filter_status: Optional[str] = None):
        if not self.experiments:
            print("\nNenhum experimento encontrado em models/")
            return
        self.index_map = self._build_index_map(filter_status)
        order  = [ExperimentStatus.COMPLETE, ExperimentStatus.RUNNING,
                  ExperimentStatus.INCOMPLETE, ExperimentStatus.ORPHAN]
        icons  = {ExperimentStatus.COMPLETE: "[OK]", ExperimentStatus.INCOMPLETE: "[!]",
                  ExperimentStatus.RUNNING: "[~]",   ExperimentStatus.ORPHAN: "[X]"}
        labels = {ExperimentStatus.COMPLETE: "COMPLETOS", ExperimentStatus.INCOMPLETE: "INCOMPLETOS",
                  ExperimentStatus.RUNNING: "RODANDO",     ExperimentStatus.ORPHAN: "ORFAOS"}
        by_status: dict[str, list] = {s: [] for s in order}
        for exp in self.experiments:
            by_status[exp.classify_status()].append(exp)

        print("\n" + "=" * 100)
        print("GERENCIADOR DE EXPERIMENTOS FG2P")
        print("=" * 100)
        idx = 0
        for status in order:
            if filter_status and status != filter_status:
                continue
            exps = by_status[status]
            if not exps:
                continue
            print(f"\n{icons[status]} {labels[status]} ({len(exps)})")
            print("-" * 100)
            for exp in exps:
                meta    = exp.get_metadata()
                size_mb = exp.get_total_size() / (1024 * 1024)
                arts    = exp.artifacts
                n_core = sum(1 for k in arts
                             if self._ARTIFACT_INFO.get(k, ("","",""))[2] == "core")
                n_pipe = sum(1 for k in arts
                             if self._ARTIFACT_INFO.get(k, ("","",""))[2] == "pipeline")
                print(f"\n[{idx}] {exp.base_name}")
                print(f"    Status: {icons[status]} {status.upper()}")
                print(f"    Core: {n_core}/4 | Pipeline: {n_pipe}/4 | {size_mb:.2f} MB")
                if meta:
                    ep = meta.get("current_epoch", meta.get("final_epoch", "?"))
                    print(f"    Epoch: {ep}/{meta.get('total_epochs', '?')} | "
                          f"Loss: {meta.get('best_loss', 0):.4f} | "
                          f"Params: {meta.get('total_params', 0):,}")
                idx += 1
        print("\n" + "=" * 100)
        print(f"Total: {len(self.experiments)} | "
              f"Completos: {len(by_status[ExperimentStatus.COMPLETE])} | "
              f"Rodando: {len(by_status[ExperimentStatus.RUNNING])} | "
              f"Incompletos: {len(by_status[ExperimentStatus.INCOMPLETE])} | "
              f"Orfaos: {len(by_status[ExperimentStatus.ORPHAN])}")
        print("=" * 100)
        print("\nUso: --show N | --missing | --run [N] | --clean N | --compare EXP\n")

    # Descrição e origem de cada artefato (para --show)
    _ARTIFACT_INFO = {
        "model_checkpoint":   ("Modelo treinado (.pt)",               "train.py",          "core"),
        "model_metadata":     ("Metadados do run (JSON)",             "train.py",          "core"),
        "history_csv":        ("Histórico de loss por epoch",         "train.py",          "core"),
        "summary_txt":        ("Resumo de treino (texto)",            "train.py",          "core"),
        "evaluation_txt":     ("Métricas PER/WER no test set",        "--run [N]",         "pipeline"),
        "predictions_tsv":    ("Predições word-level (TSV)",          "--run [N]",         "pipeline"),
        "error_analysis_txt": ("Análise de erros detalhada",          "--run [N]",         "pipeline"),
        "convergence_plot":   ("Gráfico loss treino/validação (PNG)", "--run [N]",         "pipeline"),
        "checkpoint_plot":    ("Plot de checkpoint durante treino",   "train.py (antigo)", "optional"),
        "analysis_plot":      ("Gráfico gap+throughput de análise",   "analysis.py --plot","optional"),
    }

    def show_experiment(self, index: int):
        self._require_index_map()
        if index not in self.index_map:
            print(f"Indice {index} invalido. Use --list para ver experimentos.")
            return
        exp    = self.index_map[index]
        meta   = exp.get_metadata()
        status = exp.classify_status()
        arts   = exp.artifacts
        n_core     = sum(1 for k in arts if self._ARTIFACT_INFO.get(k, ("","",""))[2] == "core")
        n_pipeline = sum(1 for k in arts if self._ARTIFACT_INFO.get(k, ("","",""))[2] == "pipeline")
        n_opt      = sum(1 for k in arts if self._ARTIFACT_INFO.get(k, ("","",""))[2] == "optional")

        print(f"\n{'='*100}\nDETALHES [{index}]: {exp.base_name}\n{'='*100}")
        print(f"Status: {status.upper()} | Tamanho: {exp.get_total_size()/(1024*1024):.2f} MB"
              f" | Core: {n_core}/4 | Pipeline: {n_pipeline}/4 | Opcional: {n_opt}/2")
        if meta:
            print("\n--- METADADOS ---")
            print(f"Experimento: {meta.get('experiment_name', 'N/A')}")
            print(f"Timestamp:   {meta.get('timestamp', 'N/A')}")
            print(f"Training OK: {meta.get('training_completed', False)}")
            print(f"Epoch:       {meta.get('current_epoch','?')}/{meta.get('total_epochs','?')}")
            print(f"Best Loss:   {meta.get('best_loss', 'N/A')}")
            print(f"Params:      {meta.get('total_params', 'N/A'):,}")
            cfg = meta.get("config", {})
            if "model" in cfg:
                m = cfg["model"]
                print("\n--- ARQUITETURA ---")
                print(f"emb={m.get('emb_dim','?')} hidden={m.get('hidden_dim','?')} "
                      f"layers={m.get('num_layers','?')} type={m.get('embedding_type','?')}")
            if "training" in cfg:
                t = cfg["training"]
                print("\n--- TREINO ---")
                print(f"lr={t.get('lr','?')} batch={t.get('batch_size','?')} "
                      f"patience={t.get('early_stopping_patience','?')}")

        print("\n--- ARTEFATOS (11 possíveis) ---")
        print(f"  {'artefato':<25} {'status':^8} {'tamanho':>10}  {'descrição'}")
        print(f"  {'-'*80}")
        for key, (desc, origin, kind) in self._ARTIFACT_INFO.items():
            if key in arts:
                size_kb = arts[key]["size"] / 1024
                tag = "[OK] " if kind != "optional" else "[OK*]"
                print(f"  {key:<25} {tag:^8} {size_kb:>8.1f} KB  {desc}")
            else:
                tag = "[--] " if kind != "optional" else "[--*]"
                how = f"  ← {origin}" if kind != "optional" else f"  ← opcional: {origin}"
                print(f"  {key:<25} {tag:^8} {'':>10}  {desc}{how}")
        print("\n  * = artefato opcional (não bloqueante)")
        print()

    # ------------------------------------------------------------------
    # Remoção
    # ------------------------------------------------------------------

    def prune_experiment(self, index: int, dry_run: bool = False):
        self._require_index_map()
        if index not in self.index_map:
            print(f"Indice {index} invalido. Use --list para ver indices.")
            return
        exp    = self.index_map[index]
        meta   = exp.get_metadata()
        status = exp.classify_status()
        size   = exp.get_total_size() / (1024 * 1024)

        print(f"\n{'='*80}")
        print(f"APAGAR [{index}]: {exp.base_name}")
        print(f"  Status: {status.upper()} | {exp.count_artifacts()} artefatos | {size:.2f} MB")
        if meta:
            epoch  = meta.get("final_epoch") or meta.get("current_epoch", "?")
            print(f"  Epoch: {epoch}/{meta.get('total_epochs','?')} | Loss: {meta.get('best_loss',0):.4f}")
        print(f"{'='*80}")

        if dry_run:
            deleted = exp.delete_all(dry_run=True)
            print(f"\n[DRY RUN] Seriam removidos {len(deleted)} arquivo(s):")
            for p in deleted:
                print(f"  {p}")
            return

        confirm = input(f"\nApagar {exp.count_artifacts()} arquivo(s) de [{index}]? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("Cancelado.")
            return

        deleted = exp.delete_all(dry_run=False)
        print(f"\nRemovidos: {len(deleted)} arquivo(s)")
        for p in deleted:
            print(f"  {p}")

    def prune_incomplete(self, dry_run: bool = False):
        self._require_index_map()
        targets = [(idx, exp) for idx, exp in self.index_map.items()
                   if exp.classify_status() in (ExperimentStatus.INCOMPLETE, ExperimentStatus.ORPHAN)]
        if not targets:
            print("Nenhum experimento incompleto/orfao encontrado.")
            return

        print(f"\n{'='*80}")
        print(f"APAGAR {len(targets)} EXPERIMENTO(S) INCOMPLETO(S)/ORFAO(S)")
        print(f"{'='*80}")
        for idx, exp in targets:
            size = exp.get_total_size() / (1024 * 1024)
            print(f"  [{idx}] {exp.base_name} ({size:.2f} MB)")

        if dry_run:
            print(f"\n[DRY RUN] Seriam removidos {len(targets)} experimento(s).")
            return

        confirm = input(f"\nApagar {len(targets)} experimento(s)? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("Cancelado.")
            return

        for _, exp in targets:
            deleted = exp.delete_all(dry_run=False)
            print(f"  {exp.base_name}: {len(deleted)} arquivo(s) removido(s)")

    # ------------------------------------------------------------------
    # Estatísticas
    # ------------------------------------------------------------------

    def show_stats(self):
        order = [ExperimentStatus.COMPLETE, ExperimentStatus.RUNNING,
                 ExperimentStatus.INCOMPLETE, ExperimentStatus.ORPHAN]
        by_status: dict[str, list] = {s: [] for s in order}
        for exp in self.experiments:
            by_status[exp.classify_status()].append(exp)
        total_size = sum(exp.get_total_size() for exp in self.experiments)
        print(f"\nTotal: {len(self.experiments)} experimentos | "
              f"{total_size/(1024*1024):.1f} MB")
        for s in order:
            print(f"  {s}: {len(by_status[s])}")

    # ------------------------------------------------------------------
    # Delegação para submódulos especializados
    # ------------------------------------------------------------------

    def process_all_pending(self, dry_run=False, force=False, force_inference=False,
                            verify_subprocess_dry_run=True, index=None):
        from ._pipeline import process_all_pending
        process_all_pending(self, dry_run=dry_run, force=force,
                            force_inference=force_inference,
                            verify_subprocess_dry_run=verify_subprocess_dry_run,
                            index=index)

    def train_experiment(self, config_name: str, dry_run: bool = False):
        from ._pipeline import train_experiment
        train_experiment(self, config_name, dry_run=dry_run)

    def train_all(self, dry_run: bool = False):
        from ._pipeline import train_all
        train_all(self, dry_run=dry_run)

    def rebuild_registry(self):
        from ._sync import rebuild_registry
        rebuild_registry(self)

    def show_missing(self):
        from ._reports import show_missing
        show_missing(self)

    def guide(self):
        from ._reports import guide
        guide(self)

    def check_consistency(self):
        from ._reports import check_consistency
        return check_consistency(self)

    def compare_experiment(self, exp_name: str):
        from ._reports import compare_experiment
        compare_experiment(self, exp_name)

    def show_status(self):
        from ._reports import show_status
        show_status(self)
