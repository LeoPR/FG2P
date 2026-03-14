"""
Modelo de dados de um experimento: artefatos no disco e classificação de status.

ExperimentStatus — enum de strings para os 4 estados possíveis.
ExperimentArtifacts — escaneia models/ e results/ para um dado experimento,
    classifica status, calcula tamanho, apaga artefatos.
"""
import json
from datetime import datetime
from typing import Optional

from ._constants import MODELS_DIR, RESULTS_DIR


class ExperimentStatus:
    COMPLETE   = "completo"
    INCOMPLETE = "incompleto"
    ORPHAN     = "orfao"
    RUNNING    = "rodando"


class ExperimentArtifacts:
    """Representa todos os artefatos de um experimento no disco."""

    def __init__(self, base_name: str):
        """
        Args:
            base_name: Nome com timestamp, ex: exp0_baseline_70split__20260218_044620
        """
        self.base_name = base_name
        self.artifacts = self._scan()

    def _scan(self) -> dict:
        # Delega a construção de paths ao FileRegistry — fonte única de verdade.
        from file_registry import ExperimentRecord
        rec = ExperimentRecord(MODELS_DIR / self.base_name.split("__")[0] / f"{self.base_name}.pt")
        reg = rec._reg

        candidates = {
            "model_checkpoint":   rec.pt_path,
            "model_metadata":     rec.metadata_path,
            "history_csv":        rec.history_path,
            "summary_txt":        rec.summary_path,
            "evaluation_txt":     rec.evaluation_path,
            "predictions_tsv":    rec.predictions_path,
            "error_analysis_txt": rec.error_analysis_path,
            "convergence_plot":   rec.convergence_plot_path,
            "analysis_plot":      reg.get_analysis_metrics_plot_path(),
            "checkpoint_plot":    reg.get_checkpoint_plot_path(),
        }
        found = {}
        for key, path in candidates.items():
            if path.exists():
                st = path.stat()
                found[key] = {"path": path, "size": st.st_size, "mtime": st.st_mtime}

        latest = rec.latest_benchmark_path()
        if latest is not None:
            st = latest.stat()
            found["benchmark_json"] = {"path": latest, "size": st.st_size, "mtime": st.st_mtime}
        return found

    def get_metadata(self) -> Optional[dict]:
        if "model_metadata" not in self.artifacts:
            return None
        try:
            with open(self.artifacts["model_metadata"]["path"], encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def classify_status(self) -> str:
        # Delega ao ExperimentRecord — mesma lógica usada por todos os scripts.
        from file_registry import ExperimentRecord
        rec = ExperimentRecord(MODELS_DIR / self.base_name.split("__")[0] / f"{self.base_name}.pt")
        if not rec.has_checkpoint or rec.metadata is None:
            return ExperimentStatus.ORPHAN
        if not rec.training_completed:
            if rec.has_checkpoint:
                age_min = (datetime.now().timestamp() -
                           self.artifacts["model_checkpoint"]["mtime"]) / 60
                if age_min < 15:
                    return ExperimentStatus.RUNNING
            return ExperimentStatus.INCOMPLETE
        return ExperimentStatus.COMPLETE if rec.is_complete else ExperimentStatus.INCOMPLETE

    def get_total_size(self) -> int:
        return sum(a["size"] for a in self.artifacts.values())

    def count_artifacts(self) -> int:
        return len(self.artifacts)

    def delete_all(self, dry_run: bool = False) -> list:
        deleted = []
        for info in self.artifacts.values():
            path = info["path"]
            if dry_run:
                deleted.append(path)
            else:
                try:
                    path.unlink()
                    deleted.append(path)
                except OSError as e:
                    from ._constants import logger
                    logger.warning(f"Erro ao deletar {path.name}: {e}")
        if not dry_run:
            exp_name = self.base_name.split("__")[0]
            for base_dir in [MODELS_DIR, RESULTS_DIR]:
                exp_dir = base_dir / exp_name
                if exp_dir.exists() and not any(exp_dir.iterdir()):
                    exp_dir.rmdir()
        return deleted
