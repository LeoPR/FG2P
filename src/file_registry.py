#!/usr/bin/env python
"""
Registro centralizado de nomes de arquivos do projeto FG2P

Este módulo é a fonte única de verdade para geração de nomes de arquivos.
Todos os scripts (train.py, inference.py, report_generator.py) devem usar
estas funções para garantir consistência e rastreabilidade.

Padrão de nomenclatura:
- Prefixo base: {experiment_name} do config (ex: "exp2_traditional_extended")
- Identificador único: __{timestamp} (ex: "__20260217_003045")
- Nome completo: {experiment_name}__{timestamp}

Arquivos gerados (subpasta por experimento):
- Modelo: models/{experiment_name}/{base}.pt
- Metadata: models/{experiment_name}/{base}_metadata.json
- History CSV: results/{experiment_name}/{base}_history.csv
- Evaluation: results/{experiment_name}/evaluation_{base}.txt
- Predictions: results/{experiment_name}/predictions_{base}.tsv
- Error analysis: results/{experiment_name}/error_analysis_{base}.txt
- Checkpoint plot: results/{experiment_name}/{base}_checkpoint.png
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Importa paths do utils
from utils import MODELS_DIR, RESULTS_DIR


class FileRegistry:
    """Gerenciador de nomes de arquivos do projeto"""
    
    def __init__(self, config: dict, timestamp: Optional[str] = None):
        """
        Args:
            config: Dicionário de configuração carregado do JSON
            timestamp: Timestamp customizado (opcional). Se None, usa timestamp atual.
        """
        self.config = config
        self.experiment_name = config["experiment"]["name"]
        
        # Timestamp no formato YYYYMMDD_HHMMSS
        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = timestamp
        
        # Nome base completo: {experiment_name}__{timestamp}
        self.base_name = f"{self.experiment_name}__{self.timestamp}"
    
    # ========================================================================
    # Helpers de diretório por experimento
    # ========================================================================

    def _models_exp_dir(self) -> Path:
        """Retorna (e cria) subpasta do experimento em models/"""
        d = MODELS_DIR / self.experiment_name
        d.mkdir(exist_ok=True)
        return d

    def _results_exp_dir(self) -> Path:
        """Retorna (e cria) subpasta do experimento em results/"""
        d = RESULTS_DIR / self.experiment_name
        d.mkdir(exist_ok=True)
        return d

    # ========================================================================
    # Arquivos de modelo (models/{experiment_name}/)
    # ========================================================================

    def get_model_path(self) -> Path:
        """Retorna path do arquivo .pt do modelo"""
        return self._models_exp_dir() / f"{self.base_name}.pt"

    def get_metadata_path(self) -> Path:
        """Retorna path do arquivo _metadata.json"""
        return self._models_exp_dir() / f"{self.base_name}_metadata.json"

    # ========================================================================
    # Arquivos de treinamento (results/{experiment_name}/)
    # ========================================================================

    def get_history_path(self) -> Path:
        """Retorna path do CSV de histórico de treino"""
        return self._results_exp_dir() / f"{self.base_name}_history.csv"

    def get_checkpoint_plot_path(self) -> Path:
        """Retorna path do plot de convergência"""
        return self._results_exp_dir() / f"{self.base_name}_checkpoint.png"

    # ========================================================================
    # Arquivos de avaliação (results/{experiment_name}/)
    # ========================================================================

    def get_evaluation_path(self) -> Path:
        """Retorna path do arquivo de métricas de avaliação"""
        return self._results_exp_dir() / f"evaluation_{self.base_name}.txt"

    def get_predictions_path(self) -> Path:
        """Retorna path do TSV de predições word-level"""
        return self._results_exp_dir() / f"predictions_{self.base_name}.tsv"

    def get_error_analysis_path(self) -> Path:
        """Retorna path do arquivo de análise de erros"""
        return self._results_exp_dir() / f"error_analysis_{self.base_name}.txt"

    # ========================================================================
    # Arquivos de análise de treinamento (results/{experiment_name}/)
    # ========================================================================

    def get_analysis_convergence_plot_path(self) -> Path:
        """Retorna path do gráfico de convergência (loss curves)"""
        return self._results_exp_dir() / f"{self.base_name}_convergence.png"

    def get_analysis_metrics_plot_path(self) -> Path:
        """Retorna path do gráfico de análise (gap + throughput)"""
        return self._results_exp_dir() / f"{self.base_name}_analysis.png"

    def get_analysis_results_path(self) -> Path:
        """Retorna path do JSON com métricas de análise calculadas"""
        return self._results_exp_dir() / f"{self.base_name}_results.json"

    # ========================================================================
    # Arquivos de benchmark de inferência (results/{experiment_name}/ e results/benchmarks/)
    # ========================================================================

    def get_benchmark_path(self, device_tag: str) -> Path:
        """Retorna path do JSON formal de benchmark de inferência para um device."""
        safe_device = device_tag.replace(":", "_").replace("/", "_")
        return self._results_exp_dir() / f"benchmark_{self.base_name}_{safe_device}.json"

    def get_benchmark_raw_csv_path(self, device_tag: str) -> Path:
        """Retorna path do CSV com medições brutas de benchmark (sidecar do JSON)."""
        safe_device = device_tag.replace(":", "_").replace("/", "_")
        return self._results_exp_dir() / f"benchmark_{self.base_name}_{safe_device}_raw.csv"

    @classmethod
    def get_benchmarks_dir(cls) -> Path:
        """Retorna (e cria) diretório global para índices agregados de benchmark."""
        d = RESULTS_DIR / "benchmarks"
        d.mkdir(exist_ok=True)
        return d

    @classmethod
    def get_benchmark_run_path(cls, run_label: str = "all", timestamp: Optional[str] = None) -> Path:
        """Retorna path do JSON agregado de uma execução de benchmark."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = run_label.replace(" ", "_").replace(":", "_").replace("/", "_")
        return cls.get_benchmarks_dir() / f"benchmark_run_{safe_label}__{timestamp}.json"
    
    # ========================================================================
    # Métodos auxiliares
    # ========================================================================
    
    def get_all_paths(self) -> dict[str, Path]:
        """Retorna dicionário com todos os paths possíveis"""
        return {
            "model": self.get_model_path(),
            "metadata": self.get_metadata_path(),
            "history": self.get_history_path(),
            "checkpoint_plot": self.get_checkpoint_plot_path(),
            "evaluation": self.get_evaluation_path(),
            "predictions": self.get_predictions_path(),
            "error_analysis": self.get_error_analysis_path(),
            "analysis_convergence_plot": self.get_analysis_convergence_plot_path(),
            "analysis_metrics_plot": self.get_analysis_metrics_plot_path(),
            "analysis_results": self.get_analysis_results_path(),
            "benchmark_cpu": self.get_benchmark_path("cpu"),
            "benchmark_cuda": self.get_benchmark_path("cuda"),
            "benchmark_raw_cpu": self.get_benchmark_raw_csv_path("cpu"),
            "benchmark_raw_cuda": self.get_benchmark_raw_csv_path("cuda"),
        }
    
    @classmethod
    def from_base_name(cls, base_name: str) -> "FileRegistry":
        """Cria FileRegistry a partir de um base_name existente (sem config)."""
        exp_name, timestamp = base_name.split("__", 1)
        instance = object.__new__(cls)
        instance.config = {}
        instance.experiment_name = exp_name
        instance.timestamp = timestamp
        instance.base_name = base_name
        return instance

    def __repr__(self):
        return f"FileRegistry(experiment='{self.experiment_name}', timestamp='{self.timestamp}')"


# ============================================================================
# ExperimentRecord — visão somente-leitura de um experimento no disco
# ============================================================================

class ExperimentRecord:
    """
    Representa um experimento existente no disco.

    Fonte única de verdade para paths e status de um run.
    Todos os scripts (train, inference, analyze_errors, analysis, manager)
    devem usar esta classe para evitar duplicação de lógica de descoberta.

    Uso:
        from file_registry import list_experiments, ExperimentRecord

        for rec in list_experiments(complete_only=True):
            print(rec.base_name, rec.has_evaluation)
    """

    def __init__(self, pt_path: Path):
        self._pt_path = pt_path
        self._reg     = FileRegistry.from_base_name(pt_path.stem)
        self._meta:   Optional[dict] = None
        self._meta_loaded = False

    # ------------------------------------------------------------------
    # Identidade
    # ------------------------------------------------------------------

    @property
    def base_name(self) -> str:
        return self._reg.base_name

    @property
    def exp_name(self) -> str:
        return self._reg.experiment_name

    # ------------------------------------------------------------------
    # Paths (delegam ao FileRegistry — nunca duplicar aqui)
    # ------------------------------------------------------------------

    @property
    def pt_path(self) -> Path:
        return self._reg.get_model_path()

    @property
    def metadata_path(self) -> Path:
        return self._reg.get_metadata_path()

    @property
    def history_path(self) -> Path:
        return self._reg.get_history_path()

    @property
    def evaluation_path(self) -> Path:
        return self._reg.get_evaluation_path()

    @property
    def predictions_path(self) -> Path:
        return self._reg.get_predictions_path()

    @property
    def error_analysis_path(self) -> Path:
        return self._reg.get_error_analysis_path()

    @property
    def convergence_plot_path(self) -> Path:
        return self._reg.get_analysis_convergence_plot_path()

    @property
    def summary_path(self) -> Path:
        return self._reg._results_exp_dir() / f"{self.base_name}_summary.txt"

    def benchmark_paths(self, device: Optional[str] = None) -> list[Path]:
        """Retorna todos os benchmarks do run, opcionalmente filtrados por device."""
        pattern = f"benchmark_{self.base_name}_*.json" if device is None else f"benchmark_{self.base_name}_{device}.json"
        return sorted(self._reg._results_exp_dir().glob(pattern), key=lambda p: p.stat().st_mtime)

    def latest_benchmark_path(self, device: Optional[str] = None) -> Optional[Path]:
        """Retorna benchmark mais recente do run, opcionalmente por device."""
        paths = self.benchmark_paths(device=device)
        return paths[-1] if paths else None

    def has_benchmark(self, device: Optional[str] = None) -> bool:
        """True se o run já possui benchmark formal, opcionalmente por device."""
        return self.latest_benchmark_path(device=device) is not None

    # ------------------------------------------------------------------
    # Existência de artefatos
    # ------------------------------------------------------------------

    @property
    def has_checkpoint(self) -> bool:
        return self.pt_path.exists()

    @property
    def has_history(self) -> bool:
        return self.history_path.exists()

    @property
    def has_evaluation(self) -> bool:
        return self.evaluation_path.exists()

    @property
    def has_predictions(self) -> bool:
        return self.predictions_path.exists()

    @property
    def has_error_analysis(self) -> bool:
        return self.error_analysis_path.exists()

    @property
    def has_convergence_plot(self) -> bool:
        return self.convergence_plot_path.exists()

    # ------------------------------------------------------------------
    # Metadata (lazy-load)
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> Optional[dict]:
        if not self._meta_loaded:
            path = self.metadata_path
            if path.exists():
                try:
                    self._meta = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    self._meta = None
            self._meta_loaded = True
        return self._meta

    @property
    def training_completed(self) -> bool:
        return bool(self.metadata and self.metadata.get("training_completed", False))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True se o treino está completo e os artefatos essenciais existem."""
        return (self.training_completed
                and self.has_checkpoint
                and self.metadata_path.exists()
                and self.has_history)

    def __repr__(self) -> str:
        status = "COMPLETE" if self.is_complete else "INCOMPLETE"
        return f"ExperimentRecord({self.base_name!r}, {status})"


# ============================================================================
# list_experiments — descoberta centralizada de experimentos
# ============================================================================

def list_experiments(complete_only: bool = False) -> list[ExperimentRecord]:
    """
    Retorna todos os experimentos encontrados em models/, ordenados por mtime.

    Esta é A fonte única de verdade para listagem e indexação de experimentos.
    Todos os scripts devem chamar esta função para garantir índices idênticos.

    Args:
        complete_only: se True, retorna apenas runs com training_completed=True
                       e artefatos essenciais presentes.

    Returns:
        Lista de ExperimentRecord ordenada por st_mtime (mais antigo primeiro).
        O índice posicional (0, 1, 2, ...) é o mesmo em todos os scripts.
    """
    from utils import get_all_models_sorted
    records = [ExperimentRecord(pt) for pt in get_all_models_sorted()]
    if complete_only:
        records = [r for r in records if r.is_complete]
    return records


# ============================================================================
# Funções utilitárias para uso direto
# ============================================================================

def create_registry_from_config(config: dict, timestamp: Optional[str] = None) -> FileRegistry:
    """
    Cria um FileRegistry a partir de um config
    
    Args:
        config: Dict do config.json
        timestamp: Timestamp customizado (opcional)
    
    Returns:
        FileRegistry configurado
    """
    return FileRegistry(config, timestamp)


def extract_experiment_name_from_path(file_path: Path) -> Optional[str]:
    """
    Extrai o nome do experimento de um path de arquivo
    
    Args:
        file_path: Path do arquivo (ex: models/exp2_traditional__20260217_003045.pt)
    
    Returns:
        Nome do experimento (ex: "exp2_traditional") ou None se não conseguir extrair
    """
    name = file_path.stem
    
    # Remove sufixos conhecidos
    for suffix in ["_metadata", "_history", "_checkpoint"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Remove prefixos conhecidos
    for prefix in ["evaluation_", "predictions_", "error_analysis_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Separa experiment_name do timestamp
    # Formato: {experiment_name}__{timestamp}
    if "__" in name:
        experiment_name = name.split("__")[0]
        return experiment_name
    
    return None


def get_base_name_from_path(file_path: Path) -> Optional[str]:
    """
    Extrai o nome base completo (experiment__timestamp) de um path
    
    Args:
        file_path: Path do arquivo
    
    Returns:
        Nome base (ex: "exp2_traditional__20260217_003045") ou None
    """
    name = file_path.stem
    
    # Remove sufixos conhecidos
    for suffix in ["_metadata", "_history", "_checkpoint"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Remove prefixos conhecidos
    for prefix in ["evaluation_", "predictions_", "error_analysis_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Validar formato {experiment}__{timestamp}
    if "__" in name:
        return name
    
    return None
