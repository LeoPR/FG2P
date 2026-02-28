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

Arquivos gerados:
- Modelo: models/{base}.pt
- Metadata: models/{base}_metadata.json
- History CSV: results/{base}_history.csv
- Evaluation: results/evaluation_{base}.txt
- Predictions: results/predictions_{base}.tsv
- Error analysis: results/error_analysis_{base}.txt
- Checkpoint plot: results/{base}_checkpoint.png
"""

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
    # Arquivos de modelo (models/)
    # ========================================================================
    
    def get_model_path(self) -> Path:
        """Retorna path do arquivo .pt do modelo"""
        return MODELS_DIR / f"{self.base_name}.pt"
    
    def get_metadata_path(self) -> Path:
        """Retorna path do arquivo _metadata.json"""
        return MODELS_DIR / f"{self.base_name}_metadata.json"
    
    # ========================================================================
    # Arquivos de treinamento (results/)
    # ========================================================================
    
    def get_history_path(self) -> Path:
        """Retorna path do CSV de histórico de treino"""
        return RESULTS_DIR / f"{self.base_name}_history.csv"
    
    def get_checkpoint_plot_path(self) -> Path:
        """Retorna path do plot de convergência"""
        return RESULTS_DIR / f"{self.base_name}_checkpoint.png"
    
    # ========================================================================
    # Arquivos de avaliação (results/)
    # ========================================================================
    
    def get_evaluation_path(self) -> Path:
        """Retorna path do arquivo de métricas de avaliação"""
        return RESULTS_DIR / f"evaluation_{self.base_name}.txt"
    
    def get_predictions_path(self) -> Path:
        """Retorna path do TSV de predições word-level"""
        return RESULTS_DIR / f"predictions_{self.base_name}.tsv"
    
    def get_error_analysis_path(self) -> Path:
        """Retorna path do arquivo de análise de erros"""
        return RESULTS_DIR / f"error_analysis_{self.base_name}.txt"
    
    # ========================================================================
    # Arquivos de análise de treinamento (results/)
    # ========================================================================
    
    def get_analysis_convergence_plot_path(self) -> Path:
        """Retorna path do gráfico de convergência (loss curves)"""
        return RESULTS_DIR / f"{self.base_name}_convergence.png"
    
    def get_analysis_metrics_plot_path(self) -> Path:
        """Retorna path do gráfico de análise (gap + throughput)"""
        return RESULTS_DIR / f"{self.base_name}_analysis.png"
    
    def get_analysis_results_path(self) -> Path:
        """Retorna path do JSON com métricas de análise calculadas"""
        return RESULTS_DIR / f"{self.base_name}_results.json"
    
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
        }
    
    def __repr__(self):
        return f"FileRegistry(experiment='{self.experiment_name}', timestamp='{self.timestamp}')"


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
