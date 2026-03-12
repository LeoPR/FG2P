#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SHIM — lógica movida para src/manager/. Este arquivo mantém compatibilidade.
# Uso: python src/manage_experiments.py [flags]
#      python -m src.manager [flags]          (equivalente)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from manager.cli import main
if __name__ == "__main__":
    main()
    sys.exit(0)

# ---- referência rápida ----
# manage.py                         Lista todos
# manage.py --show N                Detalhes do exp N
# manage.py --missing               Tabela de gaps
# manage.py --run N                 Roda pipeline do exp N
# manage.py --run N --force         Re-roda tudo do exp N
# manage.py --run                   Roda pipeline de todos
# manage.py --clean N               Apaga exp N (pede confirmação)
# manage.py --clean-broken          Apaga incompletos/órfãos
# manage.py --compare EXP_NAME      Compara runs

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Fix para encoding console Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from utils import MODELS_DIR, RESULTS_DIR, get_logger

logger = get_logger("manage_experiments")

PERFORMANCE_PATH = Path(__file__).resolve().parent.parent / "docs" / "report" / "performance.json"
REPORT_PATH = RESULTS_DIR / "model_report.html"

_EVAL_METRICS_RE = {
    "per": re.compile(r"PER \(Phoneme Error Rate\):\s*([0-9.]+)%"),
    "wer": re.compile(r"WER \(Word Error Rate\):\s*([0-9.]+)%"),
    "accuracy": re.compile(r"Accuracy \(Word-level\):\s*([0-9.]+)%"),
}

_EXPERIMENT_TO_NAME = {
    "exp0_baseline_70split":                         "FG2P Exp0 (Baseline 70/10/20)",
    "exp1_baseline_60split":                         "FG2P Exp1 (Baseline 60/10/30)",
    "exp2_extended_512hidden":                       "FG2P Exp2 (Extended)",
    "exp3_panphon_trainable":                        "FG2P Exp3 (PanPhon)",
    "exp4_panphon_fixed_24d":                        "FG2P Exp4 (PanPhon Fixed)",
    "exp5_intermediate_60split":                     "FG2P Exp5 (Intermediate)",
    "exp6_distance_aware_loss":                      "FG2P Exp6 (Distance-Aware Loss)",
    "exp7_lambda_lower_bound_0.05":                  "FG2P Exp7 (Lambda Lower Bound (\u03bb=0.05))",
    "exp7_lambda_mid_candidate_0.20":                "FG2P Exp7 (Lambda Mid Candidate (\u03bb=0.20))",
    "exp7_lambda_upper_bound_0.50":                  "FG2P Exp7 (Lambda Upper Bound (\u03bb=0.50))",
    "exp8_panphon_distance_aware":                   "FG2P Exp8 (PanPhon + Distance-Aware)",
    "exp9_intermediate_distance_aware":              "FG2P Exp9 (Intermediate + Distance-Aware)",
    "exp10_extended_distance_aware":                 "FG2P Exp10 (Extended + Distance-Aware)",
    "exp11_baseline_decomposed":                     "FG2P Exp11 (Baseline Decomposed Encoding)",
    "exp101_baseline_60split_separators":            "FG2P Exp101 (Baseline + Syllabic Separator)",
    "exp102_intermediate_60split_separators":        "FG2P Exp102 (Intermediate + Syllabic Separator)",
    "exp103_intermediate_sep_distance_aware":        "FG2P Exp103 (Intermediate + Sep + Distance-Aware)",
    "exp104_intermediate_sep_da_custom_dist":        "FG2P Exp104 (Intermediate + Sep + DA + Custom Dist \u2014 bug)",
    "exp104b_intermediate_sep_da_custom_dist_fixed": "FG2P Exp104b (DA + Sep + Custom Dist Fixed) \u2014 SOTA PER",
    "exp105_reduced_data_50split":                   "FG2P Exp105 (50% train data, with hyphen) \u2014 ablation",
    "exp106_no_hyphen_50split":                      "FG2P Exp106 (50% train data, no hyphen) \u2014 ablation",
}


def _parse_eval_metrics(path: Path) -> Optional[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    metrics = {}
    for key, rx in _EVAL_METRICS_RE.items():
        match = rx.search(text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics or None


def _map_experiment_to_name(exp_name: str) -> Optional[str]:
    if exp_name in _EXPERIMENT_TO_NAME:
        return _EXPERIMENT_TO_NAME[exp_name]
    for key, display in _EXPERIMENT_TO_NAME.items():
        if exp_name.startswith(key):
            return display
    return None


def _load_performance_index() -> dict:
    if not PERFORMANCE_PATH.exists():
        return {}
    try:
        data = json.loads(PERFORMANCE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    models = data.get("fg2p_models", [])
    return {m.get("name"): m for m in models if m.get("name")}


class ExperimentStatus:
    """Classificação de status do experimento"""
    COMPLETE = "completo"
    INCOMPLETE = "incompleto"
    ORPHAN = "órfão"
    RUNNING = "rodando"


class ExperimentArtifacts:
    """Representa todos os artefatos de um experimento"""
    
    def __init__(self, base_name: str):
        """
        Args:
            base_name: Nome base do experimento (ex: exp0_baseline_70split__20260218_044620)
        """
        self.base_name = base_name
        self.artifacts = self._scan_artifacts()
    
    def _scan_artifacts(self) -> dict:
        """Escaneia e retorna todos os artefatos encontrados"""
        exp_name = self.base_name.split("__")[0]
        m_dir = MODELS_DIR / exp_name
        r_dir = RESULTS_DIR / exp_name

        artifacts = {
            # Essenciais (models/{exp_name}/)
            "model_checkpoint": m_dir / f"{self.base_name}.pt",
            "model_metadata": m_dir / f"{self.base_name}_metadata.json",

            # Treinamento (results/{exp_name}/)
            "history_csv": r_dir / f"{self.base_name}_history.csv",
            "summary_txt": r_dir / f"{self.base_name}_summary.txt",
            "results_json": r_dir / f"{self.base_name}_results.json",

            # Avaliação (results/{exp_name}/)
            "evaluation_txt": r_dir / f"evaluation_{self.base_name}.txt",
            "predictions_tsv": r_dir / f"predictions_{self.base_name}.tsv",
            "error_analysis_txt": r_dir / f"error_analysis_{self.base_name}.txt",

            # Visualizações (results/{exp_name}/)
            "convergence_plot": r_dir / f"{self.base_name}_convergence.png",
            "analysis_plot": r_dir / f"{self.base_name}_analysis.png",
            "checkpoint_plot": r_dir / f"{self.base_name}_checkpoint.png",
        }
        
        # Verificar existência
        found = {}
        for key, path in artifacts.items():
            if path.exists():
                found[key] = {
                    "path": path,
                    "size": path.stat().st_size,
                    "mtime": path.stat().st_mtime
                }
        
        return found
    
    def get_metadata(self) -> Optional[dict]:
        """Carrega e retorna metadados do modelo"""
        if "model_metadata" not in self.artifacts:
            return None
        
        metadata_path = self.artifacts["model_metadata"]["path"]
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    
    def classify_status(self) -> str:
        """Classifica o status do experimento"""
        metadata = self.get_metadata()
        
        # Sem modelo ou metadados = órfão
        if "model_checkpoint" not in self.artifacts or metadata is None:
            return ExperimentStatus.ORPHAN
        
        # Verificar se training foi concluído
        training_completed = metadata.get("training_completed", False)
        
        if training_completed:
            # Completo se tiver checkpoint + metadata + history + (evaluation OU summary)
            has_core = all(k in self.artifacts for k in ["model_checkpoint", "model_metadata", "history_csv"])
            has_results = any(k in self.artifacts for k in ["evaluation_txt", "summary_txt", "results_json"])
            
            if has_core and has_results:
                return ExperimentStatus.COMPLETE
            else:
                return ExperimentStatus.INCOMPLETE
        else:
            # Training não concluído: verificar se está rodando (arquivo modificado recentemente)
            if "model_checkpoint" in self.artifacts:
                mtime = self.artifacts["model_checkpoint"]["mtime"]
                age_minutes = (datetime.now().timestamp() - mtime) / 60
                
                # Se foi modificado nos últimos 15 minutos, provavelmente está rodando
                if age_minutes < 15:
                    return ExperimentStatus.RUNNING
            
            return ExperimentStatus.INCOMPLETE
    
    def get_total_size(self) -> int:
        """Retorna tamanho total de todos os artefatos em bytes"""
        return sum(artifact["size"] for artifact in self.artifacts.values())
    
    def count_artifacts(self) -> int:
        """Retorna número de artefatos encontrados"""
        return len(self.artifacts)
    
    def delete_all(self, dry_run: bool = False) -> list[Path]:
        """Remove todos os artefatos do experimento
        
        Args:
            dry_run: Se True, apenas lista o que seria deletado sem deletar
            
        Returns:
            Lista de paths deletados (ou que seriam deletados)
        """
        deleted = []
        for artifact_info in self.artifacts.values():
            path = artifact_info["path"]
            if dry_run:
                deleted.append(path)
            else:
                try:
                    path.unlink()
                    deleted.append(path)
                    logger.info(f"Deletado: {path.name}")
                except OSError as e:
                    logger.warning(f"Erro ao deletar {path.name}: {e}")

        # Remover subpastas vazias após deletar artefatos
        if not dry_run:
            exp_name = self.base_name.split("__")[0]
            for base_dir in [MODELS_DIR, RESULTS_DIR]:
                exp_dir = base_dir / exp_name
                if exp_dir.exists() and not any(exp_dir.iterdir()):
                    exp_dir.rmdir()
                    logger.info(f"Removida pasta vazia: {exp_dir}")

        return deleted


class ExperimentManager:
    """Gerenciador centralizado de experimentos"""
    
    def __init__(self):
        self.experiments = self._discover_experiments()
        self.index_map = {}  # Mapeamento índice_visual → experimento
    
    def _discover_experiments(self) -> list[ExperimentArtifacts]:
        """Descobre todos os experimentos baseado nos modelos .pt"""
        from utils import get_all_models_sorted
        model_files = get_all_models_sorted()
        experiments = []
        
        for model_file in model_files:
            base_name = model_file.stem
            exp = ExperimentArtifacts(base_name)
            experiments.append(exp)
        
        return experiments
    
    def _build_index_map(self, filter_status: Optional[str] = None) -> dict:
        """Constrói mapeamento de índices visuais para experimentos
        
        Retorna dict {índice_visual: experimento} na mesma ordem de exibição
        """
        # Agrupar por status
        by_status = {
            ExperimentStatus.COMPLETE: [],
            ExperimentStatus.INCOMPLETE: [],
            ExperimentStatus.RUNNING: [],
            ExperimentStatus.ORPHAN: []
        }
        
        for exp in self.experiments:
            status = exp.classify_status()
            by_status[status].append(exp)
        
        # Construir mapeamento na ordem de exibição
        index_map = {}
        idx = 0
        for status in [ExperimentStatus.COMPLETE, ExperimentStatus.RUNNING, 
                       ExperimentStatus.INCOMPLETE, ExperimentStatus.ORPHAN]:
            
            if filter_status and status != filter_status:
                continue
            
            for exp in by_status[status]:
                index_map[idx] = exp
                idx += 1
        
        return index_map
    
    def list_experiments(self, filter_status: Optional[str] = None):
        """Lista todos os experimentos com detalhes
        
        Args:
            filter_status: Filtrar por status (completo, incompleto, órfão, rodando)
        """
        if not self.experiments:
            print("\n❌ Nenhum experimento encontrado em models/")
            print("Execute train.py primeiro para treinar um modelo.\n")
            return
        
        print("\n" + "=" * 100)
        print("GERENCIADOR DE EXPERIMENTOS FG2P")
        print("=" * 100)
        
        # Construir mapeamento de índices (atualiza self.index_map)
        self.index_map = self._build_index_map(filter_status)
        
        # Agrupar por status para exibição
        by_status = {
            ExperimentStatus.COMPLETE: [],
            ExperimentStatus.INCOMPLETE: [],
            ExperimentStatus.RUNNING: [],
            ExperimentStatus.ORPHAN: []
        }
        
        for exp in self.experiments:
            status = exp.classify_status()
            by_status[status].append(exp)
        
        # Exibir por grupo
        status_icons = {
            ExperimentStatus.COMPLETE: "✓",
            ExperimentStatus.INCOMPLETE: "⚠",
            ExperimentStatus.RUNNING: "⏳",
            ExperimentStatus.ORPHAN: "❌"
        }
        
        status_labels = {
            ExperimentStatus.COMPLETE: "COMPLETOS",
            ExperimentStatus.INCOMPLETE: "INCOMPLETOS",
            ExperimentStatus.RUNNING: "RODANDO",
            ExperimentStatus.ORPHAN: "ÓRFÃOS"
        }
        
        idx = 0
        for status in [ExperimentStatus.COMPLETE, ExperimentStatus.RUNNING, 
                       ExperimentStatus.INCOMPLETE, ExperimentStatus.ORPHAN]:
            
            if filter_status and status != filter_status:
                continue
            
            experiments_in_status = by_status[status]
            if not experiments_in_status:
                continue
            
            print(f"\n{status_icons[status]} {status_labels[status]} ({len(experiments_in_status)})")
            print("-" * 100)
            
            for exp in experiments_in_status:
                metadata = exp.get_metadata()
                size_mb = exp.get_total_size() / (1024 * 1024)
                
                print(f"\n[{idx}] {exp.base_name}")
                print(f"    Status: {status_icons[status]} {status.upper()}")
                print(f"    Artefatos: {exp.count_artifacts()}/11 arquivos | {size_mb:.2f} MB total")
                
                if metadata:
                    print(f"    Experimento: {metadata.get('experiment_name', 'N/A')}")
                    print(f"    Timestamp: {metadata.get('timestamp', 'N/A')}")
                    
                    current_epoch = metadata.get('current_epoch', metadata.get('final_epoch', '?'))
                    total_epochs = metadata.get('total_epochs', '?')
                    print(f"    Progresso: Epoch {current_epoch}/{total_epochs}")
                    
                    if 'best_loss' in metadata:
                        print(f"    Best Loss: {metadata['best_loss']:.4f}")
                    
                    if 'config' in metadata and 'model' in metadata['config']:
                        config = metadata['config']['model']
                        print(f"    Arquitetura: emb={config.get('emb_dim', '?')} " +
                              f"hidden={config.get('hidden_dim', '?')} " +
                              f"layers={config.get('num_layers', '?')}")
                
                # Mostrar artefatos presentes
                artifact_symbols = {
                    "model_checkpoint": "📦",
                    "model_metadata": "📋",
                    "history_csv": "📊",
                    "evaluation_txt": "✅",
                    "error_analysis_txt": "🔍",
                    "predictions_tsv": "📄",
                    "convergence_plot": "📈",
                    "analysis_plot": "📉",
                }
                
                artifacts_present = [
                    f"{artifact_symbols.get(key, '•')}{key.replace('_', ' ')}"
                    for key in list(exp.artifacts.keys())[:8]  # Primeiros 8
                ]
                
                if artifacts_present:
                    print(f"    Arquivos: {', '.join(artifacts_present)}")
                    if len(exp.artifacts) > 8:
                        print(f"              ... e mais {len(exp.artifacts) - 8} arquivo(s)")
                
                idx += 1
        
        print("\n" + "=" * 100)
        print(f"Total: {len(self.experiments)} experimento(s)")
        print(f"  ✓ Completos: {len(by_status[ExperimentStatus.COMPLETE])}")
        print(f"  ⏳ Rodando: {len(by_status[ExperimentStatus.RUNNING])}")
        print(f"  ⚠ Incompletos: {len(by_status[ExperimentStatus.INCOMPLETE])}")
        print(f"  ❌ Órfãos: {len(by_status[ExperimentStatus.ORPHAN])}")
        print("=" * 100)
        
        print("\nUso:")
        print("  python src/manage_experiments.py --show N            # Detalhes do experimento N")
        print("  python src/manage_experiments.py --prune N           # Remove experimento N")
        print("  python src/manage_experiments.py --prune-incomplete  # Remove todos incompletos")
        print("  python src/manage_experiments.py --missing            # Tabela compacta de gaps")
        print("  python src/manage_experiments.py --guide             # Sugere proximo passo")
        print("  python src/manage_experiments.py --process-all       # Processa tudo pendente")
        print("  python src/manage_experiments.py --stats             # Estatísticas gerais\n")
    
    def show_experiment(self, index: int):
        """Mostra detalhes completos de um experimento específico"""
        # Se index_map ainda não foi construído, construir agora
        if not self.index_map:
            self.index_map = self._build_index_map()
        
        if index not in self.index_map:
            logger.error(f"Índice {index} inválido. Use --list para ver experimentos disponíveis.")
            return
        
        exp = self.index_map[index]
        metadata = exp.get_metadata()
        status = exp.classify_status()
        
        print("\n" + "=" * 100)
        print(f"DETALHES DO EXPERIMENTO [{index}]")
        print("=" * 100)
        
        print(f"\nNome: {exp.base_name}")
        print(f"Status: {status.upper()}")
        print(f"Tamanho total: {exp.get_total_size() / (1024 * 1024):.2f} MB")
        print(f"Artefatos: {exp.count_artifacts()}/11 arquivos")
        
        if metadata:
            print("\n--- METADADOS ---")
            print(f"Experimento: {metadata.get('experiment_name', 'N/A')}")
            print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
            print(f"Training Completed: {metadata.get('training_completed', False)}")
            print(f"Epoch: {metadata.get('current_epoch', '?')}/{metadata.get('total_epochs', '?')}")
            print(f"Best Loss: {metadata.get('best_loss', 'N/A')}")
            print(f"Total Params: {metadata.get('total_params', 'N/A'):,}")
            
            if 'config' in metadata:
                print("\n--- CONFIGURAÇÃO ---")
                config = metadata['config']
                if 'model' in config:
                    print("Model:")
                    for key, value in config['model'].items():
                        print(f"  {key}: {value}")
                if 'training' in config:
                    print("Training:")
                    for key, value in config['training'].items():
                        if key not in ['note', 'notes']:
                            print(f"  {key}: {value}")
        
        print("\n--- ARTEFATOS ENCONTRADOS ---")
        for artifact_name, artifact_info in exp.artifacts.items():
            path = artifact_info["path"]
            size_kb = artifact_info["size"] / 1024
            mtime = datetime.fromtimestamp(artifact_info["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
            
            location = "models/" if path.parent.name == "models" else "results/"
            print(f"  ✓ {artifact_name:25s} | {location:10s} | {size_kb:8.1f} KB | {mtime}")
        
        print("\n" + "=" * 100)
    
    def prune_experiment(self, index: int, dry_run: bool = False):
        """Remove todos os artefatos de um experimento
        
        Args:
            index: Índice do experimento
            dry_run: Se True, apenas mostra o que seria deletado
        """
        # Se index_map ainda não foi construído, construir agora
        if not self.index_map:
            self.index_map = self._build_index_map()
        
        if index not in self.index_map:
            logger.error(f"Índice {index} inválido. Use --list para ver experimentos disponíveis.")
            return
        
        exp = self.index_map[index]
        status = exp.classify_status()
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Removendo experimento [{index}]: {exp.base_name}")
        print(f"Status: {status.upper()}")
        print(f"Artefatos: {exp.count_artifacts()} arquivo(s)")
        print(f"Tamanho total: {exp.get_total_size() / (1024 * 1024):.2f} MB")
        
        if not dry_run:
            confirm = input("\n⚠ ATENÇÃO: Esta ação é irreversível! Continuar? (yes/no): ")
            if confirm.lower() != "yes":
                print("❌ Operação cancelada.")
                return
        
        print()
        deleted = exp.delete_all(dry_run=dry_run)
        
        if dry_run:
            print(f"\n[DRY RUN] Seriam deletados {len(deleted)} arquivo(s):")
            for path in deleted:
                print(f"  - {path}")
        else:
            print(f"\n✓ {len(deleted)} arquivo(s) removido(s) com sucesso.")
    
    def prune_incomplete(self, dry_run: bool = False):
        """Remove todos os experimentos incompletos (exceto rodando)"""
        incomplete = [
            exp for exp in self.experiments
            if exp.classify_status() == ExperimentStatus.INCOMPLETE
        ]
        
        if not incomplete:
            print("\n✓ Nenhum experimento incompleto encontrado.")
            return
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Encontrados {len(incomplete)} experimento(s) incompleto(s):")
        for exp in incomplete:
            print(f"  - {exp.base_name} ({exp.count_artifacts()} artefatos)")
        
        total_size = sum(exp.get_total_size() for exp in incomplete)
        print(f"\nTamanho total: {total_size / (1024 * 1024):.2f} MB")
        
        if not dry_run:
            confirm = input("\n⚠ ATENÇÃO: Esta ação é irreversível! Continuar? (yes/no): ")
            if confirm.lower() != "yes":
                print("❌ Operação cancelada.")
                return
        
        print()
        total_deleted = 0
        for exp in incomplete:
            deleted = exp.delete_all(dry_run=dry_run)
            total_deleted += len(deleted)
        
        if dry_run:
            print(f"\n[DRY RUN] Seriam deletados {total_deleted} arquivo(s) no total.")
        else:
            print(f"\n✓ {total_deleted} arquivo(s) removido(s) de {len(incomplete)} experimento(s).")
    
    def show_stats(self):
        """Mostra estatísticas gerais sobre os experimentos"""
        if not self.experiments:
            print("\n❌ Nenhum experimento encontrado.")
            return
        
        # Classificar experimentos
        by_status = {
            ExperimentStatus.COMPLETE: [],
            ExperimentStatus.INCOMPLETE: [],
            ExperimentStatus.RUNNING: [],
            ExperimentStatus.ORPHAN: []
        }
        
        for exp in self.experiments:
            status = exp.classify_status()
            by_status[status].append(exp)
        
        # Calcular estatísticas
        _total_size = sum(exp.get_total_size() for exp in self.experiments)
        _total_artifacts = sum(exp.count_artifacts() for exp in self.experiments)
        
        print("\n" + "=" * 100)

    def process_all_pending(self, dry_run=False, force=False, force_inference=False, verify_subprocess_dry_run=True, index: int | None = None):
        """Executa automaticamente todos os passos pendentes para completar os experimentos

        Args:
            dry_run: Se True, apenas mostra o que seria executado sem executar
            force: Reexecuta etapas leves (error_analysis, plots, report) mesmo se já existirem
            force_inference: Se True, também força reexecução de inference
            verify_subprocess_dry_run: Em dry-run, consulta inference.py --dry-run para validação cruzada
            index: Se fornecido, processa apenas o experimento com esse índice
        """
        import subprocess
        import sys

        if not self.experiments:
            print("\n❌ Nenhum experimento encontrado.")
            return

        self.index_map = self._build_index_map()

        if index is not None:
            if index not in self.index_map:
                print(f"\n❌ Índice {index} não encontrado. Use --missing ou --list para ver os índices disponíveis.")
                return
            self.index_map = {index: self.index_map[index]}

        print("\n" + "=" * 100)
        if index is not None:
            print(f"PROCESSAMENTO AUTOMÁTICO — EXPERIMENTO [{index}]: {self.index_map[index].base_name}")
        else:
            print("PROCESSAMENTO AUTOMÁTICO DE EXPERIMENTOS")
        print("=" * 100)
        
        # Coletar todas as tarefas pendentes
        tasks = {
            'inference': [],
            'error_analysis': [],
            'plots': [],
            'report': False,
        }

        report_input_mtimes = []
        
        for idx, exp in self.index_map.items():
            status = exp.classify_status()
            base_name = exp.base_name
            artifacts = exp.artifacts
            
            # Apenas processar experimentos completos
            if status != ExperimentStatus.COMPLETE:
                continue
            
            # Verificar inference pendente (ou forçada explicitamente)
            needs_inference = (
                force_inference
                or "evaluation_txt" not in artifacts
                or "predictions_tsv" not in artifacts
            )
            if needs_inference:
                model_path = artifacts.get("model_checkpoint", {}).get("path")
                if model_path:
                    tasks['inference'].append((base_name, model_path))
            
            # Verificar error_analysis pendente (ou forçada)
            needs_error_analysis = (
                "predictions_tsv" in artifacts
                and (force or "error_analysis_txt" not in artifacts)
            )
            if needs_error_analysis:
                pred_path = artifacts["predictions_tsv"]["path"]
                tasks['error_analysis'].append((base_name, pred_path))
            
            # Verificar plots pendentes (ou forçados)
            needs_plot = (
                "history_csv" in artifacts
                and (force or "convergence_plot" not in artifacts)
            )
            if needs_plot:
                tasks['plots'].append(base_name)

            # Insumos para decidir atualização de report
            for key in ("evaluation_txt", "error_analysis_txt", "convergence_plot", "analysis_plot"):
                if key in artifacts:
                    mtime = artifacts[key].get("mtime")
                    if mtime:
                        report_input_mtimes.append(mtime)

        # Decidir geração de report (incremental por timestamp, ou forçada)
        report_mtime = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else 0
        latest_input_mtime = max(report_input_mtimes) if report_input_mtimes else 0
        report_outdated = latest_input_mtime > report_mtime
        tasks['report'] = force or report_outdated
        
        # Mostrar resumo
        total_tasks = len(tasks['inference']) + len(tasks['error_analysis']) + len(tasks['plots']) + (1 if tasks['report'] else 0)
        
        if total_tasks == 0:
            print("\n✓ Todos os experimentos estão completos!")
            print("Nenhuma tarefa pendente detectada.\n")
            return
        
        print(f"\nTarefas pendentes detectadas: {total_tasks}")
        print(f"  • Inference: {len(tasks['inference'])} modelo(s)")
        print(f"  • Error Analysis: {len(tasks['error_analysis'])} experimento(s)")
        print(f"  • Plots: {len(tasks['plots'])} experimento(s)")
        print(f"  • Report HTML: {'1' if tasks['report'] else '0'}")

        if force and not force_inference:
            print("\nModo FORCE (fraco): reexecuta etapas leves, preserva incrementalidade de inference.")
        if force_inference:
            print("\nModo FORCE INFERENCE (forte): inference também será reexecutada.")
        
        if dry_run:
            print("\n[DRY RUN] Comandos que seriam executados:\n")
            
            if tasks['inference']:
                print("1. INFERENCE:")
                for base_name, _ in tasks['inference']:
                    cmd = f"python src/inference.py --model {base_name}"
                    if force_inference:
                        cmd += " --force"
                    print(f"   {cmd}")
            
            if tasks['error_analysis']:
                print("\n2. ERROR ANALYSIS:")
                for base_name, pred_path in tasks['error_analysis']:
                    print(f"   python src/analyze_errors.py --model {base_name}")
            
            if tasks['plots']:
                print("\n3. CONVERGENCE PLOTS:")
                for base_name in tasks['plots']:
                    print(f"   python src/analysis.py --model-name {base_name}")

            if tasks['report']:
                print("\n4. REPORT HTML:")
                print("   python src/reporting/report_generator.py")

            if verify_subprocess_dry_run and tasks['inference']:
                print("\n[DRY RUN] Validação cruzada com inference.py --dry-run:")
                max_checks = 3
                sample = tasks['inference'][:max_checks]
                for base_name, _ in sample:
                    cmd = [sys.executable, "src/inference.py", "--model", base_name, "--dry-run"]
                    if force_inference:
                        cmd.append("--force")
                    try:
                        result = subprocess.run(
                            cmd,
                            cwd=Path.cwd(),
                            capture_output=True,
                            text=True,
                            timeout=20,
                        )
                        if result.returncode == 0:
                            lines = [line for line in result.stdout.splitlines() if "[DRY RUN]" in line]
                            preview = lines[0] if lines else "[DRY RUN] inferência respondeu sem detalhe filtrável"
                            print(f"   {base_name}: {preview}")
                        else:
                            err = (result.stderr or "").strip().splitlines()
                            err_preview = err[0] if err else "sem stderr"
                            print(f"   {base_name}: falha na validação cruzada (code={result.returncode}) | {err_preview}")
                    except Exception as e:
                        print(f"   {base_name}: falha na validação cruzada ({e})")
                if len(tasks['inference']) > max_checks:
                    print(f"   ... validação cruzada resumida: {max_checks}/{len(tasks['inference'])} modelos")
            
            print("\nUse sem --dry-run para executar.")
            return
        
        # Confirmar execução
        print("\n⚠ Esta operação executará automaticamente os comandos acima.")
        confirm = input("Continuar? (yes/no): ")
        if confirm.lower() != "yes":
            print("❌ Operação cancelada.")
            return
        
        success_count = 0
        error_count = 0
        
        # Executar inference
        if tasks['inference']:
            print(f"\n{'='*100}")
            print(f"EXECUTANDO INFERENCE ({len(tasks['inference'])} modelo(s))")
            print(f"{'='*100}")
            
            for base_name, model_path in tasks['inference']:
                print(f"\n→ {base_name}")
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "src/inference.py",
                            "--model",
                            base_name,
                            *( ["--force"] if force_inference else [] ),
                        ],
                        cwd=Path.cwd(),
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 min timeout
                    )
                    if result.returncode == 0:
                        print("  ✓ Inference completo")
                        success_count += 1
                    else:
                        print(f"  ✗ Erro (código {result.returncode})")
                        print(f"  {result.stderr[:200]}")
                        error_count += 1
                except subprocess.TimeoutExpired:
                    print("  ✗ Timeout (>10min)")
                    error_count += 1
                except Exception as e:
                    print(f"  ✗ Erro: {e}")
                    error_count += 1

        # Sincronizar performance.json a partir das avaliações recentes
        if tasks['inference'] or tasks['report']:
            print(f"\n{'='*100}")
            print("SINCRONIZANDO performance.json")
            print(f"{'='*100}")
            try:
                result = subprocess.run(
                    [sys.executable, "src/update_performance.py"],
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    print(f"  ✓ {(result.stdout or '').strip().splitlines()[-1] if result.stdout else 'OK'}")
                    success_count += 1
                else:
                    print(f"  ✗ Erro (código {result.returncode}): {(result.stderr or '')[:200]}")
                    error_count += 1
            except Exception as e:
                print(f"  ✗ Erro: {e}")
                error_count += 1

        # Reconstruir model_registry.json após novos dados de inference
        if tasks['inference']:
            print(f"\n{'='*100}")
            print("ATUALIZANDO model_registry.json")
            print(f"{'='*100}")
            try:
                self.rebuild_registry()
                success_count += 1
            except Exception as e:
                print(f"  ✗ Erro ao reconstruir registry: {e}")
                error_count += 1

        # Gerar report HTML (quando necessário)
        if tasks['report']:
            print(f"\n{'='*100}")
            print("GERANDO REPORT HTML")
            print(f"{'='*100}")
            try:
                result = subprocess.run(
                    [sys.executable, "src/reporting/report_generator.py"],
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    print("  ✓ Report HTML gerado")
                    success_count += 1
                else:
                    print(f"  ✗ Erro (código {result.returncode})")
                    print(f"  {result.stderr[:200]}")
                    error_count += 1
            except subprocess.TimeoutExpired:
                print("  ✗ Timeout (>5min)")
                error_count += 1
            except Exception as e:
                print(f"  ✗ Erro: {e}")
                error_count += 1
        
        # Executar error analysis
        if tasks['error_analysis']:
            print(f"\n{'='*100}")
            print(f"EXECUTANDO ERROR ANALYSIS ({len(tasks['error_analysis'])} experimento(s))")
            print(f"{'='*100}")
            
            for base_name, pred_path in tasks['error_analysis']:
                print(f"\n→ {base_name}")
                try:
                    result = subprocess.run(
                        [sys.executable, "src/analyze_errors.py", "--model", base_name],
                        cwd=Path.cwd(),
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 min timeout
                    )
                    if result.returncode == 0:
                        print("  ✓ Error analysis completo")
                        success_count += 1
                    else:
                        print(f"  ✗ Erro (código {result.returncode})")
                        print(f"  {result.stderr[:200]}")
                        error_count += 1
                except subprocess.TimeoutExpired:
                    print("  ✗ Timeout (>5min)")
                    error_count += 1
                except Exception as e:
                    print(f"  ✗ Erro: {e}")
                    error_count += 1
        
        # Executar plots
        if tasks['plots']:
            print(f"\n{'='*100}")
            print(f"GERANDO PLOTS ({len(tasks['plots'])} experimento(s))")
            print(f"{'='*100}")
            
            for base_name in tasks['plots']:
                print(f"\n→ {base_name}")
                try:
                    result = subprocess.run(
                        [sys.executable, "src/analysis.py", "--model-name", base_name],
                        cwd=Path.cwd(),
                        capture_output=True,
                        text=True,
                        timeout=120  # 2 min timeout
                    )
                    if result.returncode == 0:
                        print("  ✓ Plot gerado")
                        success_count += 1
                    else:
                        print(f"  ✗ Erro (código {result.returncode})")
                        print(f"  {result.stderr[:200]}")
                        error_count += 1
                except subprocess.TimeoutExpired:
                    print("  ✗ Timeout (>2min)")
                    error_count += 1
                except Exception as e:
                    print(f"  ✗ Erro: {e}")
                    error_count += 1
        
        # Resumo final
        print(f"\n{'='*100}")
        print("RESUMO DO PROCESSAMENTO")
        print(f"{'='*100}")
        print(f"Tarefas executadas: {success_count + error_count}/{total_tasks}")
        print(f"  ✓ Sucesso: {success_count}")
        print(f"  ✗ Erros: {error_count}")
        
        if error_count == 0:
            print("\n✓ Todos os experimentos processados com sucesso!")
        else:
            print(f"\n⚠ {error_count} tarefa(s) falharam. Verifique os logs acima.")
        
        print(f"{'='*100}\n")

    def show_missing(self):
        """Tabela compacta de cobertura de artefatos por experimento.

        Mostra apenas experimentos COMPLETOS (treino OK) e indica quais etapas
        de pos-processamento (inference, error_analysis, plots) ainda faltam.
        Experimentos sem nenhuma lacuna aparecem na lista 'OK' ao final.
        """
        if not self.experiments:
            print("\nNenhum experimento encontrado.")
            return

        self.index_map = self._build_index_map()

        # Colunas de interesse e como derivar seu status
        COLS = [
            ("eval",    ("evaluation_txt", "predictions_tsv")),   # ambos devem existir
            ("err_an",  ("error_analysis_txt",)),
            ("plot",    ("convergence_plot",)),
        ]
        COL_LABELS = ["eval", "err_an", "plot"]

        # Separar experimentos completos dos demais
        complete = []
        non_complete = []
        for idx, exp in self.index_map.items():
            status = exp.classify_status()
            if status == ExperimentStatus.COMPLETE:
                complete.append((idx, exp))
            else:
                non_complete.append((idx, exp, status))

        # Calcular largura máxima do nome
        max_name = max((len(exp.base_name) for _, exp in complete), default=40)
        max_name = max(max_name, 40)

        header = f"{'idx':>4}  {'experimento':<{max_name}}  {'eval':^6}  {'err_an':^6}  {'plot':^6}"
        sep    = "-" * len(header)

        print("\n" + "=" * len(header))
        print("COBERTURA DE ARTEFATOS — EXPERIMENTOS COMPLETOS")
        print("=" * len(header))
        print(header)
        print(sep)

        ok_list   = []   # sem lacunas
        gap_lists = {c: [] for c in COL_LABELS}  # lacunas por coluna

        for idx, exp in complete:
            arts = exp.artifacts
            cells = []
            has_gap = False
            for label, keys in COLS:
                present = all(k in arts for k in keys)
                cells.append("OK" if present else "--")
                if not present:
                    has_gap = True
                    gap_lists[label].append((idx, exp.base_name))

            name_display = exp.base_name
            row = f"{idx:>4}  {name_display:<{max_name}}  {cells[0]:^6}  {cells[1]:^6}  {cells[2]:^6}"
            # Destaca linhas com gaps
            marker = " <" if has_gap else ""
            print(row + marker)

            if not has_gap:
                ok_list.append(exp.base_name)

        print(sep)

        # Detectar experimentos com resultados mas sem modelo (checkpoint ausente)
        results_only = []
        for subdir in sorted(RESULTS_DIR.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue
            exp_folder = subdir.name
            has_eval = any(subdir.glob("evaluation_*.txt"))
            model_exists = any((MODELS_DIR / exp_folder).glob("*.pt"))
            if has_eval and not model_exists:
                results_only.append(exp_folder)

        if non_complete:
            print(f"\n(Nao listados: {len(non_complete)} experimento(s) nao-completos"
                  " — use --list para ver status detalhado)")
        if results_only:
            print(f"\nATENCAO: {len(results_only)} experimento(s) com resultados mas SEM modelo .pt"
                  " (checkpoint deletado ou nao movido):")
            for name in results_only:
                print(f"  {name}")
            print("  → Resultados preservados em results/. Para reanalisar, o modelo precisaria ser retreinado.")

        # Resumo de gaps
        print("\nRESUMO DE GAPS")
        print("-" * 50)
        any_gap = False
        for label, keys in COLS:
            gaps = gap_lists[label]
            if gaps:
                any_gap = True
                names = ", ".join(f"[{i}] {n.split('__')[0]}" for i, n in gaps)
                print(f"  {label:8s} faltando em {len(gaps)} exp: {names}")

        if not any_gap:
            print("  Todos os experimentos completos estao totalmente processados.")

        # Comandos sugeridos
        needs_inference    = gap_lists["eval"]
        needs_error        = gap_lists["err_an"]
        needs_plot         = gap_lists["plot"]

        print("\nCOMANDOS SUGERIDOS")
        print("-" * 50)
        if needs_inference:
            for idx, name in needs_inference:
                print(f"  python src/manage_experiments.py --process-all --force-inference --index {idx}")
        if needs_error or needs_plot:
            cheap_indices: dict[int, str] = {}
            for idx, name in needs_error:
                cheap_indices[idx] = name
            for idx, name in needs_plot:
                cheap_indices.setdefault(idx, name)
            for idx in sorted(cheap_indices):
                print(f"  python src/manage_experiments.py --process-all --force --index {idx}")
        if not needs_inference and not needs_error and not needs_plot:
            print("  Nenhuma acao necessaria.")

        print()

    def guide(self):
        """Mostra orientacoes de proximo passo por experimento."""
        if not self.experiments:
            print("\n❌ Nenhum experimento encontrado.")
            return

        self.index_map = self._build_index_map()
        perf_index = _load_performance_index()

        # Estatisticas basicas (reaproveita logica de show_stats)
        by_status = {
            ExperimentStatus.COMPLETE: [],
            ExperimentStatus.INCOMPLETE: [],
            ExperimentStatus.RUNNING: [],
            ExperimentStatus.ORPHAN: []
        }
        for exp in self.experiments:
            status = exp.classify_status()
            by_status[status].append(exp)

        total_size = sum(exp.get_total_size() for exp in self.experiments)
        total_artifacts = sum(exp.count_artifacts() for exp in self.experiments)

        to_infer = []
        to_analyze = []
        to_plot = []
        perf_outdated = []
        latest_eval_mtime = None
        print("\n" + "=" * 100)
        print("ORIENTACAO DE FLUXO (NEXT STEPS)")
        print("=" * 100)

        for idx, exp in self.index_map.items():
            status = exp.classify_status()
            base_name = exp.base_name
            artifacts = exp.artifacts

            print(f"\n[{idx}] {base_name}")
            print(f"    Status: {status.upper()}")

            if status == ExperimentStatus.RUNNING:
                print("    → Em execucao. Aguarde finalizar o treino.")
                continue

            if status == ExperimentStatus.ORPHAN:
                print("    → Orfao: sem metadados ou checkpoint. Sugestao: prune.")
                continue

            if status == ExperimentStatus.INCOMPLETE:
                print("    → Incompleto: treinos interrompidos ou sem artefatos finais.")
                print("      Sugestao: prune (ou retreinar) se nao for mais usar.")
                continue

            # Status COMPLETE: verificar lacunas de avaliacao
            missing = []
            if "evaluation_txt" not in artifacts:
                missing.append("evaluation_txt")
            if "predictions_tsv" not in artifacts:
                missing.append("predictions_tsv")
            if "error_analysis_txt" not in artifacts:
                missing.append("error_analysis_txt")

            if not missing:
                print("    → Completo: treino e avaliacao OK.")
            else:
                print(f"    → Completo, mas faltam: {', '.join(missing)}")
                if "evaluation_txt" in missing or "predictions_tsv" in missing:
                    print(f"      Sugestao: python src/inference.py --model {base_name}")
                    to_infer.append(base_name)
                if "error_analysis_txt" in missing:
                    print(f"      Sugestao: python src/analyze_errors.py --model {base_name}")
                    to_analyze.append(base_name)

            missing_plots = []
            if "history_csv" in artifacts:
                if "convergence_plot" not in artifacts:
                    missing_plots.append("convergence_plot")

            if missing_plots:
                print("    → Falta grafico de convergencia")
                print(f"      Sugestao: python src/analysis.py --model-name {base_name}")
                to_plot.append(base_name)

            # Checar se performance.json esta desatualizado
            if "evaluation_txt" in artifacts:
                eval_mtime = artifacts["evaluation_txt"].get("mtime")
                if eval_mtime is not None:
                    latest_eval_mtime = max(latest_eval_mtime or 0, eval_mtime)
                metadata = exp.get_metadata() or {}
                exp_name = metadata.get("experiment_name", "")
                perf_name = _map_experiment_to_name(exp_name) if exp_name else None
                perf_entry = perf_index.get(perf_name) if perf_name else None
                eval_metrics = _parse_eval_metrics(artifacts["evaluation_txt"]["path"])

                if eval_metrics and perf_entry:
                    perf_metrics = {
                        "per": perf_entry.get("per"),
                        "wer": perf_entry.get("wer"),
                        "accuracy": perf_entry.get("accuracy"),
                    }
                    if eval_metrics != perf_metrics:
                        perf_outdated.append(perf_name)
                elif eval_metrics and not perf_entry:
                    perf_outdated.append(perf_name or base_name)

        report_outdated = False
        if latest_eval_mtime is not None:
            report_mtime = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else 0
            report_outdated = report_mtime < latest_eval_mtime

        print("\nResumo de proximo passo:")
        if to_infer:
            print("  1) Rodar inferencia para: " + ", ".join(sorted(set(to_infer))))
        if to_analyze:
            print("  2) Rodar analyze_errors para: " + ", ".join(sorted(set(to_analyze))))
        if to_plot:
            print("  3) Gerar graficos de treino/validacao para: " + ", ".join(sorted(set(to_plot))))
            print("     → Sugestao: python src/analysis.py --model-name <nome>")
        if perf_outdated:
            print("  4) performance.json possivelmente desatualizado para: " + ", ".join(sorted(set(perf_outdated))))
            print("     → Sugestao: python src/update_performance.py --dry-run")
        if report_outdated:
            print("  5) Report possivelmente desatualizado")
            print("     → Sugestao: python src/reporting/report_generator.py")
        if not to_infer and not to_analyze and not perf_outdated and not report_outdated:
            print("  Nenhuma acao pendente detectada.")

        print("\nDica: use --dry-run antes de --prune/--prune-incomplete se tiver duvidas.")
        print("\n" + "=" * 100)
        print("ESTATÍSTICAS GERAIS")
        print("=" * 100)
        
        print(f"\nTotal de experimentos: {len(self.experiments)}")
        print(f"  ✓ Completos: {len(by_status[ExperimentStatus.COMPLETE])}")
        print(f"  ⏳ Rodando: {len(by_status[ExperimentStatus.RUNNING])}")
        print(f"  ⚠ Incompletos: {len(by_status[ExperimentStatus.INCOMPLETE])}")
        print(f"  ❌ Órfãos: {len(by_status[ExperimentStatus.ORPHAN])}")
        
        print("\nArmazenamento:")
        print(f"  Total: {total_size / (1024 * 1024):.2f} MB")
        print(f"  Artefatos: {total_artifacts} arquivo(s)")
        print(f"  Média por experimento: {total_size / len(self.experiments) / (1024 * 1024):.2f} MB")
        
        # Espaço recuperável
        incomplete_size = sum(
            exp.get_total_size() 
            for exp in by_status[ExperimentStatus.INCOMPLETE]
        )
        orphan_size = sum(
            exp.get_total_size() 
            for exp in by_status[ExperimentStatus.ORPHAN]
        )
        
        print("\nEspaço recuperável:")
        print(f"  Incompletos: {incomplete_size / (1024 * 1024):.2f} MB")
        print(f"  Órfãos: {orphan_size / (1024 * 1024):.2f} MB")
        print(f"  Total: {(incomplete_size + orphan_size) / (1024 * 1024):.2f} MB")
        
        # Distribuição por tipo de artefato
        artifact_counts = {}
        for exp in self.experiments:
            for artifact_name in exp.artifacts.keys():
                artifact_counts[artifact_name] = artifact_counts.get(artifact_name, 0) + 1
        
        print("\nDistribuição de artefatos:")
        for artifact_name, count in sorted(artifact_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {artifact_name:25s}: {count}/{len(self.experiments)} experimentos")
        
        print("\n" + "=" * 100)

    # -------------------------------------------------------------------------
    # Novos métodos: --registry, --compare, --status
    # -------------------------------------------------------------------------

    def rebuild_registry(self):
        """
        Reconstrói models/model_registry.json a partir dos metadados e arquivos de avaliação.

        Regras de promoção:
          - N_test >= 5000 para claims best_per / best_wer
          - O melhor PER e WER são determinados apenas entre modelos elegíveis

        Aliases fixos:
          - comparison_latphon: exp107 (N_test < 5000, não elegível)
          - fast: placeholder até exp106 ter checkpoint
        """
        REGISTRY_PATH = MODELS_DIR / "model_registry.json"
        MIN_TEST = 5000

        # Coletar métricas de todos os modelos com checkpoint + evaluation
        candidates = []
        for exp in self.experiments:
            if "model_checkpoint" not in exp.artifacts:
                continue
            if "evaluation_txt" not in exp.artifacts:
                continue
            meta = exp.get_metadata() or {}
            metrics = _parse_eval_metrics(exp.artifacts["evaluation_txt"]["path"])
            if not metrics:
                continue
            n_test = meta.get("dataset", {}).get("test_size", 0)
            exp_name = meta.get("experiment_name", exp.base_name.split("__")[0])
            run_id   = meta.get("run_id", "")
            uses_sep = bool(meta.get("config", {}).get("data", {})
                            .get("keep_syllable_separators", False))
            candidates.append({
                "experiment": exp_name,
                "run_id": run_id,
                "per": metrics.get("per", 999) / 100,
                "wer": metrics.get("wer", 999) / 100,
                "n_test": n_test,
                "uses_separators": uses_sep,
                "eligible": n_test >= MIN_TEST,
            })

        eligible = [c for c in candidates if c["eligible"]]

        best_per_entry = None
        best_wer_entry = None
        if eligible:
            best_per_entry = min(eligible, key=lambda c: c["per"])
            best_wer_entry = min(eligible, key=lambda c: c["wer"])

        def _make_alias(entry, description, notes=""):
            if entry is None:
                return {"status": "no_eligible_model", "experiment": None}
            return {
                "experiment": entry["experiment"],
                "run_id": entry["run_id"],
                "per": round(entry["per"], 6),
                "wer": round(entry["wer"], 6),
                "n_test": entry["n_test"],
                "uses_separators": entry["uses_separators"],
                "description": description,
                "notes": notes,
            }

        # Verificar se exp106 tem checkpoint
        exp106_dirs = [d for d in MODELS_DIR.iterdir()
                       if d.is_dir() and "exp106" in d.name]
        exp106_has_pt = any(
            list(d.glob("*.pt")) for d in exp106_dirs
        ) if exp106_dirs else False

        fast_alias = {"status": "pending_retrain", "experiment": None, "run_id": None,
                      "per": None, "wer": None, "n_test": None, "uses_separators": None,
                      "description": "Fastest inference model (exp106, no-hyphen, ~2.58x faster). Blocked: no .pt checkpoint.",
                      "notes": "Retrain exp106 to activate."} if not exp106_has_pt else _make_alias(
            next((c for c in eligible if "exp106" in c["experiment"]), None),
            "Fastest inference model.", "exp106 no-hyphen variant."
        )

        # Alias comparison_latphon: exp107 (fixo, não elegível)
        exp107_entry = next(
            (c for c in candidates if "exp107" in c["experiment"]), None
        )
        latphon_alias = {
            "experiment": exp107_entry["experiment"] if exp107_entry else "exp107_maxdata_95train",
            "run_id": exp107_entry["run_id"] if exp107_entry else None,
            "per": round(exp107_entry["per"], 6) if exp107_entry else 0.0046,
            "wer": round(exp107_entry["wer"], 6) if exp107_entry else 0.0556,
            "n_test": exp107_entry["n_test"] if exp107_entry else 960,
            "uses_separators": exp107_entry["uses_separators"] if exp107_entry else True,
            "description": "Methodological comparison with LatPhon 2025 (N=960). NOT eligible for best_per (N < 5000).",
            "notes": "Small N_test intentional: reproduces LatPhon evaluation protocol.",
        }

        registry = {
            "_version": "1.0",
            "_generated_by": "python src/manage_experiments.py --registry",
            "_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "_promotion_rules": {
                "min_test_words": MIN_TEST,
                "note": f"N_test >= {MIN_TEST} required for best_per/best_wer aliases.",
            },
            "aliases": {
                "best_per": _make_alias(
                    best_per_entry,
                    "Best phoneme accuracy. Recommended for TTS and phonetic alignment.",
                    "DA Loss + syllable separators + structural distance correction.",
                ),
                "best_wer": _make_alias(
                    best_wer_entry,
                    "Best word accuracy. Recommended for NLP, search, word-level tasks.",
                    "DA Loss lambda=0.2, intermediate capacity 9.7M params.",
                ),
                "fast": fast_alias,
                "comparison_latphon": latphon_alias,
            },
        }

        REGISTRY_PATH.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(f"\nRegistry salvo em: {REGISTRY_PATH}")
        print(f"\nModelos elegíveis (N_test >= {MIN_TEST}): {len(eligible)}")
        for c in sorted(eligible, key=lambda x: x["per"]):
            marker = ""
            if best_per_entry and c["experiment"] == best_per_entry["experiment"]:
                marker += " [best_per]"
            if best_wer_entry and c["experiment"] == best_wer_entry["experiment"]:
                marker += " [best_wer]"
            print(f"  {c['experiment']}: PER={c['per']:.4%} WER={c['wer']:.4%} N={c['n_test']}{marker}")
        if eligible:
            print(f"\nbest_per → {best_per_entry['experiment']} (PER={best_per_entry['per']:.4%})")
            print(f"best_wer → {best_wer_entry['experiment']} (WER={best_wer_entry['wer']:.4%})")
        print(f"fast     → {'PENDENTE (sem checkpoint)' if not exp106_has_pt else 'OK'}")

    def compare_experiment(self, exp_name: str):
        """
        Compara todos os runs do mesmo experimento (identificados por timestamp).

        Útil após re-treino: mostra métricas lado a lado e aponta qual run tem
        melhor PER, melhor WER e menor best_loss.

        Args:
            exp_name: Nome base do experimento, ex: "exp0_baseline_70split"
        """
        runs = [
            exp for exp in self.experiments
            if exp.base_name.split("__")[0] == exp_name
        ]
        if not runs:
            # Tentativa parcial
            runs = [
                exp for exp in self.experiments
                if exp_name in exp.base_name
            ]
        if not runs:
            print(f"Nenhum run encontrado para '{exp_name}'.")
            print("Use --missing para ver experimentos disponíveis.")
            return
        if len(runs) == 1:
            print(f"Apenas um run encontrado para '{exp_name}'. Nada a comparar.")
            self.show_experiment(
                next(idx for idx, e in self.index_map.items() if e is runs[0])
            )
            return

        print(f"\n{'='*80}")
        print(f"COMPARAÇÃO: {exp_name} ({len(runs)} runs)")
        print(f"{'='*80}")

        rows = []
        for exp in sorted(runs, key=lambda e: e.base_name):
            meta    = exp.get_metadata() or {}
            run_id  = meta.get("run_id", "?")
            best_loss = meta.get("best_loss", None)
            epoch   = meta.get("final_epoch") or meta.get("current_epoch", "?")
            has_pt  = "model_checkpoint" in exp.artifacts
            metrics = None
            if "evaluation_txt" in exp.artifacts:
                metrics = _parse_eval_metrics(exp.artifacts["evaluation_txt"]["path"])
            rows.append({
                "run_id": run_id,
                "epoch": epoch,
                "best_loss": best_loss,
                "per": metrics.get("per") if metrics else None,
                "wer": metrics.get("wer") if metrics else None,
                "has_pt": has_pt,
                "base_name": exp.base_name,
            })

        # Cabeçalho
        print(f"\n{'Run ID':<22} {'Epoch':>6} {'BestLoss':>10} {'PER%':>7} {'WER%':>7}  Checkpoint")
        print("-" * 70)
        for r in rows:
            per_s = f"{r['per']:.2f}" if r["per"] is not None else "  -"
            wer_s = f"{r['wer']:.2f}" if r["wer"] is not None else "  -"
            loss_s = f"{r['best_loss']:.6f}" if r["best_loss"] is not None else "       -"
            pt_s = "[OK]" if r["has_pt"] else "[sem .pt]"
            print(f"{r['run_id']:<22} {r['epoch']:>6} {loss_s:>10} {per_s:>7} {wer_s:>7}  {pt_s}")

        # Resumo: melhor em cada métrica
        with_per = [r for r in rows if r["per"] is not None]
        with_wer = [r for r in rows if r["wer"] is not None]
        with_loss = [r for r in rows if r["best_loss"] is not None]

        print()
        if with_per:
            best = min(with_per, key=lambda r: r["per"])
            print(f"Melhor PER   : run {best['run_id']} ({best['per']:.2f}%)")
        if with_wer:
            best = min(with_wer, key=lambda r: r["wer"])
            print(f"Melhor WER   : run {best['run_id']} ({best['wer']:.2f}%)")
        if with_loss:
            best = min(with_loss, key=lambda r: r["best_loss"])
            print(f"Menor loss   : run {best['run_id']} ({best['best_loss']:.6f})")

        print()
        print("Para remover um run específico:")
        for r in rows:
            idx = next((i for i, e in self.index_map.items() if e.base_name == r["base_name"]), None)
            if idx is not None:
                print(f"  python src/manage_experiments.py --prune {idx}  # {r['run_id']}")

    def show_status(self):
        """
        Mostra experimentos em execução (checkpoint modificado nos últimos 30 min).
        Lê o arquivo de progresso de treino se existir (training_progress.json).
        """
        running = []
        for idx, exp in self.index_map.items():
            if "model_checkpoint" not in exp.artifacts:
                continue
            mtime = exp.artifacts["model_checkpoint"]["mtime"]
            age_minutes = (datetime.now().timestamp() - mtime) / 60
            if age_minutes <= 30:
                running.append((idx, exp, age_minutes))

        if not running:
            print("\nNenhum experimento em execução detectado (checkpoint mais recente > 30min).")
            return

        print(f"\n{'='*70}")
        print(f"EXPERIMENTOS EM EXECUCAO ({len(running)} detectados)")
        print(f"{'='*70}")
        for idx, exp, age in running:
            meta = exp.get_metadata() or {}
            epoch     = meta.get("current_epoch", "?")
            total_ep  = meta.get("total_epochs", "?")
            exp_name  = meta.get("experiment_name", exp.base_name)
            print(f"\n[{idx}] {exp_name}")
            print(f"    Checkpoint modificado: {age:.1f} min atrás")
            print(f"    Epoch: {epoch}/{total_ep}")

            # Tentar ler progress file (se existir)
            r_dir = RESULTS_DIR / exp_name
            progress_files = list(r_dir.glob("training_progress*.json")) if r_dir.exists() else []
            if progress_files:
                latest = max(progress_files, key=lambda p: p.stat().st_mtime)
                try:
                    prog = json.loads(latest.read_text(encoding="utf-8"))
                    ep   = prog.get("epoch", "?")
                    loss = prog.get("train_loss", "?")
                    vloss = prog.get("val_loss", "?")
                    speed = prog.get("epoch_time_s", "?")
                    print(f"    Progress: epoch={ep} train_loss={loss} val_loss={vloss} epoch_time={speed}s")
                except (json.JSONDecodeError, OSError):
                    pass


def main():
    parser = argparse.ArgumentParser(
        description="Gerenciador de Experimentos FG2P",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista todos os experimentos com status"
    )
    
    parser.add_argument(
        "--show",
        type=int,
        metavar="N",
        help="Mostra detalhes completos do experimento N"
    )
    
    parser.add_argument(
        "--prune",
        type=int,
        metavar="N",
        help="Remove todos os artefatos do experimento N"
    )
    
    parser.add_argument(
        "--prune-incomplete",
        action="store_true",
        help="Remove todos os experimentos incompletos (exceto rodando)"
    )

    parser.add_argument(
        "--guide",
        action="store_true",
        help="Mostra orientacoes do que falta rodar por experimento"
    )
    
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Executa automaticamente todos os passos pendentes (inference, error analysis, plots)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="No --process-all, reexecuta etapas leves (error analysis, plots, report) mesmo se artefatos existirem"
    )

    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="No --process-all, também força reexecução de inference (modo forte)"
    )

    parser.add_argument(
        "--index",
        type=int,
        metavar="N",
        help="No --process-all, processa apenas o experimento N (ver índices em --missing)"
    )

    parser.add_argument(
        "--missing",
        action="store_true",
        help="Tabela compacta: quais experimentos faltam inference / error_analysis / plots"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostra estatísticas gerais sobre os experimentos"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula execução sem alterar arquivos"
    )

    parser.add_argument(
        "--filter",
        choices=["completo", "incompleto", "órfão", "rodando"],
        help="Filtra listagem por status"
    )

    parser.add_argument(
        "--registry",
        action="store_true",
        help="Reconstrói models/model_registry.json a partir dos metadados e avaliações"
    )

    parser.add_argument(
        "--compare",
        type=str,
        metavar="EXP_NAME",
        help="Compara todos os runs de um mesmo experimento (ex: exp0_baseline_70split)"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Mostra experimentos em execução (checkpoint modificado nos últimos 30 min)"
    )
    
    args = parser.parse_args()
    
    # Criar manager
    manager = ExperimentManager()
    
    # Executar comando
    if args.missing:
        manager.show_missing()
    elif args.stats:
        manager.show_stats()
    elif args.guide:
        manager.guide()
    elif args.registry:
        manager.rebuild_registry()
    elif args.compare:
        manager.compare_experiment(args.compare)
    elif args.status:
        manager.show_status()
    elif args.process_all:
        manager.process_all_pending(
            dry_run=args.dry_run,
            force=args.force,
            force_inference=args.force_inference,
            verify_subprocess_dry_run=True,
            index=args.index,
        )
    elif args.show is not None:
        manager.show_experiment(args.show)
    elif args.prune is not None:
        manager.prune_experiment(args.prune, dry_run=args.dry_run)
    elif args.prune_incomplete:
        manager.prune_incomplete(dry_run=args.dry_run)
    elif args.list or len(manager.experiments) > 0:
        manager.list_experiments(filter_status=args.filter)
    else:
        print("Nenhuma acao especificada. Use --help para ver opcoes.")


if __name__ == "__main__":
    main()
