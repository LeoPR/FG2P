#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gerenciador de Experimentos FG2P

Ferramenta centralizada para manutenção e gestão de artefatos de experimentos.
Lista, inspeciona e remove experimentos completos, incompletos ou órfãos.

Uso:
    python src/manage_experiments.py --list              # Lista todos os experimentos
    python src/manage_experiments.py --show N            # Mostra detalhes do experimento N
    python src/manage_experiments.py --prune N           # Remove todos os artefatos do experimento N
    python src/manage_experiments.py --prune-incomplete  # Remove todos os experimentos incompletos
    python src/manage_experiments.py --guide             # Orienta proximo passo por experimento
    python src/manage_experiments.py --process-all       # Executa fluxo pendente (inference/analyze/plots/report)
    python src/manage_experiments.py --process-all --force
    python src/manage_experiments.py --process-all --force --force-inference
    python src/manage_experiments.py --stats             # Mostra estatísticas gerais
"""

import argparse
import json
import re
import shutil
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

PERFORMANCE_PATH = Path(__file__).resolve().parent.parent / "docs" / "performance.json"
REPORT_PATH = RESULTS_DIR / "model_report.html"

_EVAL_METRICS_RE = {
    "per": re.compile(r"PER \(Phoneme Error Rate\):\s*([0-9.]+)%"),
    "wer": re.compile(r"WER \(Word Error Rate\):\s*([0-9.]+)%"),
    "accuracy": re.compile(r"Accuracy \(Word-level\):\s*([0-9.]+)%"),
}

_EXPERIMENT_TO_NAME = {
    "exp0_baseline_70split": "FG2P Exp0 (Baseline 70/10/20)",
    "exp1_baseline_60split": "FG2P Exp1 (Baseline 60/10/30)",
    "exp2_extended_512hidden": "FG2P Exp2 (Extended)",
    "exp3_panphon_trainable": "FG2P Exp3 (PanPhon)",
    "exp4_panphon_fixed_24d": "FG2P Exp4 (PanPhon Fixed)",
    "exp5_intermediate": "FG2P Exp5 (Intermediate)",
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
    if exp_name.startswith("exp0_"):
        return _EXPERIMENT_TO_NAME.get("exp0_baseline_70split")
    if exp_name.startswith("exp1_"):
        return _EXPERIMENT_TO_NAME.get("exp1_baseline_60split")
    if exp_name.startswith("exp2_"):
        return _EXPERIMENT_TO_NAME.get("exp2_extended_512hidden")
    if exp_name.startswith("exp3_"):
        return _EXPERIMENT_TO_NAME.get("exp3_panphon_trainable")
    if exp_name.startswith("exp4_"):
        return _EXPERIMENT_TO_NAME.get("exp4_panphon_fixed_24d")
    if exp_name.startswith("exp5_"):
        return _EXPERIMENT_TO_NAME.get("exp5_intermediate")
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
        artifacts = {
            # Essenciais (models/)
            "model_checkpoint": MODELS_DIR / f"{self.base_name}.pt",
            "model_metadata": MODELS_DIR / f"{self.base_name}_metadata.json",
            
            # Treinamento (results/)
            "history_csv": RESULTS_DIR / f"{self.base_name}_history.csv",
            "summary_txt": RESULTS_DIR / f"{self.base_name}_summary.txt",
            "results_json": RESULTS_DIR / f"{self.base_name}_results.json",
            
            # Avaliação (results/)
            "evaluation_txt": RESULTS_DIR / f"evaluation_{self.base_name}.txt",
            "predictions_tsv": RESULTS_DIR / f"predictions_{self.base_name}.tsv",
            "error_analysis_txt": RESULTS_DIR / f"error_analysis_{self.base_name}.txt",
            
            # Visualizações (results/)
            "convergence_plot": RESULTS_DIR / f"{self.base_name}_convergence.png",
            "analysis_plot": RESULTS_DIR / f"{self.base_name}_analysis.png",
            "checkpoint_plot": RESULTS_DIR / f"{self.base_name}_checkpoint.png",
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
                    logger.info(f"✓ Deletado: {path.name}")
                except OSError as e:
                    logger.warning(f"✗ Erro ao deletar {path.name}: {e}")
        
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
            print(f"\n--- METADADOS ---")
            print(f"Experimento: {metadata.get('experiment_name', 'N/A')}")
            print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
            print(f"Training Completed: {metadata.get('training_completed', False)}")
            print(f"Epoch: {metadata.get('current_epoch', '?')}/{metadata.get('total_epochs', '?')}")
            print(f"Best Loss: {metadata.get('best_loss', 'N/A')}")
            print(f"Total Params: {metadata.get('total_params', 'N/A'):,}")
            
            if 'config' in metadata:
                print(f"\n--- CONFIGURAÇÃO ---")
                config = metadata['config']
                if 'model' in config:
                    print(f"Model:")
                    for key, value in config['model'].items():
                        print(f"  {key}: {value}")
                if 'training' in config:
                    print(f"Training:")
                    for key, value in config['training'].items():
                        if key not in ['note', 'notes']:
                            print(f"  {key}: {value}")
        
        print(f"\n--- ARTEFATOS ENCONTRADOS ---")
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
        total_size = sum(exp.get_total_size() for exp in self.experiments)
        total_artifacts = sum(exp.count_artifacts() for exp in self.experiments)
        
        print("\n" + "=" * 100)

    def process_all_pending(self, dry_run=False, force=False, force_inference=False, verify_subprocess_dry_run=True):
        """Executa automaticamente todos os passos pendentes para completar os experimentos
        
        Args:
            dry_run: Se True, apenas mostra o que seria executado sem executar
            force: Reexecuta etapas leves (error_analysis, plots, report) mesmo se já existirem
            force_inference: Se True, também força reexecução de inference
            verify_subprocess_dry_run: Em dry-run, consulta inference.py --dry-run para validação cruzada
        """
        import subprocess
        import sys
        
        if not self.experiments:
            print("\n❌ Nenhum experimento encontrado.")
            return
        
        self.index_map = self._build_index_map()
        
        print("\n" + "=" * 100)
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
                        print(f"  ✓ Inference completo")
                        success_count += 1
                    else:
                        print(f"  ✗ Erro (código {result.returncode})")
                        print(f"  {result.stderr[:200]}")
                        error_count += 1
                except subprocess.TimeoutExpired:
                    print(f"  ✗ Timeout (>10min)")
                    error_count += 1
                except Exception as e:
                    print(f"  ✗ Erro: {e}")
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
                        print(f"  ✓ Error analysis completo")
                        success_count += 1
                    else:
                        print(f"  ✗ Erro (código {result.returncode})")
                        print(f"  {result.stderr[:200]}")
                        error_count += 1
                except subprocess.TimeoutExpired:
                    print(f"  ✗ Timeout (>5min)")
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
                        print(f"  ✓ Plot gerado")
                        success_count += 1
                    else:
                        print(f"  ✗ Erro (código {result.returncode})")
                        print(f"  {result.stderr[:200]}")
                        error_count += 1
                except subprocess.TimeoutExpired:
                    print(f"  ✗ Timeout (>2min)")
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
                print(f"    → Falta grafico de convergencia")
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
        
        print(f"\nArmazenamento:")
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
        
        print(f"\nEspaço recuperável:")
        print(f"  Incompletos: {incomplete_size / (1024 * 1024):.2f} MB")
        print(f"  Órfãos: {orphan_size / (1024 * 1024):.2f} MB")
        print(f"  Total: {(incomplete_size + orphan_size) / (1024 * 1024):.2f} MB")
        
        # Distribuição por tipo de artefato
        artifact_counts = {}
        for exp in self.experiments:
            for artifact_name in exp.artifacts.keys():
                artifact_counts[artifact_name] = artifact_counts.get(artifact_name, 0) + 1
        
        print(f"\nDistribuição de artefatos:")
        for artifact_name, count in sorted(artifact_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {artifact_name:25s}: {count}/{len(self.experiments)} experimentos")
        
        print("\n" + "=" * 100)


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
    
    args = parser.parse_args()
    
    # Criar manager
    manager = ExperimentManager()
    
    # Executar comando
    if args.stats:
        manager.show_stats()
    elif args.guide:
        manager.guide()
    elif args.process_all:
        manager.process_all_pending(
            dry_run=args.dry_run,
            force=args.force,
            force_inference=args.force_inference,
            verify_subprocess_dry_run=True,
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
        print("❌ Nenhuma ação especificada. Use --help para ver opções.")


if __name__ == "__main__":
    main()
