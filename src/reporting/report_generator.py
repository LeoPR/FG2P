#!/usr/bin/env python
"""
Gerador de relatórios HTML unificado para FG2P

Cria página HTML interativa com:
- Listagem de todos os modelos treinados
- Seleção para comparar até N modelos simultaneamente
- Métricas de treino (loss, convergência)
- Métricas de teste (PER/WER/Accuracy)
- Análise PanPhon graduada
- Benchmark com literatura

Uso:
    python src/reporting/report_generator.py                    # Gera relatório completo
    python src/reporting/report_generator.py --list             # Lista modelos disponíveis
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    get_logger, RESULTS_DIR, MODELS_DIR, PROJECT_ROOT as ROOT_DIR,
    get_model_summary
)

logger = get_logger("report_generator")


def load_performance_data():
    """Carrega benchmarks manuais do arquivo docs/report/performance.json"""
    perf_path = ROOT_DIR / "docs" / "report" / "performance.json"
    if not perf_path.exists():
        return None

    try:
        with open(perf_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Falha ao ler {perf_path.name}: {exc}")
        return None


def _pick_plot(plots: list[Path], suffix: str) -> Path | None:
    for plot in plots:
        if plot.name.endswith(suffix):
            return plot
    return None


def _relative_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _extract_experiment_name(model: dict[str, Any]) -> str:
    return model.get("config", {}).get("experiment", {}).get("name", model.get("name", ""))


def _extract_experiment_index(name: str) -> int:
    match = re.search(r"exp\s*_?(\d+)", name, re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


def _model_sort_key(model: dict[str, Any]) -> tuple[int, str, str]:
    exp_name = _extract_experiment_name(model)
    return (
        _extract_experiment_index(exp_name),
        exp_name.lower(),
        model.get("name", "").lower(),
    )


def render_benchmark_rows(entries: list[dict]) -> str:
    """Renderiza linhas HTML para tabela de benchmarks (com Speed)"""
    if not entries:
        return "<tr><td colspan=\"6\" style=\"text-align:center; color:#666;\">Sem dados</td></tr>"

    rows = ""
    for entry in entries:
        per = entry.get("per")
        wer = entry.get("wer")
        acc = entry.get("accuracy")
        speed_wps = entry.get("inference_speed_wps")
        per_str = f"{per:.2f}%" if per is not None else "—"
        wer_str = f"{wer:.2f}%" if wer is not None else "—"
        acc_str = f"{acc:.2f}%" if acc is not None else "—"
        speed_str = f"{speed_wps:.2f}" if speed_wps is not None else "—"
        notes = entry.get("notes", "")
        rows += (
            "<tr>"
            f"<td>{entry.get('name', 'N/A')}</td>"
            f"<td>{per_str}</td>"
            f"<td>{wer_str}</td>"
            f"<td>{acc_str}</td>"
            f"<td style=\"text-align:center;\">{speed_str}</td>"
            f"<td>{notes}</td>"
            "</tr>"
        )
    return rows


# ============================================================================
# Data Loading and Parsing (usando utils.py)
# ============================================================================

def list_available_models() -> list[dict[str, Any]]:
    """Lista todos os modelos com metadata disponível usando utils.py"""
    models = []

    # Usar glob recursivo para buscar em subpastas de experimentos (FileRegistry pattern)
    for metadata_file in sorted(MODELS_DIR.glob("**/*_metadata.json")):
        try:
            # Extract model name
            model_name = metadata_file.stem.replace("_metadata", "")
            
            # Use get_model_summary from utils.py
            summary = get_model_summary(model_name)
            if not summary or not summary.get("metadata"):
                logger.warning(f"Metadata não encontrado para {model_name}")
                continue
            
            # Build model dict compatible with existing code
            metadata = summary["metadata"]
            artifacts = summary["artifacts"]

            plots = artifacts.get("plots", []) or []
            convergence_plot = _pick_plot(plots, "_convergence.png")
            
            # Filter evaluations/predictions to match THIS model name
            evals = [e for e in artifacts.get("evaluations", []) if model_name in e.stem]
            preds = [p for p in artifacts.get("predictions", []) if model_name in p.stem]
            
            models.append({
                "name": model_name,
                "metadata_file": metadata_file,
                "history_csv": artifacts.get("history"),
                "evaluation_file": evals[0] if evals else None,
                "predictions_file": preds[0] if preds else None,
                "convergence_plot": convergence_plot,
                "metadata": metadata,
                "config": metadata.get("config", {}),
                "completed": metadata.get("training_completed", False),
                "best_loss": metadata.get("best_loss", None),
                "current_epoch": metadata.get("current_epoch", "?"),
                "total_epochs": metadata.get("total_epochs", "?"),
                # Add artifact counts
                "artifact_counts": summary["counts"],
            })
        except Exception as e:
            logger.warning(f"Erro ao carregar {metadata_file}: {e}")
            continue
    
    models.sort(key=_model_sort_key)
    return models


def load_training_history(csv_path: Path) -> list[dict]:
    """Carrega histórico de treino do CSV"""
    if not csv_path or not csv_path.exists():
        return []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_evaluation_results(eval_file: Path) -> dict[str, Any]:
    """Parse do arquivo evaluation_*.txt para extrair métricas"""
    if not eval_file or not eval_file.exists():
        return {}
    
    results = {
        "per": None,
        "wer": None,
        "accuracy": None,
        "test_words": None,
        "correct_words": None,
        "graduated_wer": None,
        "weighted_per": None,
        "class_a": None,
        "class_b": None,
        "class_c": None,
        "class_d": None,
    }
    
    import re
    
    with open(eval_file, "r", encoding="utf-8") as f:
        content = f.read()
        
        # Parse métricas clássicas — formato: "PER (Phoneme Error Rate): 0.58%"
        for line in content.split("\n"):
            line_stripped = line.strip()
            
            # PER — matches "PER:" or "PER (...):"
            per_match = re.search(r'PER[^:]*:\s*([\d.]+)%', line_stripped)
            if per_match and results["per"] is None:
                try:
                    results["per"] = float(per_match.group(1))
                except ValueError:
                    pass
            
            # WER — matches "WER:" or "WER (...):"
            wer_match = re.search(r'WER[^:]*:\s*([\d.]+)%', line_stripped)
            if wer_match and results["wer"] is None:
                try:
                    results["wer"] = float(wer_match.group(1))
                except ValueError:
                    pass
            
            # Accuracy — matches "Accuracy (Word-level): 95.11%"
            acc_match = re.search(r'Accuracy[^:]*:\s*([\d.]+)%', line_stripped)
            if acc_match and results["accuracy"] is None:
                try:
                    results["accuracy"] = float(acc_match.group(1))
                except ValueError:
                    pass
            
            # Test words — "Test set: 28782 words" or "Total palavras: 28782"
            tw_match = re.search(r'(?:Test set|Total palavras)[^:]*:\s*([\d,]+)', line_stripped)
            if tw_match and results["test_words"] is None:
                try:
                    results["test_words"] = int(tw_match.group(1).replace(",", ""))
                except ValueError:
                    pass
            
            # Correct words — "Correct words: 27374/28782" or "Corretas: 27374/28782"
            cw_match = re.search(r'(?:Correct|Corretas)[^:]*:\s*([\d,]+)', line_stripped)
            if cw_match and results["correct_words"] is None:
                try:
                    results["correct_words"] = int(cw_match.group(1).replace(",", ""))
                except ValueError:
                    pass
    
    return results


def load_predictions(pred_file: Path) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Carrega predições do TSV"""
    if not pred_file or not pred_file.exists():
        return [], [], []
    
    words = []
    predictions = []
    references = []
    
    with open(pred_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            words.append(row["word"])
            predictions.append(row["prediction"].split())
            references.append(row["reference"].split())
    
    return words, predictions, references


def _find_error_analysis_file(model_name: str) -> Path | None:
    """Busca error_analysis em subdiretório (FileRegistry pattern) ou raiz."""
    # Primeiro: subdiretório (padrão atual: results/exp_base/error_analysis_model.txt)
    matches = list(RESULTS_DIR.glob(f"*/error_analysis_{model_name}.txt"))
    if matches:
        return matches[0]
    # Fallback: raiz (legado)
    flat = RESULTS_DIR / f"error_analysis_{model_name}.txt"
    return flat if flat.exists() else None


def run_analyze_errors_if_needed(model_name: str, predictions_file: Path) -> bool:
    """Executa analyze_errors.py se o arquivo de análise não existir"""
    error_file = _find_error_analysis_file(model_name)

    if error_file is not None:
        return True
    
    if not predictions_file or not predictions_file.exists():
        logger.warning(f"Predictions não encontradas para {model_name}")
        return False
    
    logger.info(f"Gerando análise de erros para {model_name}...")
    
    try:
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "src/analyze_errors.py", "--model", model_name],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Análise concluída: {error_file.name}")
            return True
        else:
            logger.error(f"Erro ao executar analyze_errors: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Falha ao executar analyze_errors: {e}")
        return False


def load_error_analysis(model_name: str) -> dict[str, Any]:
    """Parse do arquivo error_analysis_*.txt para extrair análise detalhada"""
    error_file = _find_error_analysis_file(model_name)

    if error_file is None:
        return {}
    
    import re
    
    results = {
        "graduated_wer": None,
        "weighted_per": None,
        "class_distribution": {},
        "class_proportions": {},
        "word_distribution": {},
        "word_proportions": {},
        "top_confusions": [],
        "word_examples": {"B": [], "C": [], "D": []},
        "anomalies": {
            "length_stats": {},
            "truncated": [],
            "overgenerated": [],
            "hallucinations": []
        }
    }
    
    with open(error_file, "r", encoding="utf-8") as f:
        content = f.read()
        
        # Parse métricas graduadas
        grad_wer = re.search(r'WER Graduado:\s*([\d.]+)%', content)
        if grad_wer:
            results["graduated_wer"] = float(grad_wer.group(1))
        
        grad_per = re.search(r'PER Ponderado:\s*([\d.]+)%', content)
        if grad_per:
            results["weighted_per"] = float(grad_per.group(1))
        
        # Parse distribuição de classes (POR FONEMA)
        for cls in ['A', 'B', 'C', 'D']:
            cls_match = re.search(rf'Classe {cls}:\s*([\d]+)\s*\(\s*([\d.]+)%\)', content)
            if cls_match:
                results["class_distribution"][cls] = int(cls_match.group(1))
                results["class_proportions"][cls] = float(cls_match.group(2))
        
        # Parse distribuição POR PALAVRA (WER SEGMENTADO)
        word_section = re.search(
            r'WER SEGMENTADO \(por pior classe na palavra\)\n-+\n(.+?)\n\n',
            content, re.DOTALL
        )
        if word_section:
            for cls in ['A', 'B', 'C', 'D']:
                word_match = re.search(rf'Classe {cls}:\s*([\d]+)\s+palavras\s*\(([\d.]+)%\)', word_section.group(1))
                if word_match:
                    results["word_distribution"][cls] = int(word_match.group(1))
                    results["word_proportions"][cls] = float(word_match.group(2))
        
        # Parse confusões fonéticas (top 10)
        confusion_section = re.search(
            r'TOP-10 CONFUSÕES FONÉTICAS\n-+\n(.+?)\n\n',
            content, re.DOTALL
        )
        if confusion_section:
            confusion_lines = confusion_section.group(1).strip().split('\n')[1:]  # Skip header
            for line in confusion_lines[:10]:
                match = re.search(r'(\S+)\s*→\s*(\S+)\s+(\d+)\s+([\d.]+)\s+(\w+)', line)
                if match:
                    results["top_confusions"].append({
                        "ref": match.group(1).strip(),
                        "pred": match.group(2).strip(),
                        "count": int(match.group(3)),
                        "distance": float(match.group(4)),
                        "class": match.group(5).strip()
                    })
        
        # Parse exemplos por classe (dentro da seção EXEMPLOS POR CLASSE)
        examples_section = re.search(
            r'EXEMPLOS POR CLASSE.*?\n-+\n(.+?)(?:TOP-15|TOP-10|\Z)',
            content, re.DOTALL
        )
        if examples_section:
            examples_text = examples_section.group(1)
            for cls in ['B', 'C', 'D']:
                cls_section = re.search(
                    rf'Classe {cls}:\n(.+?)(?:\nClasse [BCD]:\n|\nTOP-15|\nTOP-10|\Z)',
                    examples_text, re.DOTALL
                )
                if cls_section:
                    example_lines = cls_section.group(1).strip().split('\n')[:10]
                    for line in example_lines:
                        word_match = re.search(r'(\S+)\s+score=([\d.]+)\s+pred=([^\s]+.*?)\s+ref=([^\s]+.*?)$', line.strip())
                        if word_match:
                            results["word_examples"][cls].append({
                                "word": word_match.group(1),
                                "score": float(word_match.group(2)),
                                "pred": word_match.group(3),
                                "ref": word_match.group(4)
                            })
        
        # Parse anomalias comportamentais
        anomalies_section = re.search(
            r'ANOMALIAS COMPORTAMENTAIS\n-+\n(.+?)(?:\nTOP-10|\Z)',
            content, re.DOTALL
        )
        if anomalies_section:
            anom_text = anomalies_section.group(1)
            
            # Length stats
            length_match = re.search(r'Length stats: mean=([\d\+\-\.]+) std=([\d\.]+) median=([\d\+\-]+)', anom_text)
            if length_match:
                results["anomalies"]["length_stats"] = {
                    "mean": float(length_match.group(1)),
                    "std": float(length_match.group(2)),
                    "median": int(length_match.group(3))
                }
            
            # Counts
            trunc_count = re.search(r'Truncated: (\d+) palavras', anom_text)
            if trunc_count:
                results["anomalies"]["truncated_count"] = int(trunc_count.group(1))
            
            overgen_count = re.search(r'Over-generated: (\d+) palavras', anom_text)
            if overgen_count:
                results["anomalies"]["overgenerated_count"] = int(overgen_count.group(1))
            
            halluc_count = re.search(r'Hallucinations: (\d+) palavras', anom_text)
            if halluc_count:
                results["anomalies"]["hallucinations_count"] = int(halluc_count.group(1))
            
            # Truncated examples
            trunc_section = re.search(r'Top-10 Truncadas:\n(.+?)(?:\n\n|Top-10|\Z)', anom_text, re.DOTALL)
            if trunc_section:
                for line in trunc_section.group(1).strip().split('\n')[:10]:
                    trunc_match = re.search(r'(\S+)\s+ref=\s*(\d+)\s+pred=\s*(\d+)\s+\(Δ([\d\+\-]+)\)', line)
                    if trunc_match:
                        results["anomalies"]["truncated"].append({
                            "word": trunc_match.group(1),
                            "ref_len": int(trunc_match.group(2)),
                            "pred_len": int(trunc_match.group(3)),
                            "diff": int(trunc_match.group(4))
                        })
            
            # Hallucination examples
            halluc_section = re.search(r'Top-10 Alucinações:\n(.+?)(?:\n\n|TOP-10|\Z)', anom_text, re.DOTALL)
            if halluc_section:
                lines = halluc_section.group(1).strip().split('\n')
                i = 0
                while i < len(lines) and len(results["anomalies"]["hallucinations"]) < 10:
                    line = lines[i].strip()
                    halluc_match = re.search(r'(\S+)\s+patterns=\[(.+?)\]', line)
                    if halluc_match:
                        word = halluc_match.group(1)
                        patterns = halluc_match.group(2)
                        pred = ""
                        ref = ""
                        # Next 2 lines are pred and ref
                        if i+1 < len(lines):
                            pred_line = lines[i+1].strip()
                            pred_match = re.search(r'pred:\s+(.+)', pred_line)
                            if pred_match:
                                pred = pred_match.group(1)
                        if i+2 < len(lines):
                            ref_line = lines[i+2].strip()
                            ref_match = re.search(r'ref:\s+(.+)', ref_line)
                            if ref_match:
                                ref = ref_match.group(1)
                        
                        results["anomalies"]["hallucinations"].append({
                            "word": word,
                            "patterns": patterns,
                            "pred": pred,
                            "ref": ref
                        })
                        i += 3
                    else:
                        i += 1
    
    return results


def load_dataset_statistics() -> dict[str, Any]:
    """Carrega estatísticas permanentes do dataset do cache"""
    cache_file = ROOT_DIR / "data" / "dataset_stats.json"
    
    if not cache_file.exists():
        logger.warning(f"Cache de estatísticas não encontrado: {cache_file}")
        logger.info("Execute 'python src/compute_dataset_stats.py' para gerar o cache")
        return {}
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        logger.info(f"✓ Estatísticas do dataset carregadas: {cache_file.name}")
        return stats
    except Exception as e:
        logger.warning(f"Erro ao carregar estatísticas do dataset: {e}")
        return {}


def generate_representativeness_section_html(repr_metrics: dict[str, Any]) -> str:
    """Gera seção HTML com métricas de representatividade do dataset"""
    quality = repr_metrics.get("quality_summary", {})
    cramers = repr_metrics.get("cramers_v", {})
    chi = repr_metrics.get("chi_square", {})
    cv = repr_metrics.get("coefficient_of_variation", {})
    ci = repr_metrics.get("confidence_intervals", {})
    
    quality_class = quality.get("classification", "").lower()
    quality_color = "#28a745" if quality_class == "excelente" else "#17a2b8" if quality_class == "bom" else "#ffc107"
    
    html = f"""
                <!-- Representativeness Metrics -->
                <h3 style="margin-top: 25px;">Métricas de Representatividade</h3>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <span style="font-size: 1.2em; font-weight: bold; color: {quality_color};">
                            Qualidade do Split: {quality.get("classification", "N/A").upper()}
                        </span>
                        <span style="margin-left: 10px; color: #666;">
                            ({quality.get("score", 0)}/{quality.get("max_score", 10)} pontos)
                        </span>
                    </div>
                    <p style="color: #666; font-size: 0.9em; margin-bottom: 0;">
                        ✓ {", ".join(quality.get("factors", []))}
                    </p>
                </div>
                
                <table>
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                        <th>Interpretação</th>
                    </tr>
"""
    
    if cramers:
        cramers_val = cramers.get("value", 0)
        cramers_interp = cramers.get("interpretation", "N/A")
        cramers_color = "#28a745" if cramers_val < 0.1 else "#17a2b8" if cramers_val < 0.3 else "#ffc107"
        html += f"""
                    <tr>
                        <td>
                            <span class="tooltip-trigger"><strong>Cramér's V</strong>
                                <span class="tooltip-text">
                                    Mede a força da associação entre splits (0 = independentes, 1 = dependentes).<br><br>
                                    <strong>Excelente:</strong> &lt; 0.1 (splits homogêneos, praticamente independentes)<br>
                                    <strong>Bom:</strong> 0.1-0.3 (baixa associação)<br>
                                    <strong>Ruim:</strong> &gt; 0.3 (alta associação, splits não balanceados)
                                </span>
                            </span>
                        </td>
                        <td class="metric-value" style="color: {cramers_color};">{cramers_val:.4f}</td>
                        <td style="color: {cramers_color};">{cramers_interp}</td>
                    </tr>
"""
    
    if chi:
        chi_pval = chi.get("p_value", 0)
        chi_interp = chi.get("interpretation", "N/A")
        chi_color = "#28a745" if chi_pval > 0.5 else "#17a2b8" if chi_pval > 0.2 else "#ffc107"
        html += f"""
                    <tr>
                        <td>
                            <span class="tooltip-trigger"><strong>χ² p-value</strong>
                                <span class="tooltip-text">
                                    Teste qui-quadrado: avalia se as distribuições de comprimento são independentes entre splits.<br><br>
                                    <strong>Excelente:</strong> p &gt; 0.5 (splits muito similares, hipótese nula aceita)<br>
                                    <strong>Bom:</strong> p &gt; 0.2 (splits razoavelmente similares)<br>
                                    <strong>Revisar:</strong> p &lt; 0.05 (diferenças significativas entre splits)
                                </span>
                            </span>
                        </td>
                        <td class="metric-value" style="color: {chi_color};">{chi_pval:.4f}</td>
                        <td style="color: {chi_color};">{chi_interp}</td>
                    </tr>
"""
    
    if cv:
        cv_val = cv.get("value", 0)
        cv_interp = cv.get("interpretation", "N/A")
        cv_color = "#28a745" if cv_val < 1 else "#17a2b8" if cv_val < 3 else "#ffc107"
        html += f"""
                    <tr>
                        <td>
                            <span class="tooltip-trigger"><strong>Coeficiente de Variação</strong>
                                <span class="tooltip-text">
                                    Variabilidade relativa das médias de comprimento entre splits (%).<br><br>
                                    <strong>Excelente:</strong> &lt; 1% (splits muito consistentes)<br>
                                    <strong>Bom:</strong> &lt; 3% (baixa variabilidade)<br>
                                    <strong>Moderado:</strong> 3-5% (variabilidade aceitável)<br>
                                    <strong>Alto:</strong> &gt; 5% (verificar balanceamento)
                                </span>
                            </span>
                        </td>
                        <td class="metric-value" style="color: {cv_color};">{cv_val:.2f}%</td>
                        <td style="color: {cv_color};">{cv_interp}</td>
                    </tr>
"""
    
    if ci and ci.get("splits"):
        overlap = ci.get("all_overlapping", False)
        overlap_str = "✓ Sim" if overlap else "✗ Não"
        overlap_color = "#28a745" if overlap else "#dc3545"
        html += f"""
                    <tr>
                        <td>
                            <span class="tooltip-trigger"><strong>Sobreposição ICs 95%</strong>
                                <span class="tooltip-text">
                                    Verifica se os intervalos de confiança (95%) das médias se sobrepõem.<br><br>
                                    <strong>Sim (✓):</strong> Splits estatisticamente indistinguíveis (excelente)<br>
                                    <strong>Não (✗):</strong> Diferenças significativas entre splits (revisar)
                                </span>
                            </span>
                        </td>
                        <td class="metric-value" style="color: {overlap_color};">{overlap_str}</td>
                        <td style="color: {overlap_color};">{ci.get('overlap_interpretation', 'N/A')}</td>
                    </tr>
"""
    
    html += """
                </table>
"""
    
    # Intervalos de confiança detalhados
    if ci and ci.get("splits"):
        html += """
                <h4 style="margin-top: 20px; margin-bottom: 10px;">Intervalos de Confiança (95%)</h4>
                <table style="font-size: 0.9em;">
                    <thead>
                        <tr>
                            <th>Split</th>
                            <th>Média</th>
                            <th>IC 95% Inferior</th>
                            <th>IC 95% Superior</th>
                            <th>Margem</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for split_name in ["train", "val", "test"]:
            split_ci = ci["splits"].get(split_name, {})
            if not split_ci:
                continue
            
            html += f"""
                        <tr>
                            <td><strong>{split_name.upper()}</strong></td>
                            <td class="metric-value">{split_ci.get('mean', 0):.2f}</td>
                            <td class="metric-value">{split_ci.get('ci_95_lower', 0):.2f}</td>
                            <td class="metric-value">{split_ci.get('ci_95_upper', 0):.2f}</td>
                            <td class="metric-value">±{split_ci.get('margin', 0):.2f}</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
"""
    
    return html


def extract_split_configurations(models: list[dict]) -> list[dict]:
    """Extrai e agrega estatísticas dos splits usados nos experimentos
    
    Returns:
        Lista de dicts com métricas agregadas por configuração de split:
        - ratio: string "70/10/20"
        - train_ratio, val_ratio, test_ratio: floats
        - experiments: lista de nomes
        - dataset_stats: média das métricas de qualidade dos experimentos
    """
    splits_dict = {}
    
    for model in models:
        config = model.get("config", {})
        metadata = model.get("metadata", {})
        data_config = config.get("data", {})
        dataset_info = metadata.get("dataset", {})
        
        test_ratio = data_config.get("test_ratio", 0.20)
        val_ratio = data_config.get("val_ratio", 0.10)
        train_ratio = 1.0 - test_ratio - val_ratio
        
        split_key = f"{int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}"
        
        if split_key not in splits_dict:
            splits_dict[split_key] = {
                "ratio": split_key,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "experiments": [],
                "dataset_samples": []
            }
        
        splits_dict[split_key]["experiments"].append(model["name"])
        
        # Coletar métricas de dataset se disponível
        if dataset_info:
            splits_dict[split_key]["dataset_samples"].append({
                "total_words": dataset_info.get("total_words", 0),
                "train_size": dataset_info.get("train_size", 0),
                "val_size": dataset_info.get("val_size", 0),
                "test_size": dataset_info.get("test_size", 0),
                "char_vocab_size": dataset_info.get("char_vocab_size", 0),
                "phoneme_vocab_size": dataset_info.get("phoneme_vocab_size", 0),
                "jsd": dataset_info.get("jsd", {}),
                "chi2_pvalue": dataset_info.get("chi2_pvalue", {}),
                "cramers_v": dataset_info.get("cramers_v", {}),
                "verdict": dataset_info.get("verdict", {}),
                "train_phoneme_coverage": dataset_info.get("train_phoneme_coverage", 0),
                "val_phoneme_coverage": dataset_info.get("val_phoneme_coverage", 0),
                "train_bigram_coverage": dataset_info.get("train_bigram_coverage", 0),
                "val_bigram_coverage": dataset_info.get("val_bigram_coverage", 0),
            })
    
    # Agregar estatísticas por split
    for split_key, split_data in splits_dict.items():
        samples = split_data["dataset_samples"]
        if not samples:
            split_data["stats"] = None
            continue
        
        # Pegar primeiro sample (todos deveriam ser iguais para mesmo split_key)
        first = samples[0]
        
        # Calcular médias das métricas que podem variar
        import numpy as np
        
        def safe_mean(values):
            valid = [v for v in values if v is not None and v > 0]
            return float(np.mean(valid)) if valid else 0.0
        
        def safe_mean_dict(dict_key, metric_key):
            values = [s.get(dict_key, {}).get(metric_key, 0) for s in samples]
            return safe_mean(values)
        
        split_data["stats"] = {
            "total_words": first["total_words"],
            "train_size": first["train_size"],
            "val_size": first["val_size"],
            "test_size": first["test_size"],
            "char_vocab_size": first["char_vocab_size"],
            "phoneme_vocab_size": first["phoneme_vocab_size"],
            
            # Métricas de estratificação (médias)
            "jsd_stress": safe_mean_dict("jsd", "stress_type"),
            "jsd_syllable": safe_mean_dict("jsd", "syllable_bin"),
            "jsd_length": safe_mean_dict("jsd", "length_bin"),
            "jsd_ratio": safe_mean_dict("jsd", "ratio_bin"),
            
            "chi2_stress": safe_mean_dict("chi2_pvalue", "stress_type"),
            "chi2_syllable": safe_mean_dict("chi2_pvalue", "syllable_bin"),
            "chi2_length": safe_mean_dict("chi2_pvalue", "length_bin"),
            "chi2_ratio": safe_mean_dict("chi2_pvalue", "ratio_bin"),
            
            "cramers_stress": safe_mean_dict("cramers_v", "stress_type"),
            "cramers_syllable": safe_mean_dict("cramers_v", "syllable_bin"),
            "cramers_length": safe_mean_dict("cramers_v", "length_bin"),
            "cramers_ratio": safe_mean_dict("cramers_v", "ratio_bin"),
            
            # Cobertura
            "train_phoneme_coverage": safe_mean([s["train_phoneme_coverage"] for s in samples]),
            "val_phoneme_coverage": safe_mean([s["val_phoneme_coverage"] for s in samples]),
            "train_bigram_coverage": safe_mean([s["train_bigram_coverage"] for s in samples]),
            "val_bigram_coverage": safe_mean([s["val_bigram_coverage"] for s in samples]),
            
            # Veredito (pegar do primeiro, assumindo consistência)
            "verdict_quality": first["verdict"].get("quality", "N/A"),
            "verdict_confidence": first["verdict"].get("confidence", "N/A"),
        }
        
        # Remover samples para economizar espaço
        del split_data["dataset_samples"]
    
    # Sort by train_ratio descending (70/10/20 before 60/10/30)
    return sorted(splits_dict.values(), key=lambda x: x["train_ratio"], reverse=True)


def generate_dataset_section_html(split_configs: list[dict] = None) -> str:
    """Gera seção HTML com comparação de configurações de split
    
    Args:
        split_configs: Lista de configurações de split extraídas dos experimentos
            Cada item deve ter: ratio, experiments, stats (com métricas de estratificação)
    """
    if not split_configs:
        return """
            <div class="section">
                <h2>📊 Configurações de Dataset</h2>
                <p style="color: #999;">Informações não disponíveis.</p>
            </div>
"""
    
    html = f"""
            <div class="section">
                <h2>📊 Configurações de Dataset</h2>
                
                <p style="color: #666; margin-bottom: 20px;">
                    Os experimentos utilizam <strong>{len(split_configs)} configurações diferentes</strong> de divisão do dataset.
                    Todos partem do mesmo vocabulário base (95.9k palavras) mas com proporções diferentes de treino/validação/teste.
                </p>
                
                <!-- Comparação Principal -->
                <h3>Tamanhos e Proporções</h3>
                <table>
                    <thead>
                        <tr>
                            <th rowspan="2">Split<br>(Train/Val/Test)</th>
                            <th rowspan="2">Experimentos</th>
                            <th colspan="3">Tamanho dos Conjuntos</th>
                            <th rowspan="2">Vocabul��rio<br>Char / Fon</th>
                        </tr>
                        <tr>
                            <th>Train</th>
                            <th>Val</th>
                            <th>Test</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for split_conf in split_configs:
        ratio = split_conf["ratio"]
        stats = split_conf.get("stats")
        experiments = split_conf["experiments"]
        
        # Abbreviate experiment names
        if len(experiments) <= 2:
            exp_str = ", ".join([e.replace("_baseline", "").replace("_extended", "").replace("_panphon", "").replace("_intermediate", "").replace("_distance", "").split("_")[0] for e in experiments])
        else:
            exp_str = f"{len(experiments)} experimentos"
        
        if not stats:
            html += f"""
                        <tr>
                            <td><strong>{ratio}</strong></td>
                            <td style="font-size: 0.85em; color: #666;">{exp_str}</td>
                            <td colspan="4" style="text-align: center; color: #999;">Dados não disponíveis</td>
                        </tr>
"""
            continue
        
        train_size = stats["train_size"]
        val_size = stats["val_size"]
        test_size = stats["test_size"]
        char_vocab = stats["char_vocab_size"]
        phon_vocab = stats["phoneme_vocab_size"]
        
        html += f"""
                        <tr>
                            <td><strong>{ratio}</strong></td>
                            <td style="font-size: 0.85em; color: #666;">{exp_str}</td>
                            <td class="metric-value">{train_size:,}</td>
                            <td class="metric-value">{val_size:,}</td>
                            <td class="metric-value">{test_size:,}</td>
                            <td class="metric-value">{char_vocab} / {phon_vocab}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
                
                <!-- Métricas de Estratificação -->
                <h3 style="margin-top: 30px;">Qualidade da Estratificação</h3>
                <p style="color: #666; margin-bottom: 15px;">
                    Métricas estatísticas que avaliam se os splits (train/val/test) têm distribuições similares.
                    <strong>Objetivo:</strong> Garantir que o modelo seja avaliado em dados representativos.
                </p>
                
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 25px;">
"""
    
    for split_conf in split_configs:
        ratio = split_conf["ratio"]
        stats = split_conf.get("stats")
        
        if not stats:
            continue
        
        # Get quality verdict
        verdict_quality = stats.get("verdict_quality", "N/A")

        # Color code based on quality
        if verdict_quality.lower() == "excelente":
            quality_color = "#28a745"
            quality_icon = "✓"
        elif verdict_quality.lower() == "bom":
            quality_color = "#17a2b8"
            quality_icon = "✓"
        else:
            quality_color = "#ffc107"
            quality_icon = "⚠"
        
        html += f"""
                    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f8f9fa;">
                        <h4 style="margin: 0 0 12px 0; color: #333;">
                            Split {ratio}
                            <span style="float: right; color: {quality_color}; font-size: 1.1em;">{quality_icon} {verdict_quality.title()}</span>
                        </h4>
                        
                        <table style="width: 100%; font-size: 0.85em;">
                            <tr style="border-bottom: 1px solid #ddd;">
                                <td colspan="2" style="font-weight: bold; padding: 6px 0; color: #555;">Jensen-Shannon Divergence (JSD)</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">Stress Type</td>
                                <td class="metric-value" style="padding: 3px 0;">{stats["jsd_stress"]:.2e}</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">Syllable Bins</td>
                                <td class="metric-value" style="padding: 3px 0;">{stats["jsd_syllable"]:.2e}</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">Length Bins</td>
                                <td class="metric-value" style="padding: 3px 0;">{stats["jsd_length"]:.2e}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #e0e0e0;">
                                <td style="padding: 3px 0 8px 0;">Ratio Bins</td>
                                <td class="metric-value" style="padding: 3px 0 8px 0;">{stats["jsd_ratio"]:.2e}</td>
                            </tr>
                            
                            <tr style="border-bottom: 1px solid #ddd;">
                                <td colspan="2" style="font-weight: bold; padding: 6px 0; color: #555;">Chi² p-value</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">Stress Type</td>
                                <td class="metric-value" style="padding: 3px 0;">{stats["chi2_stress"]:.3f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">Syllable Bins</td>
                                <td class="metric-value" style="padding: 3px 0;">{stats["chi2_syllable"]:.3f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">Length Bins</td>
                                <td class="metric-value" style="padding: 3px 0;">{stats["chi2_length"]:.3f}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #e0e0e0;">
                                <td style="padding: 3px 0 8px 0;">Ratio Bins</td>
                                <td class="metric-value" style="padding: 3px 0 8px 0;">{stats["chi2_ratio"]:.3f}</td>
                            </tr>
                            
                            <tr style="border-bottom: 1px solid #ddd;">
                                <td colspan="2" style="font-weight: bold; padding: 6px 0; color: #555;">Cramér's V</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">Stress Type</td>
                                <td class="metric-value" style="padding: 3px 0;">{stats["cramers_stress"]:.4f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">Syllable Bins</td>
                                <td class="metric-value" style="padding: 3px 0;">{stats["cramers_syllable"]:.4f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">Length Bins</td>
                                <td class="metric-value" style="padding: 3px 0;">{stats["cramers_length"]:.4f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 3px 0;">Ratio Bins</td>
                                <td class="metric-value" style="padding: 3px 0;">{stats["cramers_ratio"]:.4f}</td>
                            </tr>
                        </table>
                    </div>
"""
    
    html += """
                </div>
                
                <div style="padding: 15px; background: #e3f2fd; border-left: 4px solid #2196F3; border-radius: 4px; margin-bottom: 25px;">
                    <strong>📖 Interpretação das Métricas:</strong>
                    <ul style="margin: 10px 0 0 20px; padding: 0; font-size: 0.9em;">
                        <li><strong>JSD (Jensen-Shannon Divergence):</strong> Mede diferença entre distribuições. Quanto mais próximo de 0, mais similares são os splits. Valores &lt;0.01 são excelentes.</li>
                        <li><strong>Chi² p-value:</strong> Testa se splits são estatisticamente independentes. p-value &gt;0.05 indica que splits são homogêneos (bom). Valores próximos de 1.0 são excelentes.</li>
                        <li><strong>Cramér's V:</strong> Força da associação entre split e distribuição. Quanto mais próximo de 0, melhor. Valores &lt;0.01 são excelentes.</li>
                        <li><strong>Veredito Geral:</strong> Considera todas as métricas em conjunto. "Excelente" significa splits estatisticamente indistinguíveis.</li>
                    </ul>
                </div>
                
                <!-- Cobertura de Vocabulário -->
                <h3>Cobertura de Vocabulário</h3>
                <p style="color: #666; margin-bottom: 15px;">
                    Percentual de fonemas e bigramas presentes no treino que também aparecem na validação.
                    Alta cobertura garante que o modelo veja exemplos representativos durante treino.
                </p>
                
                <table>
                    <thead>
                        <tr>
                            <th>Split</th>
                            <th>Cobertura de Fonemas Train→Val</th>
                            <th>Cobertura de Bigramas Train→Val</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for split_conf in split_configs:
        ratio = split_conf["ratio"]
        stats = split_conf.get("stats")
        
        if not stats:
            continue
        
        phon_cov_val = stats["val_phoneme_coverage"] * 100
        bi_cov_val = stats["val_bigram_coverage"] * 100
        
        html += f"""
                        <tr>
                            <td><strong>{ratio}</strong></td>
                            <td class="metric-value">{phon_cov_val:.1f}%</td>
                            <td class="metric-value">{bi_cov_val:.1f}%</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
                
                <!-- Comparação de Impacto -->
                <h3 style="margin-top: 30px;">Impacto das Configurações nos Experimentos</h3>
                <div style="padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;">
                    <strong>💡 Análise Comparativa:</strong>
                    <ul style="margin: 10px 0 0 20px; padding: 0; font-size: 0.9em;">
"""
    
    if len(split_configs) >= 2:
        split_70 = next((s for s in split_configs if "70/" in s["ratio"]), None)
        split_60 = next((s for s in split_configs if "60/" in s["ratio"]), None)
        
        if split_70 and split_60 and split_70.get("stats") and split_60.get("stats"):
            train_diff = split_70["stats"]["train_size"] - split_60["stats"]["train_size"]
            test_diff = split_60["stats"]["test_size"] - split_70["stats"]["test_size"]
            train_pct = (train_diff / split_70["stats"]["train_size"]) * 100
            test_pct = (test_diff / split_70["stats"]["test_size"]) * 100
            
            html += f"""
                        <li><strong>70/10/20 vs 60/10/30:</strong> Trocar de 70/10/20 para 60/10/30 reduz treino em {train_diff:,} palavras ({train_pct:.1f}%) e aumenta teste em {test_diff:,} palavras (+{test_pct:.1f}%).</li>
                        <li><strong>Trade-off:</strong> Menos dados de treino pode resultar em PER ligeiramente maior, mas conjunto de teste 50% maior permite avaliação mais robusta de generalização.</li>
                        <li><strong>Estratificação:</strong> Ambos os splits mantêm qualidade "excelente" de estratificação, garantindo comparabilidade.</li>
                        <li><strong>Validação:</strong> Mantém-se constante (~9.6k palavras) em ambos os splits, garantindo early stopping consistente.</li>
"""
    
    html += """
                    </ul>
                </div>
                
            </div>
"""
    
    return html



# HTML Generation
# ============================================================================

def generate_html_header(title: str = "FG2P - Relatório de Modelos") -> str:
    """Gera cabeçalho HTML com estilos"""
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}

        .section-note {{
            color: #666;
            font-size: 0.95em;
            margin-bottom: 15px;
        }}
        
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}

        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 550px));
            gap: 16px;
            justify-content: start;
        }}

        .plot-card {{
            background: #fff;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 12px;
        }}

        .plot-title {{
            font-size: 1em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 2px solid #f0f0f0;
        }}

        .plot-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }}

        .plot-missing {{
            border-style: dashed;
            color: #666;
            text-align: center;
        }}

        .plot-empty {{
            font-size: 0.9em;
            padding: 18px 8px;
        }}

        .plot-hint {{
            font-size: 0.8em;
            color: #888;
            margin-top: 6px;
        }}
        
        .models-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .model-card {{
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .model-card:hover {{
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }}
        
        .model-card.selected {{
            border-color: #667eea;
            background: #f0f3ff;
        }}
        
        .model-card input[type="checkbox"] {{
            float: right;
            width: 20px;
            height: 20px;
            cursor: pointer;
        }}
        
        .model-name {{
            font-weight: bold;
            color: #2c3e50;
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        
        .model-info {{
            font-size: 0.9em;
            color: #6c757d;
            line-height: 1.6;
        }}
        
        .model-info span {{
            display: block;
        }}
        
        .status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-top: 8px;
        }}
        
        .status.completed {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status.training {{
            background: #fff3cd;
            color: #856404;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .metric-value {{
            font-weight: 600;
            color: #667eea;
        }}
        
        .metric-best {{
            color: #28a745;
            font-weight: 700;
        }}
        
        /* Tooltip styles */
        .tooltip-trigger {{
            position: relative;
            cursor: help;
            border-bottom: 1px dotted #667eea;
        }}
        
        .tooltip-text {{
            visibility: hidden;
            width: 350px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 12px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -175px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.85em;
            line-height: 1.4;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        
        .tooltip-text::after {{
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }}
        
        .tooltip-trigger:hover .tooltip-text {{
            visibility: visible;
            opacity: 1;
        }}
        
        .button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            margin: 10px 5px;
            transition: all 0.3s ease;
        }}
        
        .button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        .comparison-panel {{
            display: none;
            margin-top: 30px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .comparison-panel.active {{
            display: block;
        }}
        
        .benchmark-table {{
            margin-top: 20px;
        }}
        
        .chart-container {{
            margin-top: 20px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        
        footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        .legend {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        
        /* Error Analysis Styles */
        .error-analysis-section {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .error-header {{
            cursor: pointer;
            padding: 10px;
            background: white;
            border-radius: 6px;
            margin-bottom: 10px;
            transition: background 0.2s;
        }}
        
        .error-header:hover {{
            background: #f0f3ff;
        }}
        
        .toggle-icon {{
            float: right;
            font-size: 1.2em;
            transition: transform 0.3s;
        }}
        
        .toggle-icon.rotated {{
            transform: rotate(-90deg);
        }}
        
        .error-content {{
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        
        .error-subsection {{
            margin: 20px 0;
            padding: 15px;
            background: white;
            border-radius: 6px;
        }}
        
        .error-subsection h5 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        .confusion-table {{
            width: 100%;
            font-size: 0.9em;
        }}
        
        .confusion-table code {{
            background: #f0f3ff;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .class-distribution {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .class-bar {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px;
            border-radius: 6px;
            background: #f8f9fa;
            transition: background 0.2s;
        }}
        
        .class-bar:hover {{
            background: #e9ecef;
        }}
        
        .class-label {{
            min-width: 80px;
            font-weight: bold;
            color: #495057;
        }}
        
        .progress-bar {{
            flex: 1;
            height: 24px;
            background: #e9ecef;
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        
        .bar-success {{
            background: linear-gradient(90deg, #28a745, #5cb85c);
        }}
        
        .bar-info {{
            background: linear-gradient(90deg, #17a2b8, #5bc0de);
        }}
        
        .bar-warning {{
            background: linear-gradient(90deg, #ffc107, #ffdb58);
        }}
        
        .bar-danger {{
            background: linear-gradient(90deg, #dc3545, #f56c6c);
        }}
        
        .class-value {{
            min-width: 120px;
            text-align: right;
            font-weight: 600;
            color: #495057;
        }}
        
        /* Modal Styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.5);
            animation: fadeIn 0.3s;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .modal-content {{
            background-color: white;
            margin: 5% auto;
            padding: 30px;
            border-radius: 12px;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            animation: slideDown 0.3s;
        }}
        
        @keyframes slideDown {{
            from {{
                transform: translateY(-50px);
                opacity: 0;
            }}
            to {{
                transform: translateY(0);
                opacity: 1;
            }}
        }}
        
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #667eea;
        }}
        
        .modal-header h3 {{
            color: #667eea;
            margin: 0;
        }}
        
        .close-modal {{
            font-size: 28px;
            font-weight: bold;
            color: #aaa;
            cursor: pointer;
            transition: color 0.2s;
        }}
        
        .close-modal:hover {{
            color: #667eea;
        }}
        
        .example-item {{
            padding: 12px;
            margin: 8px 0;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        
        .example-word {{
            font-weight: bold;
            color: #2c3e50;
            margin-right: 10px;
        }}
        
        .example-pred {{
            color: #dc3545;
        }}
        
        .example-ref {{
            color: #28a745;
        }}
        
        .example-score {{
            float: right;
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.85em;
        }}

        .subsection {{
            margin-top: 25px;
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }}

        .subsection h4 {{
            color: #333;
            margin-bottom: 12px;
            font-size: 1.1em;
        }}

        .analysis-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
            border-radius: 4px;
            overflow: hidden;
        }}

        .analysis-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}

        .analysis-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e9ecef;
        }}

        .analysis-table tr:last-child td {{
            border-bottom: none;
        }}

        .analysis-table code {{
            background: #f0f3ff;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
"""


def generate_html_footer() -> str:
    """Gera rodapé HTML"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    <!-- Modal for Word Examples -->
    <div id="examplesModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modalTitle">Exemplos de Palavras</h3>
                <span class="close-modal" onclick="closeModal()">&times;</span>
            </div>
            <div id="modalExamples"></div>
        </div>
    </div>
    
    <footer>
        <p>FG2P - Conversão Grapheme-to-Phoneme para Português Brasileiro</p>
        <p>Relatório gerado em: {now}</p>
    </footer>
</body>
</html>
"""


def generate_model_list_html(models: list[dict]) -> str:
    """Gera HTML com lista de modelos disponíveis"""
    html = """
    <div class="section">
        <h2>📊 Modelos Disponíveis</h2>
        <p>Selecione até 3 modelos para comparação detalhada:</p>
        
        <div class="models-grid" id="modelsGrid">
"""
    
    for model in models:
        status_class = "completed" if model["completed"] else "training"
        status_text = "✓ Treinamento Completo" if model["completed"] else "⏳ Em Andamento"
        
        # Extrair config info
        config = model["config"]
        description = config.get("description", "Sem descrição")
        emb_dim = config.get("model", {}).get("emb_dim", config.get("embedding_dim", "?"))
        hidden_dim = config.get("model", {}).get("hidden_dim", config.get("hidden_dim", "?"))
        embedding_type = config.get("model", {}).get("embedding_type", config.get("embedding_type", "learned"))
        
        # Extrair split usado por este modelo
        data_config = config.get("data", {})
        test_ratio = data_config.get("test_ratio", 0.20)
        val_ratio = data_config.get("val_ratio", 0.10)
        train_ratio = 1.0 - test_ratio - val_ratio
        split_display = f"{int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}"
        
        html += f"""
            <div class="model-card" onclick="toggleModel('{model['name']}')">
                <input type="checkbox" id="select_{model['name']}" onchange="updateSelection()" />
                <div class="model-name">{model['name']}</div>
                <div style="color: #666; font-size: 0.9em; margin: 8px 0; font-style: italic;">{description}</div>
                <div class="model-info">
                    <span>Split: {split_display} (Train/Val/Test)</span>
                    <span>Embedding: {embedding_type} ({emb_dim}D)</span>
                    <span>Hidden: {hidden_dim}</span>
                    <span>Épocas: {model['current_epoch']}/{model['total_epochs']}</span>
                    {f'<span>Best Loss: {model["best_loss"]:.6f}</span>' if model['best_loss'] else ''}
                </div>
                <span class="status {status_class}">{status_text}</span>
            </div>
"""
    
    html += """
        </div>
        
        <div style="text-align: center; margin-top: 30px;">
            <button class="button" onclick="selectAll()">
                Selecionar Todos
            </button>
            <button class="button" onclick="clearSelection()">
                Limpar Seleção
            </button>
        </div>
    </div>
"""
    
    return html


def generate_training_plots_html(models: list[dict]) -> str:
    """Gera seção de gráficos de treino/validacao a partir dos plots existentes."""
    html = """
    <div class="section">
        <h2>📈 Convergência de Treino</h2>
        <p class="section-note">Gráficos gerados a partir do history.csv (use analysis.py se estiver vazio).</p>
        <div class="plot-grid">
    """

    for model in models:
        model_name = model["name"]
        exp_name = model["config"].get("experiment", {}).get("name", model_name)
        conv_plot = model.get("convergence_plot")

        if conv_plot:
            conv_path = _relative_path(conv_plot)
            html += f"""
            <div class="plot-card" data-model="{model_name}">
                <div class="plot-title">📊 {exp_name}</div>
                <a href="{conv_path}" target="_blank" rel="noopener">
                    <img src="{conv_path}" alt="{exp_name} - Convergência">
                </a>
            </div>
            """
        else:
            html += f"""
            <div class="plot-card plot-missing" data-model="{model_name}">
                <div class="plot-title">📊 {exp_name}</div>
                <div class="plot-empty">Gráfico ausente</div>
                <div class="plot-hint">Gere com: python src/analysis.py --model-name {model_name}</div>
            </div>
            """

    html += """
        </div>
    </div>
    """

    return html


def generate_comparison_html(models: list[dict]) -> str:
    """Gera HTML com comparação entre modelos — tabelas pré-renderizadas com dados reais"""
    perf_data = load_performance_data()

    # --- Seção 1: Comparação dos Modelos Treinados (sempre visível) ---
    html = """
    <div class="section">
        <h2>🔬 Comparação dos Modelos</h2>
"""

    # Métricas de Treino — pré-renderizadas com dados dos metadados
    html += """
        <h3>Métricas de Treino</h3>
        <table>
            <thead>
                <tr>
                    <th>Modelo</th>
                    <th>Split (Train/Val/Test)</th>
                    <th>Épocas</th>
                    <th>Best Loss</th>
                    <th>Embedding</th>
                    <th>Hidden Dim</th>
                    <th>Parâmetros</th>
                </tr>
            </thead>
            <tbody>
"""
    for model in models:
        cfg = model["config"]
        emb_type = cfg.get("model", {}).get("embedding_type", cfg.get("embedding_type", "learned"))
        emb_dim = cfg.get("model", {}).get("emb_dim", cfg.get("embedding_dim", "?"))
        hidden = cfg.get("model", {}).get("hidden_dim", cfg.get("hidden_dim", "?"))
        params = model["metadata"].get("total_params", None)
        params_str = f"{params/1e6:.1f}M" if params else "?"
        best_loss = model.get("best_loss")
        best_loss_str = f"{best_loss:.6f}" if best_loss else "—"
        
        # Extrair split
        data_config = cfg.get("data", {})
        test_ratio = data_config.get("test_ratio", 0.20)
        val_ratio = data_config.get("val_ratio", 0.10)
        train_ratio = 1.0 - test_ratio - val_ratio
        split_display = f"{int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}"
        
        html += (
            "<tr>"
            f"<td><strong>{cfg.get('experiment', {}).get('name', model['name'])}</strong></td>"
            f"<td>{split_display}</td>"
            f"<td>{model['current_epoch']}/{model['total_epochs']}</td>"
            f"<td>{best_loss_str}</td>"
            f"<td>{emb_type} ({emb_dim}D)</td>"
            f"<td>{hidden}</td>"
            f"<td>{params_str}</td>"
            "</tr>"
        )
    html += """
            </tbody>
        </table>
"""

    # Métricas de Teste Clássicas — parseadas dos evaluation files
    html += """
        <h3>Métricas de Teste (Clássicas)</h3>
        <table id="test-metrics-classic">
            <thead>
                <tr>
                    <th style="cursor: pointer;" data-sort="model" onclick="sortTable('test-metrics-classic', 0)">Modelo ↕</th>
                    <th>Split</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('test-metrics-classic', 2)">PER (%) ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('test-metrics-classic', 3)">WER (%) ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('test-metrics-classic', 4)">Accuracy (%) ↕</th>
                    <th>Palavras Teste</th>
                    <th>Corretas</th>
                </tr>
            </thead>
            <tbody>
"""
    for model in models:
        eval_data = load_evaluation_results(model.get("evaluation_file"))
        exp_name = model["config"].get("experiment", {}).get("name", model["name"])
        
        # Extrair split
        data_config = model["config"].get("data", {})
        test_ratio = data_config.get("test_ratio", 0.20)
        val_ratio = data_config.get("val_ratio", 0.10)
        train_ratio = 1.0 - test_ratio - val_ratio
        split_display = f"{int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}"
        
        per_str = f"{eval_data['per']:.2f}" if eval_data.get("per") is not None else "—"
        wer_str = f"{eval_data['wer']:.2f}" if eval_data.get("wer") is not None else "—"
        acc_str = f"{eval_data['accuracy']:.2f}" if eval_data.get("accuracy") is not None else "—"
        per_sort = eval_data['per'] if eval_data.get("per") is not None else 999.0
        wer_sort = eval_data['wer'] if eval_data.get("wer") is not None else 999.0
        acc_sort = eval_data['accuracy'] if eval_data.get("accuracy") is not None else -1.0
        tw_str = f"{eval_data['test_words']:,}" if eval_data.get("test_words") else "—"
        cw_str = f"{eval_data['correct_words']:,}" if eval_data.get("correct_words") else "—"
        html += (
            "<tr>"
            f"<td><strong>{exp_name}</strong></td>"
            f"<td>{split_display}</td>"
            f"<td data-value=\"{per_sort}\">{per_str}</td>"
            f"<td data-value=\"{wer_sort}\">{wer_str}</td>"
            f"<td data-value=\"{acc_sort}\">{acc_str}</td>"
            f"<td>{tw_str}</td>"
            f"<td>{cw_str}</td>"
            "</tr>"
        )
    html += """
            </tbody>
        </table>
"""

    # Métricas Graduadas PanPhon — lidas do performance.json
    has_graduated = False
    graduated_entries = []
    if perf_data:
        for entry in perf_data.get("fg2p_models", []):
            gm = entry.get("graduated_metrics")
            if gm:
                has_graduated = True
                graduated_entries.append((entry, gm))

    if has_graduated:
        graduated_entries.sort(key=lambda x: (_extract_experiment_index(x[0].get("name", "")), x[0].get("name", "").lower()))
        html += """
        <h3>Métricas Graduadas (PanPhon)</h3>
        <details style="margin-bottom: 15px; padding: 12px; background: #e7f3ff; border-left: 4px solid #2196F3; border-radius: 4px;">
            <summary style="cursor: pointer; font-weight: 600; color: #2196F3; font-size: 0.95em;">📚 O que são Métricas Graduadas? (clique para expandir)</summary>
            <div style="margin-top: 12px; font-size: 0.9em; line-height: 1.6;">
                <p><strong>Diferença: Clássicas vs Graduadas</strong></p>
                <ul style="margin-left: 20px;">
                    <li><strong>Métricas Clássicas</strong> (PER/WER): Tratam todos os erros igualmente.<br>
                        Ex: <code>ɛ→e</code> (1 feature diferente) = erro completo (1.0) | <code>a→k</code> (vogal→consoante) = erro completo (1.0)</li>
                    <li><strong>Métricas Graduadas</strong> (PanPhon): Ponderam erros pela distância fonológica.<br>
                        Ex: <code>ɛ→e</code> (1 feature) = erro leve (0.2) | <code>a→k</code> (8+ features) = erro grave (1.0)</li>
                </ul>
                <p><strong>PanPhon: 24 Features Articulatórias</strong></p>
                <p>Cada fonema é representado por um vetor de 24 dimensões: altura, recuo, arredondamento, vozeamento, nasalidade, lateral, etc.</p>
                <table style="font-size: 0.85em; margin: 10px 0; border-collapse: collapse;">
                    <tr style="background: #f0f3ff;">
                        <th style="padding: 6px; border: 1px solid #ddd;">Classe</th>
                        <th style="padding: 6px; border: 1px solid #ddd;">Distância</th>
                        <th style="padding: 6px; border: 1px solid #ddd;">Descrição</th>
                        <th style="padding: 6px; border: 1px solid #ddd;">Exemplo</th>
                    </tr>
                    <tr>
                        <td style="padding: 6px; border: 1px solid #ddd;"><strong>A</strong></td>
                        <td style="padding: 6px; border: 1px solid #ddd;">0 features</td>
                        <td style="padding: 6px; border: 1px solid #ddd;">Fonema 100% correto</td>
                        <td style="padding: 6px; border: 1px solid #ddd;"><code>a → a</code></td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 6px; border: 1px solid #ddd;"><strong>B</strong></td>
                        <td style="padding: 6px; border: 1px solid #ddd;">≤1 feature</td>
                        <td style="padding: 6px; border: 1px solid #ddd;">Erro leve (articulação quase idêntica)</td>
                        <td style="padding: 6px; border: 1px solid #ddd;"><code>ɛ↔e</code> (altura), <code>z↔s</code> (vozeamento)</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px; border: 1px solid #ddd;"><strong>C</strong></td>
                        <td style="padding: 6px; border: 1px solid #ddd;">2-3 features</td>
                        <td style="padding: 6px; border: 1px solid #ddd;">Erro médio (mesma família)</td>
                        <td style="padding: 6px; border: 1px solid #ddd;"><code>a↔ə</code> (altura+recuo)</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 6px; border: 1px solid #ddd;"><strong>D</strong></td>
                        <td style="padding: 6px; border: 1px solid #ddd;">≥4 features</td>
                        <td style="padding: 6px; border: 1px solid #ddd;">Erro grave (categorias diferentes)</td>
                        <td style="padding: 6px; border: 1px solid #ddd;"><code>a→k</code> (vogal→consoante)</td>
                    </tr>
                </table>
                <p><strong>Interpretação:</strong> Métricas graduadas revelam se o modelo está fazendo erros <em>articulatoriamente sensatos</em> (Classe B) ou erros aleatórios (Classe D).</p>
            </div>
        </details>
        
        <h4 style="margin-top: 20px; color: #2c3e50;">📊 Distribuição por Fonemas</h4>
        <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
            Avaliação fonética usando 24 features articulatórias — erros ponderados por distância fonológica.
            <br><small style="color: #999;">💡 Clique nos headers para ordenar</small>
        </p>
        <table id="graduated-metrics-phonemes">
            <thead>
                <tr>
                    <th style="cursor: pointer;" data-sort="model" onclick="sortTable('graduated-metrics-phonemes', 0)">Modelo ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('graduated-metrics-phonemes', 1)">PER Weighted (%) ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('graduated-metrics-phonemes', 2)">Classe A (exata) ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('graduated-metrics-phonemes', 3)">Classe B (leve) ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('graduated-metrics-phonemes', 4)">Classe C (média) ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('graduated-metrics-phonemes', 5)">Classe D (grave) ↕</th>
                </tr>
            </thead>
            <tbody>
"""
        for entry, gm in graduated_entries:
            dist = gm.get("error_distribution", {})
            
            # Formatar métricas, tratando valores ausentes
            per_w = gm.get('per_weighted', None)
            per_w_str = f"{per_w:.2f}" if isinstance(per_w, (int, float)) else '—'
            
            class_a = dist.get('class_a_exact', None)
            class_a_str = f"{class_a:.2f}%" if isinstance(class_a, (int, float)) else '—'
            
            class_b = dist.get('class_b_mild', None)
            class_b_str = f"{class_b:.2f}%" if isinstance(class_b, (int, float)) else '—'
            
            class_c = dist.get('class_c_medium', None)
            class_c_str = f"{class_c:.2f}%" if isinstance(class_c, (int, float)) else '—'
            
            class_d = dist.get('class_d_severe', None)
            class_d_str = f"{class_d:.2f}%" if isinstance(class_d, (int, float)) else '—'
            
            # Data attributes para ordenação
            per_w_sort = per_w if isinstance(per_w, (int, float)) else 999
            class_a_sort = class_a if isinstance(class_a, (int, float)) else -1
            class_b_sort = class_b if isinstance(class_b, (int, float)) else -1
            class_c_sort = class_c if isinstance(class_c, (int, float)) else -1
            class_d_sort = class_d if isinstance(class_d, (int, float)) else -1
            
            html += (
                "<tr>"
                f"<td><strong>{entry.get('name', '?')}</strong></td>"
                f"<td data-value=\"{per_w_sort}\">{per_w_str}</td>"
                f"<td class=\"metric-best\" data-value=\"{class_a_sort}\">{class_a_str}</td>"
                f"<td data-value=\"{class_b_sort}\">{class_b_str}</td>"
                f"<td data-value=\"{class_c_sort}\">{class_c_str}</td>"
                f"<td data-value=\"{class_d_sort}\">{class_d_str}</td>"
                "</tr>"
            )
        html += """
            </tbody>
        </table>
        <div style="margin-top: 10px; padding: 12px; background: #d4edda; border-left: 4px solid #28a745; border-radius: 4px; font-size: 0.9em;">
            <strong>Legenda:</strong>
            <strong title="100% correto - todas as features fonéticas idênticas">A</strong> = fonema exato |
            <strong title="Erro leve - apenas 1 feature diferente (ex: altura em ɛ↔e). Articulação muito próxima.">B</strong> = ≤1 feature diferente (ex: ɛ↔e) |
            <strong title="Erro médio - 2-3 features diferentes (ex: altura+recuo em a↔ə). Fonemas relacionados.">C</strong> = 2-3 features |
            <strong title="Erro grave - ≥4 features diferentes. Fonemas articulatoriamente distintos.">D</strong> = ≥4 features (erro grave)
        </div>
        
        <h4 style="margin-top: 30px; color: #2c3e50;">🎯 Distribuição por Palavras</h4>
        <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
            Classificação de palavras completas pela distância média dos fonemas — Classe A = Accuracy.
            <br><small style="color: #999;">💡 Clique nos headers para ordenar</small>
        </p>
        <table id="graduated-metrics-words">
            <thead>
                <tr>
                    <th style="cursor: pointer;" data-sort="model" onclick="sortTable('graduated-metrics-words', 0)">Modelo ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('graduated-metrics-words', 1)">WER Graduated (%) ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('graduated-metrics-words', 2)">Classe A (exata) ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('graduated-metrics-words', 3)">Classe B (leve) ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('graduated-metrics-words', 4)">Classe C (média) ↕</th>
                    <th style="cursor: pointer;" data-sort="number" onclick="sortTable('graduated-metrics-words', 5)">Classe D (grave) ↕</th>
                </tr>
            </thead>
            <tbody>
"""
        for entry, gm in graduated_entries:
            # Para palavras: Classe A = Accuracy
            # Classes B/C/D são palavras com erros, distribuídas pelo resto do WER
            accuracy = entry.get('accuracy', 0.0)
            wer_grad = gm.get('wer_graduated', None)
            
            # Estimativa de distribuição por palavras (proporcional à distribuição de fonemas)
            dist = gm.get("error_distribution", {})
            phoneme_b = dist.get('class_b_mild', None)
            phoneme_c = dist.get('class_c_medium', None)
            phoneme_d = dist.get('class_d_severe', None)
            
            # Calcular se temos dados válidos
            if all(isinstance(v, (int, float)) for v in [phoneme_b, phoneme_c, phoneme_d]):
                # Normalizar as proporções de B/C/D
                total_errors = phoneme_b + phoneme_c + phoneme_d
                if total_errors > 0:
                    # Distribuir o (100 - Accuracy) proporcionalmente
                    error_space = 100.0 - accuracy
                    word_b = (phoneme_b / total_errors) * error_space
                    word_c = (phoneme_c / total_errors) * error_space
                    word_d = (phoneme_d / total_errors) * error_space
                else:
                    word_b = word_c = word_d = 0.0
            else:
                word_b = word_c = word_d = None
            
            # Formatar valores
            wer_grad_str = f"{wer_grad:.2f}" if isinstance(wer_grad, (int, float)) else '—'
            wer_grad_sort = wer_grad if isinstance(wer_grad, (int, float)) else 999
            accuracy_str = f"{accuracy:.2f}%" if isinstance(accuracy, (int, float)) else '—'
            word_b_str = f"{word_b:.2f}%" if isinstance(word_b, (int, float)) else '—'
            word_c_str = f"{word_c:.2f}%" if isinstance(word_c, (int, float)) else '—'
            word_d_str = f"{word_d:.2f}%" if isinstance(word_d, (int, float)) else '—'
            accuracy_sort = accuracy if isinstance(accuracy, (int, float)) else -1
            word_b_sort = word_b if isinstance(word_b, (int, float)) else -1
            word_c_sort = word_c if isinstance(word_c, (int, float)) else -1
            word_d_sort = word_d if isinstance(word_d, (int, float)) else -1
            
            html += (
                "<tr>"
                f"<td><strong>{entry.get('name', '?')}</strong></td>"
                f"<td data-value=\"{wer_grad_sort}\">{wer_grad_str}</td>"
                f"<td class=\"metric-best\" data-value=\"{accuracy_sort}\">{accuracy_str}</td>"
                f"<td data-value=\"{word_b_sort}\">{word_b_str}</td>"
                f"<td data-value=\"{word_c_sort}\">{word_c_str}</td>"
                f"<td data-value=\"{word_d_sort}\">{word_d_str}</td>"
                "</tr>"
            )
        html += """
            </tbody>
        </table>
        <div style="margin-top: 10px; padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; font-size: 0.9em;">
            <strong>Nota:</strong> Classe A (palavra exata) = Accuracy. Classes B/C/D representam palavras com erros, classificadas pela distância fonológica média dos fonemas incorretos. Distribuição estimada proporcionalmente às métricas por fonema.
        </div>
"""

    # --- Seção 3: Análise Detalhada de Erros Fonéticos ---
    html += """
        <h3 style="margin-top: 30px;">🔬 Análise Detalhada de Erros Fonéticos</h3>
        <p style="color: #666; font-size: 0.9em; margin-bottom: 15px;">
            Análise PanPhon com confusões fonêmicas, distribuição de classes e exemplos de palavras.
        </p>
"""
    
    for model in models:
        model_name = model["name"]
        exp_name = model["config"].get("experiment", {}).get("name", model_name)
        
        # Auto-executar analyze_errors se necessário
        predictions_file = model.get("predictions_file")
        run_analyze_errors_if_needed(model_name, predictions_file)
        
        error_data = load_error_analysis(model_name)
        
        if not error_data.get("top_confusions"):
            continue
        
        # Cabeçalho do modelo (colapsável)
        html += f"""
        <div class="error-analysis-section" data-model="{model_name}">
            <div class="error-header" onclick="toggleErrorSection('{model_name}')">
                <h4 style="display: inline;">📊 {exp_name}</h4>
                <span class="toggle-icon" id="toggle_{model_name}">▼</span>
            </div>
            <div class="error-content" id="errors_{model_name}" style="display: none;">
"""
        
        # Top confusões fonêmicas
        if error_data.get("top_confusions"):
            html += """
                <div class="error-subsection">
                    <h5>🔤 Top 10 Confusões Fonêmicas</h5>
                    <table class="confusion-table">
                        <thead>
                            <tr>
                                <th>Referência</th>
                                <th>Predição</th>
                                <th>Ocorrências</th>
                                <th>Distância</th>
                                <th>Classe</th>
                            </tr>
                        </thead>
                        <tbody>
"""
            for conf in error_data["top_confusions"][:10]:
                class_tooltip = {
                    'B': 'Erro leve - 1 feature diferente',
                    'C': 'Erro médio - 2-3 features diferentes',
                    'D': 'Erro grave - ≥4 features diferentes'
                }.get(conf["class"], '')
                class_badge = {
                    'B': f'<span class="badge badge-success" title="{class_tooltip}">B</span>',
                    'C': f'<span class="badge badge-warning" title="{class_tooltip}">C</span>',
                    'D': f'<span class="badge badge-danger" title="{class_tooltip}">D</span>'
                }.get(conf["class"], conf["class"])
                
                html += f"""
                            <tr>
                                <td><code>{conf["ref"]}</code></td>
                                <td><code>{conf["pred"]}</code></td>
                                <td>{conf["count"]}</td>
                                <td>{conf["distance"]:.4f}</td>
                                <td>{class_badge}</td>
                            </tr>
"""
            html += """
                        </tbody>
                    </table>
                </div>
"""
        
        # Distribuição de classes POR FONEMA
        if error_data.get("class_distribution"):
            dist = error_data["class_distribution"]
            props = error_data.get("class_proportions", {})
            
            html += """
                <div class="error-subsection">
                    <h5>📊 Distribuição por FONEMA</h5>
                    <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
                        Cada fonema predito é classificado individualmente pela distância articulatória.
                    </p>
                    <div class="class-distribution">
"""
            
            for cls in ['A', 'B', 'C', 'D']:
                count = dist.get(cls, 0)
                pct = props.get(cls, 0.0)
                bar_width = min(pct, 100)
                bar_class = {'A': 'bar-success', 'B': 'bar-info', 'C': 'bar-warning', 'D': 'bar-danger'}.get(cls, '')
                class_tooltip = {
                    'A': '100% correto - todas as features fonéticas idênticas',
                    'B': 'Erro leve - apenas 1 feature diferente (ex: altura em ɛ↔e). Articulação muito próxima.',
                    'C': 'Erro médio - 2-3 features diferentes (ex: altura+recuo em a↔ə). Fonemas relacionados.',
                    'D': 'Erro grave - ≥4 features diferentes. Fonemas articulatoriamente distintos.'
                }.get(cls, '')
                
                html += f"""
                        <div class="class-bar">
                            <div class="class-label" title="{class_tooltip}">Classe {cls}</div>
                            <div class="progress-bar">
                                <div class="progress-fill {bar_class}" style="width: {bar_width}%;"></div>
                            </div>
                            <div class="class-value">{count:,} fonemas ({pct:.2f}%)</div>
                        </div>
"""
            
            html += """
                    </div>
                </div>
"""
        
        # Distribuição de classes POR PALAVRA
        if error_data.get("word_distribution"):
            word_dist = error_data["word_distribution"]
            word_props = error_data.get("word_proportions", {})
            
            html += """
                <div class="error-subsection">
                    <h5>📝 Distribuição por PALAVRA</h5>
                    <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
                        Cada palavra é classificada pela <strong>pior classe</strong> entre seus fonemas.
                    </p>
                    <div class="class-distribution">
"""
            
            for cls in ['A', 'B', 'C', 'D']:
                count = word_dist.get(cls, 0)
                pct = word_props.get(cls, 0.0)
                bar_width = min(pct, 100)
                bar_class = {'A': 'bar-success', 'B': 'bar-info', 'C': 'bar-warning', 'D': 'bar-danger'}.get(cls, '')
                examples = error_data.get("word_examples", {}).get(cls, [])
                has_examples = len(examples) > 0 and cls != 'A'
                onclick = f"onclick=\"showExamples('{model_name}', '{cls}')\"" if has_examples else ""
                cursor = "cursor: pointer;" if has_examples else ""
                
                class_desc = {
                    'A': 'exata (100% correta)',
                    'B': 'erro leve',
                    'C': 'erro médio',
                    'D': 'erro grave'
                }.get(cls, '')
                class_tooltip = {
                    'A': '100% correto - todas as features fonéticas idênticas',
                    'B': 'Erro leve - apenas 1 feature diferente (ex: altura em ɛ↔e). Articulação muito próxima.',
                    'C': 'Erro médio - 2-3 features diferentes (ex: altura+recuo em a↔ə). Fonemas relacionados.',
                    'D': 'Erro grave - ≥4 features diferentes. Fonemas articulatoriamente distintos.'
                }.get(cls, '')
                
                html += f"""
                        <div class="class-bar" {onclick} style="{cursor}">
                            <div class="class-label" title="{class_tooltip}">Classe {cls} <small>({class_desc})</small></div>
                            <div class="progress-bar">
                                <div class="progress-fill {bar_class}" style="width: {bar_width}%;"></div>
                            </div>
                            <div class="class-value">{count:,} palavras ({pct:.2f}%)</div>
                        </div>
"""
            
            html += """
                    </div>
                </div>
"""
        
        # Seção de Exemplos por Classe (inline, 10 palavras por classe B/C/D)
        word_examples = error_data.get("word_examples", {})
        has_any_examples = any(len(word_examples.get(cls, [])) > 0 for cls in ['B', 'C', 'D'])
        
        if has_any_examples:
            class_colors = {'B': '#17a2b8', 'C': '#ffc107', 'D': '#dc3545'}
            class_labels = {
                'B': 'Leves — ≤1 feature (ex: ɛ↔e, z↔s)',
                'C': 'Médios — 2-3 features (ex: a↔ə)',
                'D': 'Graves — ≥4 features'
            }
            
            html += """
                <div class="error-subsection">
                    <h5>🔍 Exemplos por Classe de Erro</h5>
                    <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
                        Palavras representativas de cada classe, mostrando <strong>por que</strong> foram assim classificadas.
                        Clique na barra de classe acima para ver todos os exemplos no modal.
                    </p>
"""
            
            for cls in ['B', 'C', 'D']:
                examples = word_examples.get(cls, [])
                if not examples:
                    continue
                
                color = class_colors[cls]
                label = class_labels[cls]
                show_count = min(10, len(examples))
                
                html += f"""
                    <details style="margin-bottom: 12px; border-left: 4px solid {color}; padding: 8px 12px; background: #f8f9fa; border-radius: 4px;">
                        <summary style="cursor: pointer; font-weight: 600; color: {color};">
                            Classe {cls}: {label} — {show_count} exemplos de {len(examples)}
                        </summary>
                        <div style="margin-top: 10px; font-family: 'Courier New', monospace; font-size: 0.82em; line-height: 1.7;">
"""
                
                for ex in examples[:10]:
                    pred_phonemes = ex['pred'].replace('/', '').strip()
                    ref_phonemes = ex['ref'].replace('/', '').strip()
                    html += f"""
                            <div style="padding: 6px 8px; margin-bottom: 6px; background: #fff; border-radius: 3px; display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline;">
                                <strong style="color: {color}; min-width: 120px;">{ex['word']}</strong>
                                <span style="color: #666;">score={ex['score']:.3f}</span>
                                <span>pred: <span style="color: #333;">{pred_phonemes}</span></span>
                                <span>ref: <span style="color: #888;">{ref_phonemes}</span></span>
                            </div>
"""
                
                html += """
                        </div>
                    </details>
"""
            
            html += """
                </div>
"""
        
        # Seção de Anomalias Comportamentais
        anomalies = error_data.get("anomalies", {})
        has_any_anomaly = (
            anomalies.get("truncated_count", 0) > 0 or
            anomalies.get("overgenerated_count", 0) > 0 or
            anomalies.get("hallucinations_count", 0) > 0
        )
        
        if has_any_anomaly:
            html += """
                <div class="error-subsection">
                    <h5>🚨 Anomalias Comportamentais</h5>
                    <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
                        Detecção de comportamentos anômalos: truncation, over-generation e alucinações (loops/repetições).
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px;">
"""
            
            # Card de Length Stats
            length_stats = anomalies.get("length_stats", {})
            if length_stats:
                mean_diff = length_stats.get("mean", 0)
                std_diff = length_stats.get("std", 0)
                html += f"""
                        <div style="padding: 12px; background: #f8f9fa; border-left: 4px solid #17a2b8; border-radius: 4px;">
                            <strong>📏 Distribuição de Comprimento</strong>
                            <div style="margin-top: 8px; font-size: 0.85em;">
                                <div>Média: <code>{mean_diff:+.2f}</code> fonemas</div>
                                <div>Desvio: <code>{std_diff:.2f}</code> fonemas</div>
                            </div>
                        </div>
"""
            
            # Card de Truncation
            trunc_count = anomalies.get("truncated_count", 0)
            if trunc_count > 0:
                trunc_examples = anomalies.get("truncated", [])[:3]
                html += f"""
                        <div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;">
                            <strong>⚠️ Truncation</strong>
                            <div style="margin-top: 8px; font-size: 0.85em;">
                                <div><strong>{trunc_count}</strong> palavra(s) truncada(s)</div>
"""
                if trunc_examples:
                    html += """
                                <div style="margin-top: 6px; font-family: monospace; font-size: 0.8em;">
"""
                    for ex in trunc_examples:
                        html += f"""
                                    <div><code>{ex['word']}</code> ref={ex['ref_len']} pred={ex['pred_len']} (Δ{ex['diff']:+d})</div>
"""
                    html += """
                                </div>
"""
                html += """
                            </div>
                        </div>
"""
            
            # Card de Over-generation
            overgen_count = anomalies.get("overgenerated_count", 0)
            if overgen_count > 0:
                html += f"""
                        <div style="padding: 12px; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 4px;">
                            <strong>⚠️ Over-generation</strong>
                            <div style="margin-top: 8px; font-size: 0.85em;">
                                <div><strong>{overgen_count}</strong> palavra(s) sobre-gerada(s)</div>
                            </div>
                        </div>
"""
            
            # Card de Hallucinations
            halluc_count = anomalies.get("hallucinations_count", 0)
            if halluc_count > 0:
                halluc_examples = anomalies.get("hallucinations", [])[:10]
                html += f"""
                        <div style="padding: 12px; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 4px;">
                            <strong>⚠️ Alucinações (Loops)</strong>
                            <div style="margin-top: 8px; font-size: 0.85em;">
                                <div><strong>{halluc_count}</strong> palavra(s) com loops/repetições</div>
"""
                if halluc_examples:
                    html += """
                                <details style="margin-top: 8px; cursor: pointer;">
                                    <summary style="color: #721c24; font-weight: 600;">Ver exemplos</summary>
                                    <div style="margin-top: 8px; padding: 8px; background: #fff; border-radius: 4px; font-family: monospace; font-size: 0.75em;">
"""
                    for ex in halluc_examples:
                        html += f"""
                                        <div style="margin-bottom: 10px; padding: 6px; border-left: 2px solid #dc3545;">
                                            <div><strong>{ex['word']}</strong> <span style="color: #666;">({ex['patterns']})</span></div>
                                            <div style="margin-left: 10px; color: #666;">
                                                <div>pred: {ex['pred']}</div>
                                                <div>ref:  {ex['ref']}</div>
                                            </div>
                                        </div>
"""
                    html += """
                                    </div>
                                </details>
"""
                html += """
                            </div>
                        </div>
"""
            
            html += """
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </div>
"""

    # --- Seção 4: Benchmark com Literatura (sempre visível) ---
    fg2p_rows = render_benchmark_rows(perf_data.get("fg2p_models", [])) if perf_data else ""
    ptbr_rows = render_benchmark_rows(perf_data.get("literature_ptbr", [])) if perf_data else ""
    general_rows = render_benchmark_rows(perf_data.get("literature_general", [])) if perf_data else ""

    if perf_data and perf_data.get("notes"):
        notes_html = "<ul style=\"margin: 0; padding-left: 18px;\">"
        for note in perf_data["notes"]:
            notes_html += f"<li>{note}</li>"
        notes_html += "</ul>"
    else:
        notes_html = (
            "PT-BR tem irregularidades ortográficas diferentes do inglês. "
            "Comparações diretas de métricas devem considerar características linguísticas específicas."
        )

    html += """
    <div class="section">
        <h3>📚 Comparação com Literatura</h3>
        
        <div style="margin-bottom: 20px; padding: 15px; background: #e7f3ff; border-left: 4px solid #2196F3; border-radius: 4px;">
            <strong>🔍 Contexto de Comparabilidade:</strong>
            <p style="margin-top: 8px; font-size: 0.95em; line-height: 1.5;">
                <strong>Modelos FG2P</strong> (≈8.5–17M params): Avaliados em dataset PT-BR estratificado (28.782 palavras teste).<br>
                <strong>Literatura PT-BR</strong> (LatPhon, XphoneBR): Especialização em português; LatPhon usa apenas 500 palavras teste.<br>
                <strong>Literatura Geral</strong> (DeepPhonemizer, ByT5, Phonetisaurus): Principalmente EN/IT; ByT5 é média multilíngue (100+ idiomas).
            </p>
            <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                ⚠️ <strong>Comparações diretas podem ser enganosas:</strong> Dataset, idioma, tamanho do split e complexidade ortográfica afetam PER/WER.
                <br>Ver <a href="LITERATURE.md" style="color: #2196F3;">LITERATURE.md</a> para análise detalhada de metodologias e limitações.
            </p>
        </div>
        <table class="benchmark-table">
            <thead>
                <tr>
                    <th>Sistema</th>
                    <th>PER (%)</th>
                    <th>WER (%)</th>
                    <th>Accuracy (%)</th>
                    <th>Speed (w/s)</th>
                    <th>Notas</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background: #f0f3ff;">
                    <td colspan="6" style="font-weight: bold; text-align: center;">Modelos Avaliados (FG2P)</td>
                </tr>
"""
    html += fg2p_rows if fg2p_rows else '<tr><td colspan="6" style="text-align:center; color:#666;">Sem dados</td></tr>'

    html += """
                <tr style="background: #f0f3ff;">
                    <td colspan="6" style="font-weight: bold; text-align: center;">Literatura PT-BR</td>
                </tr>
"""
    html += ptbr_rows if ptbr_rows else '<tr><td colspan="6" style="text-align:center; color:#666;">Sem dados</td></tr>'

    html += """
                <tr style="background: #f0f3ff;">
                    <td colspan="6" style="font-weight: bold; text-align: center;">Referências Literatura Geral</td>
                </tr>
"""
    html += general_rows if general_rows else '<tr><td colspan="6" style="text-align:center; color:#666;">Sem dados</td></tr>'

    html += f"""
            </tbody>
        </table>

        <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;">
            <strong>⚠️ Nota sobre comparações:</strong> {notes_html}
        </div>
    </div>
"""
    
    # Add inline script to populate error examples data
    html += "\n    <script>\n"
    for model in models:
        model_name = model["name"]
        error_data = load_error_analysis(model_name)
        word_examples = error_data.get("word_examples", {})
        
        for cls in ['B', 'C', 'D']:
            examples = word_examples.get(cls, [])
            if examples:
                import json
                examples_json = json.dumps(examples)
                html += f"        errorExamplesData['{model_name}_{cls}'] = {examples_json};\n"
    
    html += "    </script>\n"

    return html


def generate_javascript() -> str:
    """Gera JavaScript para interatividade (seleção visual dos cards)"""
    return """
    <script>
        let selectedModels = new Set();
        const MAX_SELECTION = 3;

        function toggleModel(modelName) {
            const checkbox = document.getElementById(`select_${modelName}`);
            checkbox.checked = !checkbox.checked;
            updateSelection();
        }

        function updateSelection() {
            selectedModels.clear();
            const checkboxes = document.querySelectorAll('input[type="checkbox"][id^="select_"]');

            checkboxes.forEach(cb => {
                if (cb.checked) {
                    selectedModels.add(cb.id.replace('select_', ''));
                }
            });

            if (selectedModels.size > MAX_SELECTION) {
                const modelName = Array.from(selectedModels).pop();
                document.getElementById(`select_${modelName}`).checked = false;
                selectedModels.delete(modelName);
                alert(`Máximo de ${MAX_SELECTION} modelos.`);
            }

            document.querySelectorAll('.model-card').forEach(card => {
                const cb = card.querySelector('input[type="checkbox"]');
                card.classList.toggle('selected', cb && cb.checked);
            });
            
            // Filter error analysis sections by selected models
            filterErrorSections();

            // Filter plot sections by selected models
            filterPlotSections();
        }
        
        function filterErrorSections() {
            const errorSections = document.querySelectorAll('.error-analysis-section');
            errorSections.forEach(section => {
                const modelName = section.dataset.model;
                const isSelected = selectedModels.has(modelName) || selectedModels.size === 0;
                section.style.display = isSelected ? 'block' : 'none';
            });
        }

        function filterPlotSections() {
            const plotCards = document.querySelectorAll('.plot-card[data-model]');
            plotCards.forEach(card => {
                const modelName = card.dataset.model;
                const isSelected = selectedModels.has(modelName) || selectedModels.size === 0;
                card.style.display = isSelected ? 'block' : 'none';
            });
        }

        function selectAll() {
            const checkboxes = document.querySelectorAll('input[type="checkbox"][id^="select_"]');
            let count = 0;
            checkboxes.forEach(cb => {
                cb.checked = (count < MAX_SELECTION);
                count++;
            });
            updateSelection();
        }

        function clearSelection() {
            document.querySelectorAll('input[type="checkbox"][id^="select_"]').forEach(cb => {
                cb.checked = false;
            });
            updateSelection();
        }
        
        // Error Analysis Functions
        function toggleErrorSection(modelName) {
            const content = document.getElementById(`errors_${modelName}`);
            const icon = document.getElementById(`toggle_${modelName}`);
            
            if (content.style.display === 'none') {
                content.style.display = 'block';
                icon.textContent = '▼';
                icon.classList.remove('rotated');
            } else {
                content.style.display = 'none';
                icon.textContent = '▶';
                icon.classList.add('rotated');
            }
        }
        
        // Table Sorting Function
        function sortTable(tableId, columnIndex) {
            const table = document.getElementById(tableId);
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            // Toggle sort direction
            const header = table.querySelector(`th:nth-child(${columnIndex + 1})`);
            const sortType = header.dataset.sort || 'auto';
            const isAscending = header.textContent.includes('↕')
                ? true
                : header.textContent.includes('↑')
                    ? false
                    : true;

            function parseNumeric(cell) {
                if (cell.dataset.value !== undefined) {
                    const val = parseFloat(cell.dataset.value);
                    return Number.isFinite(val) ? val : Number.POSITIVE_INFINITY;
                }
                const clean = cell.textContent.trim().replace('%', '').replace(',', '.');
                const parsed = parseFloat(clean);
                return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
            }

            function parseModel(cell) {
                const txt = cell.textContent.trim();
                const match = txt.match(/exp\\s*_?(\\d+)/i);
                const idx = match ? parseInt(match[1], 10) : Number.MAX_SAFE_INTEGER;
                return { idx, txt: txt.toLowerCase() };
            }

            rows.sort((a, b) => {
                const aCell = a.cells[columnIndex];
                const bCell = b.cells[columnIndex];

                if (sortType === 'number') {
                    const aVal = parseNumeric(aCell);
                    const bVal = parseNumeric(bCell);
                    return isAscending ? aVal - bVal : bVal - aVal;
                }

                if (sortType === 'model') {
                    const aVal = parseModel(aCell);
                    const bVal = parseModel(bCell);
                    if (aVal.idx !== bVal.idx) {
                        return isAscending ? aVal.idx - bVal.idx : bVal.idx - aVal.idx;
                    }
                    return isAscending
                        ? aVal.txt.localeCompare(bVal.txt, undefined, { numeric: true, sensitivity: 'base' })
                        : bVal.txt.localeCompare(aVal.txt, undefined, { numeric: true, sensitivity: 'base' });
                }

                const aText = aCell.textContent.trim().toLowerCase();
                const bText = bCell.textContent.trim().toLowerCase();
                return isAscending
                    ? aText.localeCompare(bText, undefined, { numeric: true, sensitivity: 'base' })
                    : bText.localeCompare(aText, undefined, { numeric: true, sensitivity: 'base' });
            });
            
            // Update sort indicators
            table.querySelectorAll('th').forEach((th, idx) => {
                if (idx === columnIndex) {
                    // Change indicator
                    if (th.textContent.includes('↕')) {
                        th.textContent = th.textContent.replace('↕', isAscending ? '↑' : '↓');
                    } else if (th.textContent.includes('↑')) {
                        th.textContent = th.textContent.replace('↑', '↓');
                    } else if (th.textContent.includes('↓')) {
                        th.textContent = th.textContent.replace('↓', '↑');
                    }
                } else {
                    // Reset other headers back to neutral indicator if they have one
                    if (th.textContent.includes('↑') || th.textContent.includes('↓')) {
                        th.textContent = th.textContent.replace(/[↑↓]/, '↕');
                    }
                }
            });
            
            // Reorder rows
            rows.forEach(row => tbody.appendChild(row));
        }
        
        // Store word examples data globally
        const errorExamplesData = {};
        
        function analyzeDifferences(pred, ref) {
            const predPhonemes = pred.trim().split(/\\s+/);
            const refPhonemes = ref.trim().split(/\\s+/);
            const changes = [];
            
            const maxLen = Math.max(predPhonemes.length, refPhonemes.length);
            let i = 0, j = 0;
            
            while (i < predPhonemes.length || j < refPhonemes.length) {
                if (i >= predPhonemes.length) {
                    changes.push(`<span style="color: #dc3545;">-${refPhonemes[j]}</span>`);
                    j++;
                } else if (j >= refPhonemes.length) {
                    changes.push(`<span style="color: #28a745;">+${predPhonemes[i]}</span>`);
                    i++;
                } else if (predPhonemes[i] === refPhonemes[j]) {
                    i++; j++;
                } else {
                    changes.push(`<span style="color: #ffc107;">${refPhonemes[j]}→${predPhonemes[i]}</span>`);
                    i++; j++;
                }
            }
            
            return changes.join(' ');
        }
        
        function showExamples(modelName, errorClass) {
            const key = `${modelName}_${errorClass}`;
            const examples = errorExamplesData[key] || [];
            
            if (examples.length === 0) {
                alert('Nenhum exemplo disponível para esta classe.');
                return;
            }
            
            const modal = document.getElementById('examplesModal');
            const title = document.getElementById('modalTitle');
            const content = document.getElementById('modalExamples');
            
            const classLabels = {
                'B': 'Leves (≤1 feature diferente)',
                'C': 'Médios (2-3 features)',
                'D': 'Graves (≥4 features)'
            };
            
            const classDescriptions = {
                'B': 'Erros de 1 feature fonética (ex: ɛ↔e altura, z↔s vozeamento). Articulação muito próxima.',
                'C': 'Erros de 2-3 features (ex: a↔ə altura+recuo). Mesma família fonética.',
                'D': 'Erros ≥4 features. Categorias fonéticas diferentes.'
            };
            
            title.innerHTML = `Exemplos de Erros Classe ${errorClass} — ${classLabels[errorClass]}<br>
                <small style="font-weight: normal; font-size: 0.85em; color: #666;">${classDescriptions[errorClass]}</small>`;
            
            let html = '<div style="max-height: 60vh; overflow-y: auto;">';
            html += `<p style="margin-bottom: 15px; color: #666;"><strong>${examples.length}</strong> palavras encontradas. Mostrando até 30 exemplos.</p>`;
            
            examples.slice(0, 30).forEach(ex => {
                const diffs = analyzeDifferences(ex.pred, ex.ref);
                html += `
                    <div class="example-item" style="border-left: 3px solid #667eea; padding: 12px; margin-bottom: 12px; background: #f8f9fa;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span class="example-word" style="font-size: 1.1em; font-weight: bold; color: #667eea;">${ex.word}</span>
                            <span class="example-score" style="color: #666;">Score: ${ex.score.toFixed(3)}</span>
                        </div>
                        <div style="font-family: 'Courier New', monospace; font-size: 0.9em; line-height: 1.8;">
                            <div><strong>Predito:</strong> <span class="example-pred">${ex.pred}</span></div>
                            <div><strong>Esperado:</strong> <span class="example-ref">${ex.ref}</span></div>
                            <div><strong>Mudanças:</strong> ${diffs}</div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            content.innerHTML = html;
            modal.style.display = 'block';
        }
        
        function closeModal() {
            document.getElementById('examplesModal').style.display = 'none';
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('examplesModal');
            if (event.target === modal) {
                closeModal();
            }
        };
    </script>
"""


def generate_phonological_analysis_html() -> str:
    """Gera seção de análise fonológica: neutralização vocálica e homógrafos heterófonos"""

    section_html = """
    <div class="section">
        <h2>🔬 Análise Fonológica: Limites Fundamentais do G2P</h2>
        <p>Nem todas as limitações de desempenho são falhas do modelo. Algumas refletem <strong>restrições linguísticas</strong> do paradigma word-isolation G2P.</p>
"""

    # ========== Neutralização Vocálica (e ↔ ɛ) ==========
    try:
        import json
        with open("results/vowel_analysis.json", encoding="utf-8") as f:
            vowel_data = json.load(f)

        stats = vowel_data.get("key_statistics", {})
        total_e = stats.get("total_e", 0)
        total_ε = stats.get("total_epsilon", 0)

        errors = stats.get("model_error_context", {})
        exp104b_e_to_ε = errors.get("exp104b_e_to_epsilon_errors", 0)
        exp104b_ε_to_e = errors.get("exp104b_epsilon_to_e_errors", 0)
        total_errors = errors.get("exp104b_total_errors", 0)
        error_pct = errors.get("percentage_of_all_errors", "?")

        section_html += f"""
        <h3>📝 Neutralização Vocálica: /e/ ↔ /ɛ/ em PT-BR</h3>

        <div class="subsection">
            <h4>O Fenômeno</h4>
            <p>
                Em PT-BR, as vogais médias anteriores <strong>/e/</strong> (fechada, ex: pé) e <strong>/ɛ/</strong> (aberta, ex: pélo)
                são fonemas distintos em <strong>sílabas tônicas</strong> mas <strong>neutralizam-se em átonas</strong>
                (ex: "teste", "mel-e-na" onde "-e-" é sempre /ə/).
            </p>
            <p>
                <strong>Imbalance no corpus:</strong> A distribuição é de {total_e:,} ocorrências de /e/ para {total_ε:,} de /ɛ/,
                uma proporção de <strong>7:1</strong> — isto treina o modelo com forte viés para /e/.
            </p>
        </div>

        <div class="subsection">
            <h4>Impacto no Exp104b</h4>
            <table class="analysis-table">
                <tr>
                    <th>Transição</th>
                    <th>Erros</th>
                    <th>Percentual</th>
                </tr>
                <tr>
                    <td>e → ɛ (errar a abertura)</td>
                    <td>{exp104b_e_to_ε}</td>
                    <td>{exp104b_e_to_ε / total_errors * 100:.1f}% dos erros</td>
                </tr>
                <tr>
                    <td>ɛ → e (errar o fechamento)</td>
                    <td>{exp104b_ε_to_e}</td>
                    <td>{exp104b_ε_to_e / total_errors * 100:.1f}% dos erros</td>
                </tr>
                <tr style="background-color: #fff3cd;">
                    <td><strong>Total e ↔ ɛ</strong></td>
                    <td><strong>{total_errors}</strong></td>
                    <td><strong>{error_pct} de TODOS os erros</strong></td>
                </tr>
            </table>
            <p style="font-style: italic; color: #666;">
                ✓ Este é o <strong>erro #1</strong> do modelo Exp104b. Não é uma deficiência — é um reflexo direto do corpus.
            </p>
        </div>
"""
    except Exception as e:
        section_html += f"""
        <h3>📝 Neutralização Vocálica: /e/ ↔ /ɛ/</h3>
        <p style="color: #d9534f;">Análise indisponível: {str(e)}</p>
"""

    # ========== Homógrafos Heterófonos ==========
    try:
        with open("results/homograph_analysis.json", encoding="utf-8") as f:
            homograph_data = json.load(f)

        total_pairs = homograph_data.get("total_pairs", 0)
        pairs_in_corpus = homograph_data.get("pairs_in_corpus", 0)
        pairs_missing = homograph_data.get("pairs_missing", 0)

        # Selecionar pares representativos para mostrar na tabela
        pairs = homograph_data.get("pairs", [])
        representative_pairs = [p for p in pairs if p.get("in_corpus")][:5]
        missing_pairs = [p for p in pairs if not p.get("in_corpus")][:5]

        section_html += f"""
        <h3>🔤 Homógrafos Heterófonos: Ambiguidade Léxico-Sintática</h3>

        <div class="subsection">
            <h4>O Fenômeno</h4>
            <p>
                Em PT-BR, diversas palavras têm a <strong>mesma grafia mas pronuncias diferentes</strong> dependendo da
                categoria gramatical ou flexão. Exemplos clássicos:
            </p>
            <ul>
                <li><strong>jogo</strong> (substantivo): /ˈʒ<u>ɔ</u>gʊ/ vs. /ˈʒ<u>o</u>gʊ/ (1ª p. presente do verbo)</li>
                <li><strong>gosto</strong> (substantivo): /ˈg<u>ɔ</u>ʃtʊ/ vs. /ˈg<u>o</u>ʃtʊ/ (1ª p. verbo)</li>
                <li><strong>seco</strong> (adjetivo): /ˈs<u>ɛ</u>kʊ/ vs. /ˈs<u>e</u>kʊ/ (1ª p. verbo)</li>
            </ul>
            <p>
                <strong>Porque G2P não resolve isso:</strong> O modelo processa a <strong>palavra isolada</strong> —
                sem contexto morfossintático, é impossível saber qual forma usar. Isso exigiria
                um pipeline <strong>NLP → G2P</strong> (POS tagging + análise morfológica antes da fonologia).
            </p>
        </div>

        <div class="subsection">
            <h4>Análise do Corpus PT-BR</h4>
            <p>
                De <strong>{total_pairs} pares clássicos</strong>, apenas <strong>{pairs_in_corpus}</strong>
                ({pairs_in_corpus/total_pairs*100:.0f}%) estão representados no corpus.
                <strong>{pairs_missing}</strong> pares ({pairs_missing/total_pairs*100:.0f}%) estão completamente ausentes.
            </p>
            <p style="color: #d9534f;">
                ⚠️ Impacto: Para cada par ausente, o modelo NUNCA aprenderá ambas as formas.
            </p>
"""

        # Tabela de pares presentes
        if representative_pairs:
            section_html += """
            <h4>Exemplos de Pares Presentes no Corpus</h4>
            <table class="analysis-table">
                <tr>
                    <th>Palavra</th>
                    <th>Forma A</th>
                    <th>Forma B</th>
                    <th>Corpus Armazena</th>
                </tr>
"""
            for pair in representative_pairs:
                word = pair.get("word", "?")
                form_a = pair.get("form_A", {})
                form_b = pair.get("form_B", {})
                chosen = pair.get("chosen_form", "?")
                a_desc = form_a.get("description", "")
                b_desc = form_b.get("description", "")
                a_ipa = form_a.get("ipa", "")
                b_ipa = form_b.get("ipa", "")

                missing = "ausente" if pair.get("missing_form") in [a_desc, b_desc] else ""
                section_html += f"""
                <tr>
                    <td><strong>{word}</strong></td>
                    <td>{a_desc}<br/><code style="font-size: 0.85em;">{a_ipa}</code></td>
                    <td>{b_desc}<br/><code style="font-size: 0.85em;">{b_ipa}</code></td>
                    <td>{chosen} {f'<span style=\"color:#d9534f;\">[{missing}]</span>' if missing else ''}</td>
                </tr>
"""
            section_html += """
            </table>
"""

        # Pares ausentes
        if missing_pairs:
            section_html += f"""
            <h4>Exemplos de Pares Completamente Ausentes ({pairs_missing} total)</h4>
            <p style="color: #d9534f;">
                Estes pares <strong>não estão no corpus</strong> — o modelo não pode aprender ambas as formas:
            </p>
            <ul>
"""
            for pair in missing_pairs:
                word = pair.get("word", "?")
                section_html += f"                <li><strong>{word}</strong></li>\n"
            section_html += "            </ul>\n"

        section_html += """
        </div>

        <div class="subsection" style="background-color: #f0f8ff; padding: 15px; border-left: 4px solid #0066cc;">
            <h4>💡 Conclusão</h4>
            <p>
                Homógrafos heterófonos representam um <strong>limite fundamental</strong> do paradigma G2P word-isolation.
                Não é falha do modelo — é <strong>restrição arquitetural</strong>.
            </p>
            <p>
                <strong>Solução futura:</strong> Pipeline <strong>NLP → G2P</strong> onde:
            </p>
            <ol>
                <li>POS tagger determina a categoria gramatical (noun/verb/adj)</li>
                <li>Tokenizador identifica flexões e compostos (hífen)</li>
                <li>G2P prediz fonologia <strong>com contexto morfossintático</strong></li>
            </ol>
        </div>
"""
    except Exception as e:
        section_html += f"""
        <h3>🔤 Homógrafos Heterófonos</h3>
        <p style="color: #d9534f;">Análise indisponível: {str(e)}</p>
"""

    section_html += """
    </div>
"""

    return section_html


def generate_full_html_report(models: list[dict], output_path: Path):
    """Gera relatório HTML completo"""
    html = generate_html_header()
    
    html += """
    <div class="container">
        <header>
            <h1>🎯 FG2P - Relatório de Análise</h1>
            <p>Sistema de Conversão Grapheme-to-Phoneme para Português Brasileiro</p>
            <p>BiLSTM Encoder-Decoder com Atenção Bahdanau</p>
        </header>
        
        <div class="content">
"""
    
    # Model list section
    html += generate_model_list_html(models)

    # Training/validation plots section
    html += generate_training_plots_html(models)
    
    # Comparison section
    html += generate_comparison_html(models)
    
    # Dataset section (extract split configurations from experiments)
    split_configs = extract_split_configurations(models)
    html += generate_dataset_section_html(split_configs)

    # Phonological analysis section
    html += generate_phonological_analysis_html()

    html += generate_javascript()
    html += """
    </div>
"""
    html += generate_html_footer()
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    logger.info(f"✓ Relatório HTML gerado: {output_path}")


# ============================================================================
def list_models_cli():
    """Lista modelos disponíveis no terminal"""
    models = list_available_models()
    
    if not models:
        logger.error("Nenhum modelo encontrado em models/")
        return
    
    logger.info("=" * 80)
    logger.info(f"{'MODELOS DISPONÍVEIS':<80}")
    logger.info("=" * 80)
    
    for i, model in enumerate(models, 1):
        status = "✓ Completo" if model["completed"] else "⏳ Treinando"
        config = model['config']
        description = config.get("description", "Sem descrição")
        
        logger.info(f"\n[{i}] {model['name']}")
        logger.info(f"    {description}")
        logger.info(f"    Status: {status}")
        logger.info(f"    Épocas: {model['current_epoch']}/{model['total_epochs']}")
        if model['best_loss']:
            logger.info(f"    Best Loss: {model['best_loss']:.6f}")
        
        emb_dim = config.get('model', {}).get('emb_dim', config.get('embedding_dim', '?'))
        hidden_dim = config.get('model', {}).get('hidden_dim', config.get('hidden_dim', '?'))
        embedding_type = config.get('model', {}).get('embedding_type', config.get('embedding_type', 'learned'))
        logger.info(f"    Config: {embedding_type} embedding, emb={emb_dim}, hidden={hidden_dim}")
        
        has_artifacts = []
        if model['history_csv']:
            has_artifacts.append("history.csv")
        if model['evaluation_file']:
            has_artifacts.append("evaluation.txt")
        if model['predictions_file']:
            has_artifacts.append("predictions.tsv")
        
        if has_artifacts:
            logger.info(f"    Artefatos: {', '.join(has_artifacts)}")
    
    logger.info("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Gera relatórios HTML interativos para análise de modelos G2P",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  LISTAR MODELOS:
    python src/reporting/report_generator.py --list
        Lista todos os modelos disponíveis
  
  GERAR RELATÓRIO:
    python src/reporting/report_generator.py
        Gera relatório HTML com todos os modelos
    
    python src/reporting/report_generator.py --output custom_report.html
        Gera relatório com nome personalizado
  
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista modelos disponíveis e sai"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="model_report.html",
        help="Nome do arquivo HTML de saída (padrão: model_report.html)"
    )
    
    args = parser.parse_args()
    
    # Handle CLI commands
    if args.list:
        list_models_cli()
        return
    
    # Default: Generate HTML report
    logger.info("Carregando modelos...")
    models = list_available_models()
    
    if not models:
        logger.error("Nenhum modelo encontrado em models/")
        logger.info("Execute: python src/train.py primeiro")
        return
    
    logger.info(f"Encontrados {len(models)} modelo(s)")
    
    # Generate HTML report
    output_path = ROOT_DIR / args.output
    logger.info("Gerando relatório HTML...")
    generate_full_html_report(models, output_path)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ Relatório gerado com sucesso!")
    logger.info(f"  Arquivo: {output_path}")
    logger.info("  Abra no navegador para visualizar")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
