#!/usr/bin/env python
"""
Análise unificada de treinamento e avaliação

Carrega artefatos de experimentos (CSVs, metadata, predictions) e gera:
- Gráficos de convergência e analise
- Métricas calculadas (throughput, overfitting, convergência)
- Arquivo JSON centralizado com resultados

Uso:
    python src/analysis.py                          # Analisa modelo mais recente
    python src/analysis.py --index 0                # Analisa modelo com índice 0
    python src/analysis.py --model-name EXP2        # Analisa modelo específico
    python src/analysis.py --list                   # Lista todos os modelos
    python src/analysis.py --monitor --model-name EXP0  # Status ao vivo
    python src/analysis.py --test --model-name EXP0     # Carrega métricas de teste
    python src/analysis.py --compare                    # Compara todos os experimentos
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_logger, RESULTS_DIR, MODELS_DIR
from file_registry import FileRegistry, get_base_name_from_path

logger = get_logger("analysis")


# ============================================================================
# Loading & Parsing
# ============================================================================

def load_metadata(model_name: str) -> Optional[dict]:
    """Carrega _metadata.json de um modelo"""
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_history_csv(csv_path: Path) -> list[dict]:
    """Carrega historico de treino do CSV
    
    Suporta ambos formatos:
    - Novo: epoch,train_loss,val_loss,epoch_start_ts,epoch_end_ts
    - Antigo: epoch,train_loss,val_loss
    """
    if not csv_path.exists():
        return []
    
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except (csv.Error, OSError) as e:
        logger.warning(f"Erro ao ler {csv_path.name}: {e}")
        return []
    
    return rows


def load_evaluation(eval_path: Path) -> Optional[dict]:
    """Carrega métricas de avaliação do arquivo .txt"""
    if not eval_path.exists():
        return None
    
    metrics = {}
    try:
        with open(eval_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extractors (padroes usados em inference.py)
            per_match = re.search(r'PER\s*:\s*([\d.]+)\s*%', content)
            wer_match = re.search(r'WER\s*:\s*([\d.]+)\s*%', content)
            acc_match = re.search(r'(?:Accuracy|Exact)\s*:\s*([\d.]+)\s*%', content)
            
            if per_match:
                metrics['per'] = float(per_match.group(1))
            if wer_match:
                metrics['wer'] = float(wer_match.group(1))
            if acc_match:
                metrics['accuracy'] = float(acc_match.group(1))
    except OSError:
        pass
    
    return metrics if metrics else None


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_training_metrics(rows: list[dict]) -> dict:
    """Calcula metricas de treinamento a partir do CSV historico"""
    if not rows:
        return {}
    
    metrics = {
        'num_epochs': len(rows),
        'throughput': None,
    }
    
    # Converter para numeros
    for row in rows:
        row['epoch'] = int(row['epoch'])
        row['train_loss'] = float(row['train_loss'])
        row['val_loss'] = float(row['val_loss'])
        if 'epoch_start_ts' in row and row['epoch_start_ts']:
            row['epoch_start_ts'] = float(row['epoch_start_ts'])
        if 'epoch_end_ts' in row and row['epoch_end_ts']:
            row['epoch_end_ts'] = float(row['epoch_end_ts'])
    
    # Best epoch
    best_idx = min(range(len(rows)), key=lambda i: rows[i]['val_loss'])
    best_row = rows[best_idx]
    metrics['best_epoch'] = best_row['epoch']
    metrics['best_val_loss'] = best_row['val_loss']
    metrics['best_train_loss'] = best_row['train_loss']
    
    # Initial losses
    metrics['initial_train_loss'] = rows[0]['train_loss']
    metrics['initial_val_loss'] = rows[0]['val_loss']
    
    # Final losses
    metrics['final_train_loss'] = rows[-1]['train_loss']
    metrics['final_val_loss'] = rows[-1]['val_loss']
    
    # Convergence (95% melhoria)
    initial_loss = rows[0]['val_loss']
    best_loss = best_row['val_loss']
    threshold_95 = initial_loss - 0.95 * (initial_loss - best_loss)
    
    convergence_epoch = None
    for row in rows:
        if row['val_loss'] <= threshold_95:
            convergence_epoch = row['epoch']
            break
    metrics['convergence_epoch_95percent'] = convergence_epoch
    
    # Gaps (overfitting detection)
    gap_at_best = best_row['val_loss'] - best_row['train_loss']
    gap_at_final = rows[-1]['val_loss'] - rows[-1]['train_loss']
    
    metrics['gap_at_best'] = gap_at_best
    metrics['gap_at_final'] = gap_at_final
    
    # Overfitting flag
    if gap_at_final > 0.05:
        metrics['overfitting'] = '⚠️ Leve'
    elif gap_at_final > 0.1:
        metrics['overfitting'] = '⚠️ Severo'
    else:
        metrics['overfitting'] = '[OK] Nenhum'
    
    # Last 5 epochs std (plateau detection)
    last5_losses = [r['val_loss'] for r in rows[-5:]]
    metrics['last5_std'] = (max(last5_losses) - min(last5_losses)) / 2 if len(last5_losses) > 1 else 0
    
    # Throughput (se timestamps disponiveis)
    if all('epoch_start_ts' in row and 'epoch_end_ts' in row for row in rows):
        total_time = rows[-1]['epoch_end_ts'] - rows[0]['epoch_start_ts']
        # 67155 train samples (do log)
        train_samples = 67155
        throughput_samples_per_sec = (train_samples * len(rows)) / total_time
        
        metrics['throughput'] = {
            'avg_samples_per_sec': throughput_samples_per_sec,
            'min_samples_per_sec': throughput_samples_per_sec,  # Interpolated
            'max_samples_per_sec': throughput_samples_per_sec,
        }
    
    return metrics


# ============================================================================
# Model Selection & Listing
# ============================================================================

def list_available_models() -> list[dict]:
    """Lista todos os modelos disponiveis com metadados"""
    models_info = []
    
    model_files = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    
    if not model_files:
        print("\n[X] Nenhum modelo encontrado em models/")
        print("Execute train.py primeiro para treinar um modelo.\n")
        return []
    
    sys.stdout.reconfigure(encoding='utf-8')
    print("\n" + "=" * 80)
    print("MODELOS DISPONÍVEIS")
    print("=" * 80)
    
    for idx, model_file in enumerate(model_files):
        model_name = model_file.stem
        metadata = load_metadata(model_name)
        
        print(f"\n[{idx}] {model_file.name}")
        
        if metadata:
            print(f"    Experimento: {metadata.get('experiment_name', 'N/A')}")
            print(f"    Timestamp: {metadata.get('timestamp', 'N/A')}")
            status = "[OK] Completo" if metadata.get('training_completed') else "[!] Incompleto"
            print(f"    Status: Epoch {metadata.get('current_epoch', '?')}/{metadata.get('total_epochs', '?')} {status}")
            print(f"    Melhor loss: {metadata.get('best_loss', 'N/A')}")
        else:
            print("    ⚠ Metadados nao disponiveis")
        
        models_info.append({
            'index': idx,
            'name': model_name,
            'path': model_file,
            'metadata': metadata,
        })
    
    print("\n" + "=" * 80)
    print(f"Total: {len(model_files)} modelo(s)")
    print("=" * 80)
    
    return models_info


def select_model(args, models_info: Optional[list] = None) -> Optional[str]:
    """Seleciona modelo baseado em argumentos (retorna nome do modelo)"""
    if models_info is None:
        models_info = list_available_models()
    
    if not models_info:
        return None
    
    # Índice especificado
    if args.index is not None:
        if 0 <= args.index < len(models_info):
            model_name = models_info[args.index]['name']
            logger.info(f"Usando modelo [índice {args.index}]: {model_name}")
            return model_name
        else:
            logger.error(f"Índice {args.index} invalido (0-{len(models_info)-1})")
            return None
    
    # Nome especificado
    if args.model_name:
        # Busca com padrao (substring match)
        matching = [m for m in models_info if args.model_name in m['name']]
        if matching:
            selected = matching[0]
            logger.info(f"Encontrado: {selected['name']}")
            return selected['name']
        else:
            logger.error(f"Modelo com padrao '{args.model_name}' não encontrado")
            return None
    
    # Padrao: mais recente
    latest = models_info[-1]
    logger.info(f"Usando modelo mais recente: {latest['name']}")
    return latest['name']


# ============================================================================
# Plotting Functions
# ============================================================================

def generate_plots(model_name: str, rows: list[dict], registry: FileRegistry):
    """Gera gráficos de convergência e analise"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        from matplotlib.ticker import MultipleLocator, LogLocator, FuncFormatter
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        logger.warning("matplotlib não disponível. Pulando gráficos.")
        return
    
    if not rows:
        logger.warning("Nenhum historico para plotar")
        return
    
    epochs = [int(r['epoch']) for r in rows]
    train_losses = [float(r['train_loss']) for r in rows]
    val_losses = [float(r['val_loss']) for r in rows]
    
    # Gráfico 1: Convergência
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', alpha=0.7)
    
    # Marcar melhor epoca
    best_idx = min(range(len(rows)), key=lambda i: val_losses[i])
    ax.plot(epochs[best_idx], val_losses[best_idx], 'g*', markersize=15, label=f'Best (epoch {epochs[best_idx]})')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_yscale('log')
    ax.set_title(f'{model_name} - Convergencia')
    ax.legend()
    
    # Configurar eixo X: ticks de 10 em 10
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    
    # Configurar eixo Y: multiplos valores em log scale
    # subs=[1,2,5] adiciona marcacoes em 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, etc
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=30))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    
    # Formatar numeros do eixo Y para serem mais legiveis
    def log_formatter(x, pos):
        if x >= 1:
            return f'{x:.0f}'
        elif x >= 0.01:
            return f'{x:.2f}'
        else:
            return f'{x:.3f}'
    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
    
    ax.grid(True, alpha=0.3, which='both')
    
    conv_path = registry.get_analysis_convergence_plot_path()
    plt.savefig(conv_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico salvo: {conv_path.name}")


# ============================================================================
# Results Output
# ============================================================================

def save_results(model_name: str, config: dict, training_metrics: dict, test_metrics: Optional[dict],
                metadata: Optional[dict], registry: FileRegistry):
    """Salva resultados em JSON centralizado"""
    results = {
        'experiment': model_name,
        'timestamp': datetime.now().isoformat(),
        'training': training_metrics,
        'test_metrics': test_metrics or {}
    }
    
    results_path = registry.get_analysis_results_path()
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultados salvos: {results_path}")


def print_summary(model_name: str, training_metrics: dict, test_metrics: Optional[dict]):
    """Imprime resumo de metricas"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Experiment: {model_name}")
    logger.info(f"Epochs trained: {training_metrics.get('num_epochs', '?')}")
    logger.info("")
    
    logger.info("CONVERGENCE:")
    logger.info(f"  Best epoch: {training_metrics.get('best_epoch', '?')} " +
               f"(val_loss: {training_metrics.get('best_val_loss', '?'):.6f})")
    logger.info(f"  Convergence (95%): epoch {training_metrics.get('convergence_epoch_95percent', '?')}")
    logger.info("")
    
    logger.info("LOSSES:")
    initial_train = training_metrics.get('initial_train_loss', None)
    initial_val = training_metrics.get('initial_val_loss', None)
    best_train = training_metrics.get('best_train_loss', None)
    best_val = training_metrics.get('best_val_loss', None)
    final_train = training_metrics.get('final_train_loss', None)
    final_val = training_metrics.get('final_val_loss', None)
    
    initial_str = f"{initial_train:.6f} / {initial_val:.6f}" if (initial_train and initial_val) else "? / ?"
    best_str = f"{best_train:.6f} / {best_val:.6f}" if (best_train and best_val) else "? / ?"
    final_str = f"{final_train:.6f} / {final_val:.6f}" if (final_train and final_val) else "? / ?"
    
    logger.info(f"  Initial train/val: {initial_str}")
    logger.info(f"  Best train/val:    {best_str}")
    logger.info(f"  Final train/val:   {final_str}")
    logger.info("")
    
    logger.info("GENERALIZATION:")
    logger.info(f"  Gap at best: {training_metrics.get('gap_at_best', '?'):.6f}")
    logger.info(f"  Gap at final: {training_metrics.get('gap_at_final', '?'):.6f}")
    logger.info(f"  Overfitting: {training_metrics.get('overfitting', '?')}")
    logger.info("")
    
    if training_metrics.get('throughput'):
        tput = training_metrics['throughput']
        logger.info("THROUGHPUT:")
        logger.info(f"  Avg: {tput['avg_samples_per_sec']:.0f} samples/s")
        logger.info(f"  Min: {tput['min_samples_per_sec']:.0f} samples/s")
        logger.info(f"  Max: {tput['max_samples_per_sec']:.0f} samples/s")
        logger.info("")
    
    if test_metrics:
        logger.info("TEST METRICS:")
        for key, val in test_metrics.items():
            logger.info(f"  {key}: {val}")
        logger.info("")
    else:
        logger.warning("Nenhuma métrica de teste encontrada")
        logger.info("")
    
    logger.info("=" * 80)


# ============================================================================
# Main Modes
# ============================================================================

def mode_all(models_info: list):
    """Processa todos os modelos disponiveis"""
    logger.info("="*80)
    logger.info(f"PROCESSANDO TODOS OS MODELOS ({len(models_info)} encontrados)")
    logger.info("="*80)
    logger.info("")
    
    success_count = 0
    error_count = 0
    
    for idx, model_info in enumerate(models_info, 1):
        model_name = model_info['name']
        logger.info(f"[{idx}/{len(models_info)}] Processando: {model_name}")
        logger.info("-" * 80)
        
        try:
            # Carregar config
            metadata = load_metadata(model_name)
            if not metadata or 'config' not in metadata:
                logger.error(f"  [X] Config nao encontrada")
                error_count += 1
                continue
            
            config = metadata['config']
            model_timestamp = model_name.split('__')[-1] if '__' in model_name else None
            registry = FileRegistry(config, timestamp=model_timestamp)
            
            # Carregar historico
            history_path = registry.get_history_path()
            csv_rows = load_history_csv(history_path) if history_path.exists() else []
            
            if not csv_rows:
                logger.warning(f"  ⚠ Histórico não encontrado ou vazio")
                error_count += 1
                continue
            
            # Calcular metricas
            training_metrics = calculate_training_metrics(csv_rows)
            
            # Carregar metricas de teste (se disponivel)
            eval_path = registry.get_evaluation_path()
            test_metrics = load_evaluation(eval_path) if eval_path.exists() else None
            
            # Gerar gráficos
            generate_plots(model_name, csv_rows, registry)
            
            # Salvar resultados
            save_results(model_name, config, training_metrics, test_metrics, metadata, registry)
            
            logger.info(f"  [OK] Processado com sucesso")
            success_count += 1
            
        except Exception as e:
            logger.error(f"  [X] Erro ao processar: {e}")
            error_count += 1
        
        logger.info("")
    
    # Resumo final
    logger.info("="*80)
    logger.info("RESUMO DO PROCESSAMENTO")
    logger.info("="*80)
    logger.info(f"  Total: {len(models_info)}")
    logger.info(f"  [OK] Sucesso: {success_count}")
    logger.info(f"  [X] Erros: {error_count}")
    logger.info("")


def mode_default(args, models_info: list):
    """Modo padrao: analise completa de um modelo"""
    model_name = select_model(args, models_info)
    if not model_name:
        return
    
    # Carregar config (base_name para FileRegistry)
    metadata = load_metadata(model_name)
    if not metadata or 'config' not in metadata:
        logger.error(f"Config nao encontrada para {model_name}")
        return
    
    config = metadata['config']
    
    # Extrair timestamp do nome do modelo (formato: {experiment}__{timestamp})
    # Exemplo: exp0_baseline_70split__20260218_044620 → 20260218_044620
    model_timestamp = model_name.split('__')[-1] if '__' in model_name else None
    
    # Criar FileRegistry com timestamp original do treino
    registry = FileRegistry(config, timestamp=model_timestamp)
    
    # Carregar historico
    history_path = registry.get_history_path()
    csv_rows = load_history_csv(history_path) if history_path.exists() else []
    
    if not csv_rows:
        logger.warning(f"Histórico não encontrado ou vazio")
        return
    
    # Calcular metricas de treino
    training_metrics = calculate_training_metrics(csv_rows)
    
    # Carregar metricas de teste (se disponivel)
    eval_path = registry.get_evaluation_path()
    test_metrics = load_evaluation(eval_path) if eval_path.exists() else None
    
    # Gerar gráficos
    generate_plots(model_name, csv_rows, registry)
    
    # Salvar resultados
    save_results(model_name, config, training_metrics, test_metrics, metadata, registry)
    
    # Imprimir resumo
    print_summary(model_name, training_metrics, test_metrics)


def mode_monitor(args, models_info: list):
    """Modo monitor: status ao vivo do treinamento"""
    model_name = select_model(args, models_info)
    if not model_name:
        return
    
    metadata = load_metadata(model_name)
    if not metadata:
        logger.error(f"Metadados não encontrados para {model_name}")
        return
    
    logger.info("")
    logger.info(f"Model: {model_name}")
    logger.info(f"Progress: {metadata.get('current_epoch', '?')}/{metadata.get('total_epochs', '?')} epochs")
    logger.info(f"Status: {'[OK] Training completed' if metadata.get('training_completed') else '[!] Training in progress'}")
    logger.info(f"Best val loss: {metadata.get('best_loss', 'N/A')}")
    logger.info(f"Warmup: until epoch {metadata.get('warmup_until', 'N/A')}")
    logger.info(f"Early stopping patience: {metadata.get('early_stopping_patience', 'N/A')}")
    
    # Split info
    if 'dataset' in metadata:
        ds = metadata['dataset']
        logger.info(f"Split: {ds.get('train_size', '?')} / {ds.get('val_size', '?')} / {ds.get('test_size', '?')}")
    
    logger.info("")


def mode_test(args, models_info: list):
    """Modo test: carrega metricas de avaliação"""
    model_name = select_model(args, models_info)
    if not model_name:
        return
    
    metadata = load_metadata(model_name)
    if not metadata or 'config' not in metadata:
        logger.error(f"Config nao encontrada para {model_name}")
        return
    
    config = metadata['config']
    
    # Extrair timestamp do nome do modelo
    model_timestamp = model_name.split('__')[-1] if '__' in model_name else None
    
    registry = FileRegistry(config, timestamp=model_timestamp)
    
    eval_path = registry.get_evaluation_path()
    test_metrics = load_evaluation(eval_path) if eval_path.exists() else None
    
    if test_metrics:
        logger.info(f"Test metrics for {model_name}:")
        for key, val in test_metrics.items():
            logger.info(f"  {key}: {val:.2f}")
    else:
        logger.warning("Nenhuma métrica de teste encontrada")


def mode_compare(args):
    """Modo compare: compara todos os experimentos"""
    results_files = sorted(RESULTS_DIR.glob("*_results.json"))
    
    if not results_files:
        logger.warning("Nenhum arquivo de resultados encontrado")
        return
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON OF EXPERIMENTS")
    logger.info("=" * 80)
    
    experiments = []
    for results_file in results_files:
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                experiments.append({
                    'name': data['experiment'],
                    'best_loss': data['training'].get('best_val_loss', float('inf')),
                    'per': data['test_metrics'].get('per'),
                    'wer': data['test_metrics'].get('wer'),
                })
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Sort by best loss
    experiments.sort(key=lambda x: x['best_loss'])
    
    for exp in experiments:
        per_str = f"{exp['per']:.2f}" if exp['per'] is not None else "N/A"
        logger.info(f"{exp['name']}: loss={exp['best_loss']:.6f} | PER={per_str}")
    
    logger.info("=" * 80)
    logger.info("")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Análise unificada de treinamento',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model selection
    parser.add_argument('--model-name', type=str, help='Nome ou padrao do modelo')
    parser.add_argument('--index', type=int, help='Índice do modelo (uso com --list)')
    
    # Modes
    parser.add_argument('--list', action='store_true', help='Lista modelos e sai')
    parser.add_argument('--all', action='store_true', help='Processa todos os modelos (gera graficos + analise)')
    parser.add_argument('--monitor', action='store_true', help='Modo monitor (status ao vivo)')
    parser.add_argument('--test', action='store_true', help='Carrega métricas de teste')
    parser.add_argument('--compare', action='store_true', help='Compara todos os experimentos')
    
    args = parser.parse_args()
    
    # Listar modelos
    if args.list:
        list_available_models()
        return
    
    # Comparar experimentos
    if args.compare:
        mode_compare(args)
        return
    
    # Carregar lista de modelos para outros modos
    models_info = list_available_models()
    
    if not models_info:
        logger.error("Nenhum modelo disponivel")
        return
    
    # Selecionar modo
    if args.all:
        mode_all(models_info)
    elif args.monitor:
        mode_monitor(args, models_info)
    elif args.test:
        mode_test(args, models_info)
    else:
        mode_default(args, models_info)


if __name__ == '__main__':
    main()
