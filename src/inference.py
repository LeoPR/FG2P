#!/usr/bin/env python
import argparse
import torch
import editdistance
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from utils import get_logger, DATA_DIR, RESULTS_DIR, MODELS_DIR, get_cache_info, log_common_header
from g2p import G2PCorpus, G2PLSTMModel
from file_registry import get_base_name_from_path

# Fonte primária do dataset
DICT_PATH = Path("dicts/pt-br.tsv")

logger = get_logger("inference")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PERFORMANCE_PATH = Path(__file__).resolve().parent.parent / "docs" / "report" / "performance.json"

def load_model_metadata(model_path):
    """Carrega metadados do modelo se existir arquivo _metadata.json"""
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # Arquivo existe mas está vazio ou corrompido
            return None
    return None


def load_performance_data():
    """Carrega benchmarks manuais do arquivo docs/performance.json"""
    if not PERFORMANCE_PATH.exists():
        return None

    try:
        with open(PERFORMANCE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Falha ao ler {PERFORMANCE_PATH.name}: {exc}")
        return None

def _get_complete_model_files() -> list:
    """Retorna apenas modelos com training_completed=True, em ordem cronológica.
    Garante que os índices aqui coincidem com os índices do manage_experiments.py (seção COMPLETOS).
    """
    from utils import get_all_models_sorted
    return [f for f in get_all_models_sorted()
            if (lambda m: m is not None and m.get('training_completed', False))(load_model_metadata(f))]


def list_available_models():
    """Lista todos os modelos disponíveis com suas características"""
    model_files = _get_complete_model_files()
    
    if not model_files:
        print("\n❌ Nenhum modelo encontrado em models/")
        print("Execute train.py primeiro para treinar um modelo.\n")
        return []
    
    print("\n" + "=" * 80)
    print("MODELOS DISPONÍVEIS")
    print("=" * 80)
    
    models_info = []
    for idx, model_file in enumerate(model_files):
        metadata = load_model_metadata(model_file)
        
        print(f"\n[{idx}] {model_file.name}")
        print(f"    Arquivo: {model_file.stem}")
        
        if metadata:
            print(f"    Experimento: {metadata.get('experiment_name', 'N/A')}")
            print(f"    Treinado em: {metadata.get('timestamp', 'N/A')}")
            
            current_epoch = metadata.get('current_epoch', metadata.get('final_epoch', '?'))
            total_epochs = metadata.get('total_epochs', '?')
            is_complete = metadata.get('training_completed', False)
            status = "✓ Completo" if is_complete else "⚠ Incompleto"
            print(f"    Status: Epoch {current_epoch}/{total_epochs} {status}")
            print(f"    Best Loss: {metadata.get('best_loss', 'N/A'):.4f}")
            
            if 'config' in metadata and 'model' in metadata['config']:
                config = metadata['config']['model']
                print(f"    Arquitetura: emb={config.get('emb_dim', '?')} hidden={config.get('hidden_dim', '?')} " +
                      f"layers={config.get('num_layers', '?')} dropout={config.get('dropout', '?')}")
            
            print(f"    Parâmetros: {metadata.get('total_params', 'N/A'):,}")
        else:
            print("    ⚠ Metadados não disponíveis")
            print(f"    Tamanho: {model_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        models_info.append({
            'index': idx,
            'path': model_file,
            'metadata': metadata,
            'mtime': model_file.stat().st_mtime
        })
    
    print("\n" + "=" * 80)
    print(f"Total: {len(model_files)} modelo(s)")
    print(f"Mais recente: [{len(model_files) - 1}] {model_files[-1].name}")
    print("=" * 80)
    print("\nUso:")
    print("  python src/inference.py              # Usa o modelo mais recente")
    print("  python src/inference.py --index N    # Usa o modelo com índice N")
    print("  python src/inference.py --list       # Mostra esta lista novamente\n")
    
    return models_info

def select_model(args):
    """Seleciona modelo baseado nos argumentos"""
    model_files = _get_complete_model_files()
    
    if not model_files:
        logger.error("Nenhum modelo encontrado em models/")
        logger.error("Execute train.py primeiro")
        return None
    
    # Se especificou índice
    if args.index is not None:
        if 0 <= args.index < len(model_files):
            return model_files[args.index]
        else:
            logger.error(f"Índice {args.index} inválido. Modelos disponíveis: 0 a {len(model_files) - 1}")
            logger.info("Use --list para ver todos os modelos")
            return None
    
    # Se especificou nome
    if args.model:
        model_path = MODELS_DIR / f"{args.model}.pt"
        if model_path.exists():
            return model_path
        else:
            logger.error(f"Modelo {args.model} não encontrado em models/")
            logger.info("Use --list para ver modelos disponíveis")
            return None
    
    # Usar mais recente por padrão
    logger.info(f"Usando modelo mais recente: {model_files[-1].name}")
    return model_files[-1]

def analyze_length_distribution(predictions, references, words):
    """Analisa distribuição de comprimento e truncamento"""
    shorter_count = 0
    longer_count = 0
    exact_count = 0
    missing_phonemes = []
    extra_phonemes = []
    
    for pred, ref in zip(predictions, references):
        pred_len = len(pred)
        ref_len = len(ref)
        
        if pred_len < ref_len:
            shorter_count += 1
            missing_phonemes.append(ref_len - pred_len)
        elif pred_len > ref_len:
            longer_count += 1
            extra_phonemes.append(pred_len - ref_len)
        else:
            exact_count += 1
    
    return {
        'shorter': shorter_count,
        'longer': longer_count,
        'exact': exact_count,
        'missing_phonemes': missing_phonemes,
        'extra_phonemes': extra_phonemes,
        'avg_missing': sum(missing_phonemes) / len(missing_phonemes) if missing_phonemes else 0,
        'avg_extra': sum(extra_phonemes) / len(extra_phonemes) if extra_phonemes else 0
    }



def predict_word(model, word, char_vocab, device):
    """Prediz fonemas para uma palavra usando o decoder autoregressivo."""
    model.eval()
    chars = torch.LongTensor(char_vocab.encode(word)).unsqueeze(0).to(device)
    char_lengths = torch.LongTensor([len(word)]).to(device)

    # model.predict gera autoregressivamente até EOS ou max_len
    predictions = model.predict(chars, char_lengths, max_len=50)
    return predictions[0]  # retorna lista de índices (sem EOS/PAD)

def calculate_accuracy(predictions, references):
    """Calcula Word-level Accuracy (porcentagem de palavras 100% corretas)"""
    correct = 0
    total = 0

    for pred, ref in zip(predictions, references):
        pred_str = ' '.join(pred)
        ref_str = ' '.join(ref)

        if pred_str == ref_str:
            correct += 1

        total += 1

    return (correct / total * 100) if total > 0 else 0

def calculate_wer(predictions, references):
    """Calcula Word Error Rate (complemento da accuracy)"""
    accuracy = calculate_accuracy(predictions, references)
    return 100 - accuracy

def calculate_per(predictions, references):
    """Calcula Phoneme Error Rate"""
    total_errors = 0
    total_phonemes = 0

    for pred, ref in zip(predictions, references):
        dist = editdistance.eval(pred, ref)
        total_errors += dist
        total_phonemes += len(ref)

    return (total_errors / total_phonemes * 100) if total_phonemes > 0 else 0

def analyze_errors(predictions, references, words):
    """Analisa tipos de erros e padrões"""
    analysis = {
        'substitutions': Counter(),
        'total_substitutions': 0,
        'total_deletions': 0,
        'total_insertions': 0,
        'error_examples': [],
        'correct_examples': [],
    }

    for word, pred, ref in zip(words, predictions, references):
        if pred == ref:
            if len(analysis['correct_examples']) < 20:
                analysis['correct_examples'].append((word, pred, ref))
        else:
            if len(analysis['error_examples']) < 20:
                analysis['error_examples'].append((word, pred, ref))
            
            # Contar tipos de erro
            pred_len = len(pred)
            ref_len = len(ref)
            
            if pred_len > ref_len:
                analysis['total_insertions'] += (pred_len - ref_len)
            elif pred_len < ref_len:
                analysis['total_deletions'] += (ref_len - pred_len)
            
            # Substituições
            for p, r in zip(pred, ref):
                if p != r:
                    analysis['substitutions'][(r, p)] += 1
                    analysis['total_substitutions'] += 1

    return analysis

def format_benchmark_comparison(per, wer, accuracy, dataset_size, perf_data=None):
    """Formata comparação com benchmarks da literatura"""
    benchmarks = [
        {
            'name': 'FG2P (este run)',
            'lang': 'PT-BR',
            'dataset': f'{dataset_size//1000}k (estratificado)',
            'PER': per,
            'WER': wer,
            'Accuracy': accuracy,
            'notes': 'resultado atual'
        }
    ]

    if perf_data:
        for entry in perf_data.get("fg2p_models", []):
            benchmarks.append({
                'name': entry.get('name', 'FG2P'),
                'lang': entry.get('lang', 'PT-BR'),
                'dataset': entry.get('dataset', 'n/d'),
                'PER': entry.get('per'),
                'WER': entry.get('wer'),
                'Accuracy': entry.get('accuracy'),
                'notes': entry.get('notes')
            })

        for entry in perf_data.get("literature_ptbr", []):
            benchmarks.append({
                'name': entry.get('name', 'N/A'),
                'lang': entry.get('lang', 'PT-BR'),
                'dataset': entry.get('dataset', 'n/d'),
                'PER': entry.get('per'),
                'WER': entry.get('wer'),
                'Accuracy': entry.get('accuracy'),
                'notes': entry.get('notes')
            })

        for entry in perf_data.get("literature_general", []):
            benchmarks.append({
                'name': entry.get('name', 'N/A'),
                'lang': entry.get('lang', 'N/A'),
                'dataset': entry.get('dataset', 'n/d'),
                'PER': entry.get('per'),
                'WER': entry.get('wer'),
                'Accuracy': entry.get('accuracy'),
                'notes': entry.get('notes')
            })

        return benchmarks

    # Fallback: mantém a lista fixa quando o arquivo não existe
    return benchmarks

def process_all_models(force=False):
    """Processa todos os modelos que precisam de avaliação
    
    Args:
        force: Se True, reprocessa mesmo que já exista avaliação
    """
    from utils import get_all_models_sorted
    model_files = get_all_models_sorted()
    
    if not model_files:
        logger.error("Nenhum modelo encontrado em models/")
        return
    
    print("\n" + "=" * 80)
    print("PROCESSAMENTO AUTOMÁTICO DE MODELOS")
    print("=" * 80)
    
    models_to_process = []
    skipped = []
    
    for model_file in model_files:
        base_name = get_base_name_from_path(model_file)
        if not base_name:
            base_name = model_file.stem
        
        evaluation_file = RESULTS_DIR / f"evaluation_{base_name}.txt"
        
        # Verificar se já tem avaliação
        if evaluation_file.exists() and not force:
            skipped.append(base_name)
            continue
        
        # Verificar se o modelo está completo (treinamento finalizado)
        metadata = load_model_metadata(model_file)
        if metadata:
            if not metadata.get('training_completed', False):
                logger.warning(f"⚠ {base_name}: treinamento incompleto, pulando")
                skipped.append(base_name)
                continue
        
        models_to_process.append((model_file, base_name))
    
    print(f"\nModelos a processar: {len(models_to_process)}")
    print(f"Modelos já processados (pulados): {len(skipped)}")
    
    if not models_to_process:
        print("\n✓ Todos os modelos já foram avaliados!")
        if skipped and not force:
            print("\nUse --force para reprocessar modelos já avaliados")
        return
    
    print("\nModelos que serão processados:")
    for model_file, base_name in models_to_process:
        print(f"  • {base_name}")
    
    if force:
        print("\nModo FORCE ativado: reprocessando tudo")
    
    print("\nIniciando processamento...")
    
    success_count = 0
    error_count = 0
    
    for idx, (model_file, base_name) in enumerate(models_to_process, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(models_to_process)}] Processando: {base_name}")
        print(f"{'='*80}")
        
        try:
            # Executar avaliação para este modelo
            run_inference_for_model(model_file)
            success_count += 1
            print(f"\n✓ {base_name}: avaliação completa")
        except Exception as e:
            error_count += 1
            logger.error(f"✗ {base_name}: erro durante avaliação: {e}")
    
    print(f"\n{'='*80}")
    print("RESUMO DO PROCESSAMENTO")
    print(f"{'='*80}")
    print(f"Sucesso: {success_count}/{len(models_to_process)}")
    print(f"Erros: {error_count}/{len(models_to_process)}")
    print(f"{'='*80}\n")

def run_inference_for_model(model_path):
    """Executa avaliação completa para um modelo específico
    
    Args:
        model_path: Path do modelo a ser avaliado
    """
    # Carregar metadados e iniciar timer
    metadata = load_model_metadata(model_path)
    start_time = time.perf_counter()
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # ---- Carregar corpus e reproduzir o split exato do treino ----
    grapheme_encoding = "raw"
    keep_syllable_separators = False
    if metadata and 'config' in metadata:
        data_config = metadata['config'].get('data', metadata['config'])
        grapheme_encoding = data_config.get('grapheme_encoding', 'raw')
        keep_syllable_separators = data_config.get('keep_syllable_separators', False)

    logger.info("=== Carregando corpus ===")
    logger.info(f"Fonte primária: {DICT_PATH}")
    logger.info(f"Codificação grafêmica: {grapheme_encoding}")
    logger.info(
        "Separadores de sílaba: %s",
        "manter" if keep_syllable_separators else "remover",
    )
    corpus = G2PCorpus(
        DICT_PATH,
        grapheme_encoding=grapheme_encoding,
        keep_syllable_separators=keep_syllable_separators,
    )

    # Verificar integridade se o metadata tiver checksum
    if metadata and 'dataset' in metadata:
        ds_meta = metadata['dataset']
        stored_ck = ds_meta.get('dict_checksum', '')
        if stored_ck:
            corpus.verify(stored_ck)
            logger.info("✓ Checksum do dicionário verificado")

    # Reproduzir o split exato usado no treino
    split_seed = 42
    split_test_ratio = 0.20
    split_val_ratio = 0.10
    if metadata and 'config' in metadata:
        data_config = metadata['config'].get('data', metadata['config'])
        split_seed = data_config.get('split_seed', split_seed)
        split_test_ratio = data_config.get('test_ratio', split_test_ratio)
        split_val_ratio = data_config.get('val_ratio', split_val_ratio)

    split = corpus.split(
        test_ratio=split_test_ratio,
        val_ratio=split_val_ratio,
        seed=split_seed,
    )
    words, phonemes_list = split.test_pairs()
    words_raw = [corpus.words_raw[i] for i in split.test_indices]
    phonemes = phonemes_list
    logger.info(f"Dataset de teste reproduzido: {len(words)} palavras (seed={split_seed})")

    char_vocab = corpus.char_vocab
    phoneme_vocab = corpus.phoneme_vocab

    # Carregar modelo
    logger.info("=== Carregando modelo ===")
    if metadata and 'config' in metadata and 'model' in metadata['config']:
        mc = metadata['config']['model']
        emb_dim = int(mc.get('emb_dim', 128))
        hidden_dim = int(mc.get('hidden_dim', 256))
        num_layers = int(mc.get('num_layers', 2))
        dropout = float(mc.get('dropout', 0.5))
        embedding_type = str(mc.get('embedding_type', 'learned'))
    else:
        emb_dim = 128
        hidden_dim = 256
        num_layers = 2
        dropout = 0.5
        embedding_type = 'learned'

    if embedding_type == 'panphon':
        model = G2PLSTMModel(
            len(char_vocab), len(phoneme_vocab),
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            embedding_type=embedding_type,
            phoneme_i2p=phoneme_vocab.i2p,
        ).to(device)
    else:
        model = G2PLSTMModel(
            len(char_vocab), len(phoneme_vocab),
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            embedding_type=embedding_type,
        ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    logger.info("✓ Modelo carregado com sucesso")

    # Predições
    logger.info("=== Fazendo predições ===")
    inference_start = time.perf_counter()
    predictions = []
    references = [p.split() for p in phonemes]

    for word in words:
        pred_indices = predict_word(model, word, char_vocab, device)
        pred_phonemes = phoneme_vocab.decode(pred_indices)
        predictions.append(pred_phonemes)
    
    inference_time = time.perf_counter() - inference_start
    samples_per_sec = len(words) / inference_time if inference_time > 0 else 0
    logger.info("✓ Predições completas")
    logger.info(f"Tempo: {inference_time:.2f}s | Velocidade: {samples_per_sec:.1f} palavras/s")

    # Calcular métricas
    logger.info("=== Calculando métricas ===")
    per = calculate_per(predictions, references)
    wer = calculate_wer(predictions, references)
    accuracy = calculate_accuracy(predictions, references)
    length_analysis = analyze_length_distribution(predictions, references, words)
    error_analysis = analyze_errors(predictions, references, words_raw)

    logger.info("")
    logger.info("=" * 70)
    logger.info("MÉTRICAS DE AVALIAÇÃO")
    logger.info("=" * 70)
    logger.info(f"PER: {per:.2f}% | WER: {wer:.2f}% | Accuracy: {accuracy:.2f}%")
    logger.info("=" * 70)
    
    # Salvar predições
    base_name = get_base_name_from_path(model_path)
    if not base_name:
        base_name = model_path.stem
    
    predictions_file = RESULTS_DIR / f"predictions_{base_name}.tsv"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        f.write("word\tpredicted\treference\tcorrect\n")
        for word, pred, ref in zip(words_raw, predictions, references):
            pred_str = ' '.join(pred)
            ref_str = ' '.join(ref)
            correct = '1' if pred == ref else '0'
            f.write(f"{word}\t{pred_str}\t{ref_str}\t{correct}\n")
    
    logger.info(f"Predições salvas: {predictions_file.name}")

    # Salvar resumo
    results_file = RESULTS_DIR / f"evaluation_{base_name}.txt"
    total_time = time.perf_counter() - start_time
    
    perf_data = load_performance_data()
    benchmarks = format_benchmark_comparison(per, wer, accuracy, len(words), perf_data)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("G2P LSTM - Evaluation Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {timestamp_str}\n")
        f.write(f"Model: {model_path.name}\n")
        if metadata:
            f.write(f"Experiment: {metadata.get('experiment_name', 'N/A')}\n")
            f.write(f"Training Status: Epoch {metadata.get('current_epoch', '?')}/{metadata.get('total_epochs', '?')} ")
            f.write(f"({'Complete' if metadata.get('training_completed', False) else 'Incomplete'})\n")
        f.write(f"Test set: {len(words)} words\n")
        f.write(f"Inference time: {inference_time:.2f}s ({samples_per_sec:.1f} samples/s)\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write("\n")
        f.write("METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"PER (Phoneme Error Rate): {per:.2f}%\n")
        f.write(f"WER (Word Error Rate):    {wer:.2f}%\n")
        f.write(f"Accuracy (Word-level):    {accuracy:.2f}%\n")
        f.write(f"Correct words: {int(len(words) * accuracy / 100)}/{len(words)}\n")
        f.write("\n")
        f.write("LENGTH ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Predictions shorter: {length_analysis['shorter']} ({length_analysis['shorter']/len(words)*100:.1f}%)\n")
        f.write(f"Predictions longer:  {length_analysis['longer']} ({length_analysis['longer']/len(words)*100:.1f}%)\n")
        f.write(f"Predictions exact:   {length_analysis['exact']} ({length_analysis['exact']/len(words)*100:.1f}%)\n")
        if length_analysis['shorter'] > 0:
            f.write(f"Avg missing phonemes: {length_analysis['avg_missing']:.2f}\n")
        if length_analysis['longer'] > 0:
            f.write(f"Avg extra phonemes: {length_analysis['avg_extra']:.2f}\n")
        f.write("\n")
        f.write("ERROR ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Substitutions: {error_analysis['total_substitutions']}\n")
        f.write(f"Deletions:     {error_analysis['total_deletions']}\n")
        f.write(f"Insertions:    {error_analysis['total_insertions']}\n")
        f.write("\n")
        f.write("BENCHMARK COMPARISON\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<28} {'Lang':<7} {'Dataset':<16} {'PER%':<7} {'WER%':<7} {'Acc%':<7}\n")
        f.write("-" * 70 + "\n")
        for bm in benchmarks:
            per_str = f"{bm['PER']:.2f}" if bm['PER'] is not None else 'n/d'
            wer_str = f"{bm['WER']:.2f}" if bm['WER'] is not None else 'n/d'
            acc_str = f"{bm['Accuracy']:.2f}" if bm['Accuracy'] is not None else 'n/d'
            f.write(
                f"{bm['name']:<28} {bm['lang']:<7} {bm['dataset']:<16} {per_str:<7} {wer_str:<7} {acc_str:<7}\n"
            )
        f.write("\n")
        f.write("ERROR EXAMPLES (first 20)\n")
        f.write("-" * 70 + "\n")
        for word, pred, ref in error_analysis['error_examples']:
            pred_str = ' '.join(pred)
            ref_str = ' '.join(ref)
            f.write(f"{word:15} | Pred: {pred_str:25} | Ref: {ref_str}\n")
    
    logger.info(f"Resultados salvos: {results_file.name}")

def main():
    parser = argparse.ArgumentParser(
        description='Avaliação do modelo G2P LSTM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python src/inference.py              # Usa o modelo mais recente
  python src/inference.py --list       # Lista todos os modelos disponíveis
  python src/inference.py --index 2    # Usa o modelo com índice 2
  python src/inference.py --model g2p_lear_adam_80train_20260214_223939
        """
    )
    
    parser.add_argument('--list', action='store_true', 
                       help='Lista todos os modelos disponíveis e sai')
    parser.add_argument('--index', type=int, metavar='N',
                       help='Usa o modelo com índice N da lista')
    parser.add_argument('--model', type=str, metavar='NAME',
                       help='Nome do modelo (sem extensão .pt)')
    parser.add_argument('--all', action='store_true',
                       help='Processa todos os modelos que ainda não têm avaliação')
    parser.add_argument('--force', action='store_true',
                       help='Força reprocessamento mesmo se já existir avaliação')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simula decisão incremental (rodaria ou pularia) sem executar inferência')
    
    args = parser.parse_args()
    
    # Se --list, apenas lista e sai
    if args.list:
        list_available_models()
        return
    
    # Se --all, processar todos os modelos pendentes
    if args.all:
        process_all_models(args.force)
        return
    
    # Selecionar modelo
    model_path = select_model(args)
    if model_path is None:
        return

    # Dry-run leve para validação cruzada com manager
    if args.dry_run:
        base_name = get_base_name_from_path(model_path)
        if not base_name:
            base_name = model_path.stem

        evaluation_file = RESULTS_DIR / f"evaluation_{base_name}.txt"
        predictions_file = RESULTS_DIR / f"predictions_{base_name}.tsv"

        has_eval = evaluation_file.exists()
        has_pred = predictions_file.exists()
        would_run = args.force or (not has_eval) or (not has_pred)

        print("\n[DRY RUN] inference.py")
        print(f"[DRY RUN] Modelo: {base_name}")
        print(f"[DRY RUN] Artefatos: evaluation={'OK' if has_eval else 'MISSING'} | predictions={'OK' if has_pred else 'MISSING'}")
        print(f"[DRY RUN] Decisão incremental: {'RODARIA' if would_run else 'PULARIA'}")
        if args.force:
            print("[DRY RUN] Modo FORCE: ativo")
        return
    
    # Carregar metadados e iniciar timer
    metadata = load_model_metadata(model_path)
    start_time = time.perf_counter()
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # ---- Carregar corpus e reproduzir o split exato do treino ----
    grapheme_encoding = "raw"
    keep_syllable_separators = False
    if metadata and 'config' in metadata:
        data_config = metadata['config'].get('data', metadata['config'])
        grapheme_encoding = data_config.get('grapheme_encoding', 'raw')
        keep_syllable_separators = data_config.get('keep_syllable_separators', False)

    logger.info("=== Carregando corpus ===")
    logger.info(f"Fonte primária: {DICT_PATH}")
    logger.info(f"Codificação grafêmica: {grapheme_encoding}")
    logger.info(
        "Separadores de sílaba: %s",
        "manter" if keep_syllable_separators else "remover",
    )
    corpus = G2PCorpus(
        DICT_PATH,
        grapheme_encoding=grapheme_encoding,
        keep_syllable_separators=keep_syllable_separators,
    )

    # Verificar integridade se o metadata tiver checksum
    if metadata and 'dataset' in metadata:
        ds_meta = metadata['dataset']
        stored_ck = ds_meta.get('dict_checksum', '')
        if stored_ck:
            corpus.verify(stored_ck)
            logger.info("✓ Checksum do dicionário verificado")

    # Reproduzir o split exato usado no treino
    split_seed = 42
    split_test_ratio = 0.20
    split_val_ratio = 0.10
    if metadata and 'config' in metadata:
        # Metadata salva config em config.data.{split_seed, test_ratio, val_ratio}
        data_config = metadata['config'].get('data', metadata['config'])
        split_seed = data_config.get('split_seed', split_seed)
        split_test_ratio = data_config.get('test_ratio', split_test_ratio)
        split_val_ratio = data_config.get('val_ratio', split_val_ratio)

    split = corpus.split(
        test_ratio=split_test_ratio,
        val_ratio=split_val_ratio,
        seed=split_seed,
    )
    words, phonemes_list = split.test_pairs()
    words_raw = [corpus.words_raw[i] for i in split.test_indices]
    # phonemes_list já são strings espaçadas ("k a ˈl ã").
    # NÃO usar ' '.join(p) — isso itera sobre caracteres e separa
    # combining marks (U+0303) da vogal base, inflando PER/WER.
    phonemes = phonemes_list
    logger.info(f"Dataset de teste reproduzido: {len(words)} palavras (seed={split_seed})")
    logger.info("  → Test set NUNCA visto durante treino (val usado para checkpoint).")

    char_vocab = corpus.char_vocab
    phoneme_vocab = corpus.phoneme_vocab
    logger.info(f"Caracteres únicos: {len(char_vocab)}")
    logger.info(f"Fonemas únicos: {len(phoneme_vocab)}")

    dataset_meta = None
    if metadata and 'dataset' in metadata:
        dataset_meta = metadata['dataset']
    else:
        dataset_meta = split.metadata()

    cache_names = split.cache_filenames()
    cache_info = get_cache_info(
        DATA_DIR,
        train_name=cache_names["train"],
        val_name=cache_names["val"],
        test_name=cache_names["test"],
    )
    log_common_header(
        logger,
        "AVALIACAO",
        timestamp_str,
        device,
        dataset_meta=dataset_meta,
        cache_info=cache_info,
    )

    # Informações do modelo
    logger.info("=" * 70)
    logger.info("CONFIGURACAO DO MODELO")
    logger.info("=" * 70)
    logger.info(f"Arquivo: {model_path.name}")
    
    if metadata:
        logger.info(f"Experimento: {metadata.get('experiment_name', 'N/A')}")
        logger.info(f"Run ID: {metadata.get('run_id', 'N/A')}")
        
        # Status do treino
        current_epoch = metadata.get('current_epoch', '?')
        total_epochs = metadata.get('total_epochs', '?')
        is_complete = metadata.get('training_completed', False)
        status_icon = "✓" if is_complete else "⚠"
        status_text = "Completo" if is_complete else "Incompleto"
        
        logger.info(f"Treino: Epoch {current_epoch}/{total_epochs} {status_icon} {status_text}")
        logger.info(f"Treinado em: {metadata.get('timestamp', 'N/A')}")
        logger.info(f"Best Loss: {metadata.get('best_loss', 'N/A')}")
        
        # Parâmetros do modelo
        if 'config' in metadata and 'model' in metadata['config']:
            model_config = metadata['config']['model']
            logger.info(f"Embedding Type: {model_config.get('embedding_type', 'learned')}")
            logger.info(f"Embedding Dim: {model_config.get('emb_dim', 'N/A')}")
            logger.info(f"Hidden: {model_config.get('hidden_dim', 'N/A')}")
            logger.info(f"Layers: {model_config.get('num_layers', 'N/A')}")
            logger.info(f"Dropout: {model_config.get('dropout', 'N/A')}")
        
        logger.info(f"Total Parametros: {metadata.get('total_params', 'N/A'):,}")
    else:
        logger.info("⚠ Metadados nao encontrados (modelo treinado com versao antiga)")
    
    logger.info("=" * 70)
    logger.info("")

    logger.info("")
    logger.info("=== Carregando modelo ===")
    # Usar config do metadata se disponível, senão defaults
    if metadata and 'config' in metadata and 'model' in metadata['config']:
        mc = metadata['config']['model']
        emb_dim = int(mc.get('emb_dim', 128))
        hidden_dim = int(mc.get('hidden_dim', 256))
        num_layers = int(mc.get('num_layers', 2))
        dropout = float(mc.get('dropout', 0.5))
        embedding_type = str(mc.get('embedding_type', 'learned'))
    else:
        emb_dim = 128
        hidden_dim = 256
        num_layers = 2
        dropout = 0.5
        embedding_type = 'learned'

    if embedding_type == 'panphon':
        model = G2PLSTMModel(
            len(char_vocab), len(phoneme_vocab),
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            embedding_type=embedding_type,
            phoneme_i2p=phoneme_vocab.i2p,
        ).to(device)
    else:
        model = G2PLSTMModel(
            len(char_vocab), len(phoneme_vocab),
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            embedding_type=embedding_type,
        ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    logger.info("✓ Modelo carregado com sucesso")

    logger.info("")
    logger.info("=== Fazendo predições ===")
    inference_start = time.perf_counter()
    predictions = []
    references = [p.split() for p in phonemes]

    for word in words:
        pred_indices = predict_word(model, word, char_vocab, device)
        pred_phonemes = phoneme_vocab.decode(pred_indices)
        predictions.append(pred_phonemes)
    
    inference_time = time.perf_counter() - inference_start
    samples_per_sec = len(words) / inference_time if inference_time > 0 else 0
    logger.info("✓ Predições completas")
    logger.info(f"Tempo: {inference_time:.2f}s | Velocidade: {samples_per_sec:.1f} palavras/s")

    logger.info("")
    logger.info("=== Calculando métricas ===")
    per = calculate_per(predictions, references)
    wer = calculate_wer(predictions, references)
    accuracy = calculate_accuracy(predictions, references)
    length_analysis = analyze_length_distribution(predictions, references, words)

    logger.info("")
    logger.info("=" * 70)
    logger.info("MÉTRICAS DE AVALIAÇÃO")
    logger.info("=" * 70)
    logger.info(f"PER (Phoneme Error Rate): {per:.2f}%")
    logger.info(f"WER (Word Error Rate):    {wer:.2f}%")
    logger.info(f"Accuracy (Word-level):    {accuracy:.2f}%")
    logger.info("")
    logger.info(f"Total de palavras testadas: {len(words)}")
    logger.info(f"Palavras corretas: {int(len(words) * accuracy / 100)}")
    logger.info(f"Palavras com erro: {int(len(words) * wer / 100)}")
    logger.info("=" * 70)
    logger.info("")

    # Análise de erros
    logger.info("=== Análise de erros ===")
    error_analysis = analyze_errors(predictions, references, words_raw)
    
    # Top erros de substituição
    if error_analysis['substitutions']:
        logger.info("Top 5 substituições mais comuns:")
        for (ref_ph, pred_ph), count in error_analysis['substitutions'].most_common(5):
            logger.info(f"  {ref_ph} → {pred_ph}: {count} vezes")
        logger.info("")

    # Comparação com literatura
    logger.info("=" * 70)
    logger.info("COMPARAÇÃO COM LITERATURA E ESTADO DA ARTE")
    logger.info("=" * 70)
    perf_data = load_performance_data()
    benchmarks = format_benchmark_comparison(per, wer, accuracy, len(words), perf_data)
    
    # Cabeçalho
    logger.info(f"{'Sistema':<35} {'Idioma':<7} {'Dataset':<18} {'PER':<7} {'WER':<7} {'Acc':<7}")
    logger.info("-" * 80)
    
    # Linhas de dados
    for bm in benchmarks:
        per_str = f"{bm['PER']:.2f}%" if bm['PER'] is not None else 'N/A'
        wer_str = f"{bm['WER']:.2f}%" if bm['WER'] is not None else 'N/A'
        acc_str = f"{bm['Accuracy']:.2f}%" if bm['Accuracy'] is not None else 'N/A'
        
        logger.info(
            f"{bm['name']:<35} {bm['lang']:<7} {bm['dataset']:<18} {per_str:<7} {wer_str:<7} {acc_str:<7}"
        )
    
    logger.info("=" * 70)
    logger.info("")
    logger.info("OBSERVAÇÕES E CONTEXTO:")
    notes = perf_data.get("notes") if perf_data else None
    if notes:
        for note in notes:
            logger.info(f"  • {note}")
    else:
        logger.info("  • EN (inglês) é muito mais irregular que PT-BR (português regular)")
        logger.info("    → WER 10% em EN pode ser equivalente a WER 5% em PT-BR")
        logger.info("  • XphoneBR (SOTA PT-BR) não publica métricas oficialmente")
        logger.info("  • FG2P usa split estratificado 60/10/30 com 95.937 palavras totais")
        logger.info("  • Detalhes metodológicos: ver docs/LITERATURE.md")
    logger.info("")

    # =========================================================================
    # Salvar predições (TSV) para análise posterior por analyze_errors.py
    # =========================================================================
    # Extrair base_name do modelo para manter rastreabilidade
    base_name = get_base_name_from_path(model_path)
    if not base_name:
        # Fallback se não conseguir extrair
        base_name = model_path.stem
    
    predictions_file = RESULTS_DIR / f"predictions_{base_name}.tsv"
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        f.write("word\tpredicted\treference\tcorrect\n")
        for word, pred, ref in zip(words_raw, predictions, references):
            pred_str = ' '.join(pred)
            ref_str = ' '.join(ref)
            correct = '1' if pred == ref else '0'
            f.write(f"{word}\t{pred_str}\t{ref_str}\t{correct}\n")
    
    logger.info(f"Predições salvas em: {predictions_file.name}")
    logger.info("  → Use 'python src/analyze_errors.py {predictions_file.name}' para análise fonética detalhada")
    logger.info("")

    # Salvar resumo da avaliação
    results_file = RESULTS_DIR / f"evaluation_{base_name}.txt"
    
    total_time = time.perf_counter() - start_time
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("G2P LSTM - Evaluation Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {timestamp_str}\n")
        f.write(f"Model: {model_path.name}\n")
        if metadata:
            f.write(f"Experiment: {metadata.get('experiment_name', 'N/A')}\n")
            f.write(f"Training Status: Epoch {metadata.get('current_epoch', '?')}/{metadata.get('total_epochs', '?')} ")
            f.write(f"({'Complete' if metadata.get('training_completed', False) else 'Incomplete'})\n")
        f.write(f"Test set: {len(words)} words\n")
        f.write(f"Inference time: {inference_time:.2f}s ({samples_per_sec:.1f} samples/s)\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write("\n")
        f.write("METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"PER (Phoneme Error Rate): {per:.2f}%\n")
        f.write(f"WER (Word Error Rate):    {wer:.2f}%\n")
        f.write(f"Accuracy (Word-level):    {accuracy:.2f}%\n")
        f.write(f"Correct words: {int(len(words) * accuracy / 100)}/{len(words)}\n")
        f.write("\n")
        f.write("LENGTH ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Predictions shorter: {length_analysis['shorter']} ({length_analysis['shorter']/len(words)*100:.1f}%)\n")
        f.write(f"Predictions longer:  {length_analysis['longer']} ({length_analysis['longer']/len(words)*100:.1f}%)\n")
        f.write(f"Predictions exact:   {length_analysis['exact']} ({length_analysis['exact']/len(words)*100:.1f}%)\n")
        if length_analysis['shorter'] > 0:
            f.write(f"Avg missing phonemes: {length_analysis['avg_missing']:.2f}\n")
        if length_analysis['longer'] > 0:
            f.write(f"Avg extra phonemes: {length_analysis['avg_extra']:.2f}\n")
        f.write("\n")
        f.write("ERROR ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Substitutions: {error_analysis['total_substitutions']}\n")
        f.write(f"Deletions:     {error_analysis['total_deletions']}\n")
        f.write(f"Insertions:    {error_analysis['total_insertions']}\n")
        f.write("\n")
        f.write("BENCHMARK COMPARISON\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<28} {'Lang':<7} {'Dataset':<16} {'PER%':<7} {'WER%':<7} {'Acc%':<7}\n")
        f.write("-" * 70 + "\n")
        for bm in benchmarks:
            per_str = f"{bm['PER']:.2f}" if bm['PER'] is not None else 'n/d'
            wer_str = f"{bm['WER']:.2f}" if bm['WER'] is not None else 'n/d'
            acc_str = f"{bm['Accuracy']:.2f}" if bm['Accuracy'] is not None else 'n/d'
            f.write(
                f"{bm['name']:<28} {bm['lang']:<7} {bm['dataset']:<16} {per_str:<7} {wer_str:<7} {acc_str:<7}\n"
            )
        f.write("\n")
        notes = perf_data.get("notes") if perf_data else None
        if notes:
            f.write("Notes:\n")
            for note in notes:
                f.write(f"  - {note}\n")
        else:
            f.write("Note: EN (irregular) vs PT-BR (regular) are not directly comparable.\n")
            f.write("      See docs/LITERATURE.md for detailed analysis.\n")
        f.write("\n")
        f.write("ERROR EXAMPLES (first 20)\n")
        f.write("-" * 70 + "\n")
        for word, pred, ref in error_analysis['error_examples']:
            pred_str = ' '.join(pred)
            ref_str = ' '.join(ref)
            f.write(f"{word:15} | Pred: {pred_str:25} | Ref: {ref_str}\n")
    
    logger.info(f"Resultados salvos em: {results_file.name}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("AVALIAÇÃO COMPLETA ✓")
    logger.info(f"Tempo total: {total_time:.2f}s")
    logger.info("=" * 70)
    logger.info("")

if __name__ == "__main__":
    main()
