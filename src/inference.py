#!/usr/bin/env python
import argparse
import torch
import json
import time
from datetime import datetime
from pathlib import Path
from utils import get_logger, DATA_DIR, RESULTS_DIR, MODELS_DIR, get_cache_info, log_common_header
from g2p import G2PCorpus, G2PLSTMModel
from file_registry import get_base_name_from_path, list_experiments

# Fonte primária do dataset
DICT_PATH = Path("dicts/pt-br.tsv")

logger = get_logger("inference")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de I/O e listagem
# ─────────────────────────────────────────────────────────────────────────────

def load_model_metadata(model_path):
    """Carrega metadados do modelo se existir arquivo _metadata.json"""
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _get_complete_model_files() -> list:
    """Retorna apenas modelos com training_completed=True, em ordem cronológica.
    Delega para file_registry.list_experiments — fonte única de verdade.
    """
    return [r.pt_path for r in list_experiments(complete_only=True)]


def list_available_models():
    """Lista todos os modelos disponíveis com suas características"""
    model_files = _get_complete_model_files()

    if not model_files:
        print("\n[ERRO] Nenhum modelo encontrado em models/")
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
            status = "[OK] Completo" if is_complete else "[!] Incompleto"
            print(f"    Status: Epoch {current_epoch}/{total_epochs} {status}")
            print(f"    Best Loss: {metadata.get('best_loss', 'N/A'):.4f}")

            if 'config' in metadata and 'model' in metadata['config']:
                config = metadata['config']['model']
                print(f"    Arquitetura: emb={config.get('emb_dim', '?')} "
                      f"hidden={config.get('hidden_dim', '?')} "
                      f"layers={config.get('num_layers', '?')} "
                      f"dropout={config.get('dropout', '?')}")

            print(f"    Parâmetros: {metadata.get('total_params', 'N/A'):,}")
        else:
            print("    [!] Metadados nao disponiveis")
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

    if args.index is not None:
        if 0 <= args.index < len(model_files):
            return model_files[args.index]
        logger.error("Índice %d inválido. Modelos disponíveis: 0 a %d", args.index, len(model_files) - 1)
        logger.info("Use --list para ver todos os modelos")
        return None

    if args.model:
        matches = list(MODELS_DIR.glob(f"**/{args.model}.pt"))
        if matches:
            return matches[0]
        model_path = MODELS_DIR / f"{args.model}.pt"
        if model_path.exists():
            return model_path
        logger.error("Modelo '%s' nao encontrado em models/", args.model)
        logger.info("Use --list para ver modelos disponíveis")
        return None

    logger.info("Usando modelo mais recente: %s", model_files[-1].name)
    return model_files[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Core de inferência — função única, usada por main() e process_all_models()
# ─────────────────────────────────────────────────────────────────────────────

def predict_word(model, word, char_vocab, device):
    """Prediz fonemas para uma palavra usando o decoder autoregressivo."""
    chars = torch.LongTensor(char_vocab.encode(word)).unsqueeze(0).to(device)
    char_lengths = torch.LongTensor([len(word)]).to(device)
    predictions = model.predict(chars, char_lengths, max_len=50)
    return predictions[0]


# ─────────────────────────────────────────────────────────────────────────────
# Função principal de inferência — fonte única de verdade
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_for_model(model_path, verbose=False):
    """Executa inferência completa para um modelo específico.

    Args:
        model_path: Path do arquivo .pt
        verbose:    Se True, log detalhado (header completo, config do modelo,
                    tabela de benchmark no console). Se False, log compacto
                    para uso em batch (process_all_models).
    """
    metadata = load_model_metadata(model_path)
    start_time = time.perf_counter()
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── Corpus ───────────────────────────────────────────────────────────────
    grapheme_encoding = "raw"
    keep_syllable_separators = False
    if metadata and 'config' in metadata:
        data_config = metadata['config'].get('data', metadata['config'])
        grapheme_encoding = data_config.get('grapheme_encoding', 'raw')
        keep_syllable_separators = data_config.get('keep_syllable_separators', False)

    corpus = G2PCorpus(
        DICT_PATH,
        grapheme_encoding=grapheme_encoding,
        keep_syllable_separators=keep_syllable_separators,
    )

    if metadata and 'dataset' in metadata:
        stored_ck = metadata['dataset'].get('dict_checksum', '')
        if stored_ck:
            corpus.verify(stored_ck)
            logger.info("[OK] Checksum do dicionario verificado")

    # ── Split (reproduzir exato do treino) ────────────────────────────────────
    split_seed = 42
    split_test_ratio = 0.20
    split_val_ratio = 0.10
    use_stratify = True
    if metadata and 'config' in metadata:
        data_config = metadata['config'].get('data', metadata['config'])
        split_seed = data_config.get('split_seed', split_seed)
        split_test_ratio = data_config.get('test_ratio', split_test_ratio)
        split_val_ratio = data_config.get('val_ratio', split_val_ratio)
        use_stratify = data_config.get('stratify', True)

    split = corpus.split(
        test_ratio=split_test_ratio,
        val_ratio=split_val_ratio,
        seed=split_seed,
        stratify=use_stratify,
    )
    words, phonemes_list = split.test_pairs()
    words_raw = [corpus.words_raw[i] for i in split.test_indices]
    # phonemes_list são strings espaçadas ("k a ˈl ã").
    # NÃO usar ' '.join(p) — itera sobre caracteres e separa
    # combining marks (U+0303) da vogal base, inflando PER/WER.
    phonemes = phonemes_list

    char_vocab = corpus.char_vocab
    phoneme_vocab = corpus.phoneme_vocab

    # ── Log de contexto (verbose) ─────────────────────────────────────────────
    if verbose:
        dataset_meta = (metadata.get('dataset') if metadata else None) or split.metadata()
        cache_names = split.cache_filenames()
        cache_info = get_cache_info(
            DATA_DIR,
            train_name=cache_names["train"],
            val_name=cache_names["val"],
            test_name=cache_names["test"],
        )
        log_common_header(logger, "AVALIACAO", timestamp_str, device,
                          dataset_meta=dataset_meta, cache_info=cache_info)

        logger.info("=" * 70)
        logger.info("CONFIGURACAO DO MODELO")
        logger.info("=" * 70)
        logger.info("Arquivo: %s", model_path.name)
        if metadata:
            logger.info("Experimento: %s", metadata.get('experiment_name', 'N/A'))
            logger.info("Run ID: %s", metadata.get('run_id', 'N/A'))
            current_epoch = metadata.get('current_epoch', '?')
            total_epochs = metadata.get('total_epochs', '?')
            is_complete = metadata.get('training_completed', False)
            logger.info("Treino: Epoch %s/%s %s", current_epoch, total_epochs,
                        "[OK] Completo" if is_complete else "[!] Incompleto")
            logger.info("Treinado em: %s", metadata.get('timestamp', 'N/A'))
            logger.info("Best Loss: %s", metadata.get('best_loss', 'N/A'))
            if 'config' in metadata and 'model' in metadata['config']:
                mc = metadata['config']['model']
                logger.info("Embedding Type: %s", mc.get('embedding_type', 'learned'))
                logger.info("Embedding Dim: %s | Hidden: %s | Layers: %s | Dropout: %s",
                            mc.get('emb_dim', 'N/A'), mc.get('hidden_dim', 'N/A'),
                            mc.get('num_layers', 'N/A'), mc.get('dropout', 'N/A'))
            logger.info("Total Parametros: %s", f"{metadata.get('total_params', 'N/A'):,}")
        else:
            logger.info("[!] Metadados nao encontrados (modelo treinado com versao antiga)")
        logger.info("=" * 70)
        logger.info("")
    else:
        logger.info("Dataset de teste: %d palavras (seed=%d)", len(words), split_seed)

    # ── Carregar modelo ───────────────────────────────────────────────────────
    logger.info("=== Carregando modelo ===")
    model_config = metadata['config'] if (metadata and 'config' in metadata) else {}
    model = G2PLSTMModel.from_config(model_config, char_vocab, phoneme_vocab).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logger.info("[OK] Modelo carregado com sucesso")

    # ── Inferência ────────────────────────────────────────────────────────────
    logger.info("=== Fazendo predições ===")
    inference_start = time.perf_counter()
    predictions = []
    references = [p.split() for p in phonemes]

    for word in words:
        pred_indices = predict_word(model, word, char_vocab, device)
        predictions.append(phoneme_vocab.decode(pred_indices))

    inference_time = time.perf_counter() - inference_start
    samples_per_sec = len(words) / inference_time if inference_time > 0 else 0
    logger.info("[OK] Predicoes completas")
    logger.info("Tempo: %.2fs | Velocidade: %.1f palavras/s", inference_time, samples_per_sec)

    # ── Salvar predictions.tsv ────────────────────────────────────────────────
    base_name = get_base_name_from_path(model_path) or model_path.stem
    exp_name = base_name.split("__")[0]
    results_exp_dir = RESULTS_DIR / exp_name
    results_exp_dir.mkdir(exist_ok=True, parents=True)

    predictions_file = results_exp_dir / f"predictions_{base_name}.tsv"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        f.write("word\tpredicted\treference\tcorrect\n")
        for word, pred, ref in zip(words_raw, predictions, references):
            pred_str = ' '.join(pred)
            ref_str = ' '.join(ref)
            correct = '1' if pred == ref else '0'
            f.write(f"{word}\t{pred_str}\t{ref_str}\t{correct}\n")
    logger.info("Predicoes salvas: %s", predictions_file.name)

    # ── Salvar inference_stats.json ───────────────────────────────────────────
    total_time = time.perf_counter() - start_time
    stats_file = results_exp_dir / f"inference_stats_{base_name}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp_str,
            "experiment_name": metadata.get('experiment_name', exp_name) if metadata else exp_name,
            "inference_time_s": round(inference_time, 3),
            "samples_per_sec": round(samples_per_sec, 1),
            "total_words": len(words),
            "predictions_file": predictions_file.name,
        }, f, ensure_ascii=False, indent=2)
    logger.info("Stats de inferencia salvos: %s", stats_file.name)

    if verbose:
        logger.info("")
        logger.info("=" * 70)
        logger.info("INFERENCIA COMPLETA [OK]")
        logger.info("Tempo total: %.2fs", total_time)
        logger.info("Use analyze_errors.py --index N para metricas PER/WER/PanPhon")
        logger.info("=" * 70)
        logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# Batch: processar todos os modelos pendentes
# ─────────────────────────────────────────────────────────────────────────────

def process_all_models(force=False):
    """Processa todos os modelos que precisam de avaliação."""
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
        base_name = get_base_name_from_path(model_file) or model_file.stem
        exp_name = base_name.split("__")[0]
        predictions_file = RESULTS_DIR / exp_name / f"predictions_{base_name}.tsv"

        if predictions_file.exists() and not force:
            skipped.append(base_name)
            continue

        metadata = load_model_metadata(model_file)
        if metadata and not metadata.get('training_completed', False):
            logger.warning("[!] %s: treinamento incompleto, pulando", base_name)
            skipped.append(base_name)
            continue

        models_to_process.append((model_file, base_name))

    print(f"\nModelos a processar: {len(models_to_process)}")
    print(f"Modelos já processados (pulados): {len(skipped)}")

    if not models_to_process:
        print("\n[OK] Todos os modelos ja foram avaliados!")
        if skipped and not force:
            print("\nUse --force para reprocessar modelos já avaliados")
        return

    print("\nModelos que serão processados:")
    for _, base_name in models_to_process:
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
            run_inference_for_model(model_file, verbose=False)
            success_count += 1
            print(f"\n[OK] {base_name}: avaliacao completa")
        except Exception as e:
            error_count += 1
            logger.error("[ERRO] %s: erro durante avaliacao: %s", base_name, e)

    print(f"\n{'='*80}")
    print("RESUMO DO PROCESSAMENTO")
    print(f"{'='*80}")
    print(f"Sucesso: {success_count}/{len(models_to_process)}")
    print(f"Erros: {error_count}/{len(models_to_process)}")
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Inferência do modelo G2P LSTM sobre o test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python src/inference.py              # Usa o modelo mais recente
  python src/inference.py --list       # Lista todos os modelos disponíveis
  python src/inference.py --index 2    # Usa o modelo com índice 2
  python src/inference.py --model exp1_baseline_60split__20260309_202554
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
                        help='Simula decisão incremental sem executar inferência')

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    if args.all:
        process_all_models(args.force)
        return

    model_path = select_model(args)
    if model_path is None:
        return

    if args.dry_run:
        base_name = get_base_name_from_path(model_path) or model_path.stem
        exp_name = base_name.split("__")[0]
        predictions_file = RESULTS_DIR / exp_name / f"predictions_{base_name}.tsv"
        has_pred = predictions_file.exists()
        would_run = args.force or not has_pred
        print("\n[DRY RUN] inference.py")
        print(f"[DRY RUN] Modelo: {base_name}")
        print(f"[DRY RUN] Artefatos: predictions={'OK' if has_pred else 'MISSING'}")
        print(f"[DRY RUN] Decisao incremental: {'RODARIA' if would_run else 'PULARIA'}")
        if args.force:
            print("[DRY RUN] Modo FORCE: ativo")
        return

    run_inference_for_model(model_path, verbose=True)


if __name__ == "__main__":
    main()
