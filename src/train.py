#!/usr/bin/env python
import argparse
import csv
import time
import json
from datetime import datetime
from pathlib import Path
import torch
from torch.optim import Adam
from utils import get_logger, DATA_DIR, get_cache_info, log_common_header
from g2p import G2PCorpus, G2PLSTMModel
from file_registry import FileRegistry
from losses import get_loss_function, build_panphon_features

logger = get_logger("train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Otimizações NVIDIA CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True      # Auto-seleciona kernels otimizados
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32: ~20-30% speedup em Ampere (RTX 30xx+)
    torch.backends.cudnn.allow_tf32 = True         # TF32 para operações LSTM via cuDNN
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})")
    logger.info(f"cuDNN benchmark: {torch.backends.cudnn.benchmark} | TF32: {torch.backends.cuda.matmul.allow_tf32}")

def load_config(config_path="config.json"):
    """Carrega configurações de treinamento de arquivo JSON externo.
    
    Args:
        config_path: Caminho para o arquivo de configuração (default: config.json na raiz)
    
    Returns:
        dict: Configurações de treinamento
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_file}\n"
            f"Crie um config.json na raiz ou especifique via --config"
        )
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validação básica da estrutura
    required_keys = ['data', 'model', 'training', 'experiment']
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Config inválido. Chaves faltando: {missing_keys}")
    
    return config

def train_epoch(model, train_dl, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0

    for chars, phonemes, char_lengths in train_dl:
        chars = chars.to(device)
        phonemes = phonemes.to(device)
        char_lengths = char_lengths.to(device)

        # O modelo recebe src, src_lengths e targets (fonemas com EOS)
        logits = model(chars, char_lengths, phonemes, teacher_forcing_ratio)
        # logits: (batch, target_len, vocab_size)

        # Interface unificada: todas as losses aceitam (batch, seq, vocab)
        loss = criterion(logits, phonemes)

        optimizer.zero_grad(set_to_none=True)  # set_to_none evita alocação de tensor zero (~3-8% speedup)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_dl)

def eval_epoch(model, val_dl, criterion, device):
    model.eval()
    total_loss = 0

    with torch.inference_mode():  # Fase 1: Inference mode é mais otimizado que no_grad
        for chars, phonemes, char_lengths in val_dl:
            chars = chars.to(device)
            phonemes = phonemes.to(device)
            char_lengths = char_lengths.to(device)

            # Na avaliação, teacher_forcing=1.0 (sempre usa o target real)
            logits = model(chars, char_lengths, phonemes, teacher_forcing_ratio=1.0)

            # Interface unificada: todas as losses aceitam (batch, seq, vocab)
            loss = criterion(logits, phonemes)
            total_loss += loss.item()

    return total_loss / len(val_dl)

def main():
    # Parse argumentos CLI
    parser = argparse.ArgumentParser(
        description="Treinar modelo G2P LSTM com configurações externas"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Caminho para arquivo de configuração JSON (default: config.json)"
    )
    args = parser.parse_args()
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Carregar configurações do arquivo externo
    config = load_config(args.config)
    logger.info(f"Configurações carregadas de: {args.config}")
    
    # Fonte primária do dataset (do config)
    DICT_PATH = Path(config["data"]["source"])

    logger.info("=== Carregando corpus ===")
    logger.info(f"Fonte primária: {DICT_PATH}")
    grapheme_encoding = config["data"].get("grapheme_encoding", "raw")
    keep_syllable_separators = config["data"].get("keep_syllable_separators", False)
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

    logger.info("")
    # Ler parâmetro stratify da config (padrão: True para estratificado)
    use_stratify = config["data"].get("stratify", True)
    split_type = "estratificado" if use_stratify else "aleatorio simples"
    logger.info(f"=== Dividindo treino/val/teste ({split_type}) ===")
    split = corpus.split(
        test_ratio=config["data"]["test_ratio"],
        val_ratio=config["data"]["val_ratio"],
        seed=config["data"]["split_seed"],
        cache_dir=DATA_DIR,
        stratify=use_stratify,
    )

    report = split.quality_report()
    dataset_meta = split.metadata(report=report)
    cache_names = split.cache_filenames()
    cache_info = get_cache_info(
        DATA_DIR,
        train_name=cache_names["train"],
        val_name=cache_names["val"],
        test_name=cache_names["test"],
    )
    log_common_header(
        logger,
        "TREINAMENTO",
        timestamp_str,
        device,
        dataset_meta=dataset_meta,
        cache_info=cache_info,
    )
    split.log_quality(report=report)

    logger.info("")
    logger.info("=== Criando dataloaders ===")
    train_dl, val_dl, _test_dl = split.get_dataloaders(
        batch_size=config["training"]["batch_size"]
    )
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info("Nota: test_dl NAO é usado durante treino (avaliação final apenas).")

    char_vocab = corpus.char_vocab
    phoneme_vocab = corpus.phoneme_vocab

    logger.info("")
    logger.info("=== Criando modelo ===")
    
    model = G2PLSTMModel.from_config(config, char_vocab, phoneme_vocab).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Arquitetura: %s | emb=%d hidden=%d layers=%d dropout=%.2f",
        model.__class__.__name__,
        config["model"]["emb_dim"],
        config["model"]["hidden_dim"],
        config["model"]["num_layers"],
        config["model"]["dropout"]
    )
    logger.info(f"Total de parâmetros: {total_params:,}")
    logger.info(f"Device: {device}")
    logger.info(f"Regularização: dropout={config['model']['dropout']}, weight_decay={config['training']['weight_decay']}")

    loss_type = config.get('training', {}).get('loss', {}).get('type', 'cross_entropy')
    
    if loss_type in ('distance_aware', 'soft_target'):
        panphon_features = build_panphon_features(phoneme_vocab)
        loss_config = config.get('training', {}).get('loss', {}).get('config', {})
        criterion = get_loss_function(
            loss_type=loss_type,
            phoneme_vocab=phoneme_vocab.p2i,
            panphon_features=panphon_features,
            config=loss_config
        )
        logger.info("[OK] Loss: %s (config=%s)", loss_type, loss_config)
        
    else:
        # cross_entropy (default, Exp0-5 compat) — via factory com interface unificada
        criterion = get_loss_function('cross_entropy')
        logger.info("[OK] Loss: Cross-Entropy (standard)")

    # Garantir que buffers da loss estejam no mesmo device do modelo
    criterion = criterion.to(device)
    
    optimizer = Adam(
        model.parameters(), 
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]  # L2 regularization
    )

    # Criar FileRegistry para gerenciar nomes de arquivos
    # Usa experiment.name do config como nome base
    file_registry = FileRegistry(config)
    experiment_name = config["experiment"]["name"]
    run_id = file_registry.timestamp
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("CONFIGURACAO DO TREINO")
    logger.info("=" * 70)
    logger.info(f"Experimento: {experiment_name}")
    logger.info(f"Nome completo: {file_registry.base_name}")
    logger.info(f"Timestamp: {timestamp_str}")
    logger.info("=" * 70)
    logger.info("")

    logger.info("=== Treinando ===")
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0  # Early stopping counter
    patience = config["training"]["early_stopping_patience"]
    warmup_epochs = config["training"].get("warmup_epochs", 0)  # NOVO: warmup
    warmup_notified = False  # Flag para avisar apenas uma vez

    # CSV incremental: abrir arquivo agora e escrever linha por linha
    history_file = file_registry.get_history_path()
    csv_file = open(history_file, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'epoch_start_ts', 'epoch_end_ts'])
    csv_file.flush()  # Garantir escrita imediata do header

    if warmup_epochs > 0:
        logger.info("[!] WARMUP MODE: Early stopping desabilitado ate epoch %d", warmup_epochs)
        logger.info("    Apos epoch %d: early stopping com patience=%d", warmup_epochs, patience)
    
    logger.info("[CSV] Historico sendo escrito em: results/%s", history_file.name)
    
    train_size = len(train_dl.dataset)
    start_time = time.perf_counter()

    for epoch in range(config["training"]["epochs"]):
        epoch_start = time.perf_counter()
        epoch_start_ts = time.time()  # Unix timestamp para o CSV
        
        train_start = time.perf_counter()
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        train_time = time.perf_counter() - train_start

        eval_start = time.perf_counter()
        val_loss = eval_epoch(model, val_dl, criterion, device)
        eval_time = time.perf_counter() - eval_start
        epoch_end_ts = time.time()  # Unix timestamp para o CSV

        epoch_time = time.perf_counter() - epoch_start
        samples_per_sec = (train_size / train_time) if train_time > 0 else 0.0

        # Escrever no CSV incrementalmente (nao manter em memoria)
        csv_writer.writerow([epoch + 1, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{epoch_start_ts:.3f}', f'{epoch_end_ts:.3f}'])
        csv_file.flush()  # Flush imediato: se treino cair, dados estão salvos

        logger.info(
            "Epoch %3d | Train: %.4f | Val: %.4f | time: %.1fs (train %.1fs, eval %.1fs) | speed: %.0f samples/s",
            epoch + 1,
            train_loss,
            val_loss,
            epoch_time,
            train_time,
            eval_time,
            samples_per_sec
        )

        # Notifica warmup completion apenas uma vez
        if warmup_epochs > 0 and (epoch + 1) == warmup_epochs and not warmup_notified:
            logger.info(
                "[OK] Warmup concluido (epoch %d). Early stopping ativo (patience=%d).",
                warmup_epochs, patience
            )
            warmup_notified = True

        in_warmup = (epoch + 1) <= warmup_epochs

        if val_loss < best_loss:
            if not in_warmup and patience_counter > 0:
                logger.info("   Early stopping: contador resetado (novo melhor modelo)")
            best_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            # Salvar modelo com nome descritivo
            model_path = file_registry.get_model_path()
            torch.save(model.state_dict(), model_path)

            # Salvar metadados do modelo (inclui dados do split para reproducao)
            metadata = {
                "experiment_name": experiment_name,
                "run_id": run_id,
                "full_name": file_registry.base_name,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "current_epoch": epoch + 1,
                "total_epochs": config["training"]["epochs"],
                "config": config,
                "best_loss": float(best_loss),
                "training_completed": False,
                "device": str(device),
                "total_params": total_params,
                "dataset": split.metadata(),
                "loss_type": loss_type,
                "loss_config": config.get('training', {}).get('loss', {}).get('config', {}),
            }
            metadata_path = file_registry.get_metadata_path()
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info("[OK] Melhor modelo salvo: %s (val_loss: %.4f)", model_path.name, val_loss)

        elif not in_warmup:
            patience_counter += 1
            if patience_counter == 1:
                logger.info("   Early stopping: contador iniciado (patience=%d)", patience)
            if patience_counter >= patience:
                logger.info("")
                logger.info("=" * 70)
                logger.info("[STOP] EARLY STOPPING ativado!")
                logger.info("Val loss nao melhorou por %d epochs consecutivos", patience)
                logger.info("Melhor modelo: epoch %d com val_loss %.4f", best_epoch, best_loss)
                logger.info("Parando treinamento no epoch %d...", epoch + 1)
                logger.info("=" * 70)
                logger.info("")
                break

    # Fechar CSV file
    csv_file.close()

    # Ler CSV para obter contagem de épocas
    with open(history_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header
        total_epochs_trained = sum(1 for _ in csv_reader)

    total_time = time.perf_counter() - start_time
    avg_epoch_time = total_time / max(1, total_epochs_trained)
    
    # Marcar treino como completo nos metadados
    metadata_path = file_registry.get_metadata_path()
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        metadata['training_completed'] = True
        metadata['final_epoch'] = total_epochs_trained
        metadata['total_time_seconds'] = total_time
        metadata['avg_epoch_time'] = avg_epoch_time
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info("===" + "="*60)
    logger.info("TREINAMENTO COMPLETO [OK]")
    logger.info("===" + "="*60)
    logger.info(f"Melhor loss: {best_loss:.4f} (epoch {best_epoch})")
    logger.info(f"Tempo total: {total_time:.1f}s ({total_time/60:.1f}m) | Média: {avg_epoch_time:.1f}s/epoch")
    logger.info("")
    logger.info("Nota: usar inference.py para avaliação final no TEST SET (nunca visto no treino).")
    logger.info("")
    logger.info("Arquivos gerados:")
    logger.info(f"  Modelo:    models/{file_registry.base_name}.pt")
    logger.info(f"  Metadados: models/{file_registry.base_name}_metadata.json")
    logger.info(f"  Histórico: results/{file_registry.base_name}_history.csv")
    logger.info(f"  Summary:   results/{file_registry.base_name}_summary.txt")
    logger.info("===" + "="*60)

    # CSV já foi escrito incrementalmente, não precisa reescrever

    summary_file = file_registry._results_exp_dir() / f"{file_registry.base_name}_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("G2P LSTM Training Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Full Name: {file_registry.base_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"Embedding: {config['model'].get('embedding_type', 'learned')}\n")
        f.write(f"Optimizer: {config['training'].get('fit_method', 'adam')}\n")
        f.write(f"Train/Val/Test Split: {config['data']['train_ratio']:.0%} / {config['data']['val_ratio']:.0%} / {config['data']['test_ratio']:.0%}\n")
        f.write("-" * 60 + "\n")
        f.write(
            "Model: %s | BiLSTM | emb=%d hidden=%d layers=%d dropout=%.2f\n"
            % (
                model.__class__.__name__,
                config["model"]["emb_dim"],
                config["model"]["hidden_dim"],
                config["model"]["num_layers"],
                config["model"]["dropout"]
            )
        )
        f.write(f"Total params: {total_params:,}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Train samples: {train_size:,}\n")
        f.write(f"Val samples: {len(val_dl.dataset):,}\n")
        f.write(f"Test samples (held-out): {len(_test_dl.dataset):,}\n")
        f.write(f"Batch size: {config['training']['batch_size']}\n")
        f.write(f"Learning rate: {config['training']['lr']}\n")
        f.write(f"Epochs trained: {total_epochs_trained}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Best val loss: {best_loss:.6f} (epoch {best_epoch})\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)\n")
        f.write(f"Avg epoch time: {avg_epoch_time:.1f}s\n")
        samples_per_sec_avg = (train_size / (total_time/total_epochs_trained)) if total_epochs_trained > 0 else 0
        f.write(f"Training speed (avg): {samples_per_sec_avg:.0f} samples/s\n")
        f.write("=" * 60 + "\n")
        f.write("Generated files:\n")
        f.write(f"  Model:    models/{experiment_name}.pt\n")
        f.write(f"  History:  results/{experiment_name}_history.csv\n")
        f.write(f"  Summary:  results/{experiment_name}_summary.txt\n")
        f.write("\nUse 'python src/inference.py' para avaliacao final no test set.\n")


if __name__ == "__main__":
    main()
