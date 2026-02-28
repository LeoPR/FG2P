#!/usr/bin/env python
"""Processa dicionário pt-br.tsv e cria train/test datasets"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import get_logger, DATA_DIR

logger = get_logger("prepare_data")

def load_tsv_dict(tsv_file):
    """Carrega dicionário do arquivo TSV"""
    words = []
    phonemes = []

    with open(tsv_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 2:
                continue

            word = parts[0].strip()
            phon_raw = parts[1].strip()

            # Remover pontos de separação de sílabas
            # "a . ˈ b a . k a . ˈ ʃ i" → "a ˈ b a k a ˈ ʃ i"
            phon = phon_raw.replace('. ', ' ').replace('.', '').strip()

            # Limpar múltiplos espaços
            phon = ' '.join(phon.split())

            if word and phon:
                words.append(word)
                phonemes.append(phon)

            if (i + 1) % 10000 == 0:
                logger.info(f"Processadas {i + 1} linhas...")

    return words, phonemes

def save_data(words, phonemes, output_file):
    """Salva dados em formato: palavra fonema1 fonema2 ...
    
    NOTA: EOS NÃO é escrito no arquivo — é um embedding interno
    adicionado automaticamente pelo G2PDataset.__getitem__().
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, phon in zip(words, phonemes):
            f.write(f"{word} {phon}\n")
    logger.info(f"Salvo {len(words)} linhas em: {output_file}")

def main():
    tsv_file = Path("dicts/pt-br.tsv")

    if not tsv_file.exists():
        logger.error(f"Arquivo {tsv_file} não encontrado!")
        return

    logger.info("=" * 60)
    logger.info("Processando dicionário português brasileiro")
    logger.info("=" * 60)

    logger.info(f"\nCarregando {tsv_file}...")
    words, phonemes = load_tsv_dict(tsv_file)

    logger.info(f"\nTotal de palavras carregadas: {len(words)}")
    logger.info("Amostra (primeiras 3):")
    for w, p in zip(words[:3], phonemes[:3]):
        logger.info(f"  {w:20} → {p}")

    logger.info("\nDividindo em treino (80%) e teste (20%)...")
    words_train, words_test, phon_train, phon_test = train_test_split(
        words, phonemes, test_size=0.2, random_state=42
    )

    logger.info(f"Treino: {len(words_train)} palavras")
    logger.info(f"Teste:  {len(words_test)} palavras")

    logger.info("\nSalvando arquivos...")
    train_file = DATA_DIR / "train.txt"
    test_file = DATA_DIR / "test.txt"

    save_data(words_train, phon_train, train_file)
    save_data(words_test, phon_test, test_file)

    logger.info("\n" + "=" * 60)
    logger.info("✓ Pronto para treinar!")
    logger.info("Execute: python src/train.py")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
