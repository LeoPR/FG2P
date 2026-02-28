#!/usr/bin/env python
"""
Análise de erros fonéticos — PanPhon graduated metrics.

Analisa predições geradas por inference.py usando metadata dos modelos.

Métricas geradas:
  - WER Graduado (acertos parciais ponderados por similaridade)
  - PER Ponderado (distância articulatória média)
  - Distribuição de classes A/B/C/D (por fonema)
  - WER segmentado por classe de pior erro na palavra
  - Top confusões fonéticas com distância articulatória
  - Exemplos por classe de erro

Uso:
  python src/analyze_errors.py                   # Usa modelo mais recente
  python src/analyze_errors.py --list            # Lista modelos disponíveis
  python src/analyze_errors.py --index 0         # Usa modelo com índice 0
  python src/analyze_errors.py --model MODEL     # Usa modelo específico

Dependências:
  - panphon (pip install panphon)
  - phonetic_features.py (presente no projeto)
  - g2p.py (vocabulários)
"""

import argparse
import json
import sys
import time
import editdistance
from collections import Counter
from datetime import datetime
from pathlib import Path

from utils import get_logger, RESULTS_DIR, MODELS_DIR, get_model_summary
from g2p import G2PCorpus
from file_registry import get_base_name_from_path
from phonetic_features import PhoneticSpace, graduated_metrics, load_phoneme_map

DICT_PATH = Path("dicts/pt-br.tsv")
logger = get_logger("analyze_errors")


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


def list_available_models():
    """Lista todos os modelos com predições disponíveis"""
    models = []
    
    for metadata_file in sorted(MODELS_DIR.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime):
        try:
            model_name = metadata_file.stem.replace("_metadata", "")
            summary = get_model_summary(model_name)
            
            if not summary or not summary.get("metadata"):
                continue
            
            metadata = summary["metadata"]
            artifacts = summary["artifacts"]
            
            # Verificar se tem predictions
            preds = [p for p in artifacts.get("predictions", []) if model_name in p.stem]
            
            models.append({
                "index": len(models),
                "name": model_name,
                "metadata": metadata,
                "predictions_file": preds[0] if preds else None,
                "has_predictions": bool(preds),
                "mtime": metadata_file.stat().st_mtime
            })
        except Exception as e:
            logger.warning(f"Erro ao carregar {metadata_file}: {e}")
            continue
    
    if not models:
        print("\n❌ Nenhum modelo encontrado em models/")
        print("Execute train.py e inference.py primeiro.\n")
        return []
    
    print("\n" + "=" * 80)
    print("MODELOS COM PREDIÇÕES DISPONÍVEIS")
    print("=" * 80)
    
    for model in models:
        metadata = model["metadata"]
        exp_name = metadata.get("experiment_name", model["name"])
        timestamp = metadata.get("timestamp", "N/A")
        is_complete = metadata.get("training_completed", False)
        status = "✓ Completo" if is_complete else "⚠ Incompleto"
        
        pred_status = "✓ Predições disponíveis" if model["has_predictions"] else "❌ Sem predições"
        
        print(f"\n[{model['index']}] {exp_name}")
        print(f"    Arquivo: {model['name']}")
        print(f"    Treinado em: {timestamp}")
        print(f"    Status: Epoch {metadata.get('current_epoch', '?')}/{metadata.get('total_epochs', '?')} {status}")
        
        if model["has_predictions"]:
            pred_file = model["predictions_file"]
            print(f"    Predições: {pred_file.name}")
        else:
            print(f"    {pred_status} — Execute inference.py primeiro")
    
    print("\n" + "=" * 80)
    print(f"Total: {len(models)} modelo(s)")
    
    # Mostrar apenas os que tem predições
    with_preds = [m for m in models if m["has_predictions"]]
    print(f"Com predições: {len(with_preds)} modelo(s)")
    
    if with_preds:
        latest = with_preds[-1]
        print(f"Mais recente: [{latest['index']}] {latest['name']}")
    
    print("=" * 80)
    print("\nUso:")
    print("  python src/analyze_errors.py --index N    # Analisa modelo com índice N")
    print("  python src/analyze_errors.py --list       # Mostra esta lista novamente\n")
    
    return models


def select_model(args):
    """Seleciona modelo baseado nos argumentos"""
    models = []
    
    for metadata_file in sorted(MODELS_DIR.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime):
        try:
            model_name = metadata_file.stem.replace("_metadata", "")
            summary = get_model_summary(model_name)
            
            if not summary or not summary.get("metadata"):
                continue
            
            artifacts = summary["artifacts"]
            preds = [p for p in artifacts.get("predictions", []) if model_name in p.stem]
            
            if preds:  # Apenas modelos com predições
                models.append({
                    "index": len(models),
                    "name": model_name,
                    "predictions_file": preds[0],
                    "mtime": metadata_file.stat().st_mtime
                })
        except:
            continue
    
    if not models:
        logger.error("Nenhum modelo com predições encontrado.")
        logger.info("Execute inference.py primeiro para gerar predições.")
        logger.info("Use --list para ver todos os modelos")
        return None
    
    # Seleção por --index
    if args.index is not None:
        if 0 <= args.index < len(models):
            return models[args.index]["predictions_file"]
        else:
            logger.error(f"Índice {args.index} inválido. Modelos disponíveis: 0-{len(models) - 1}")
            logger.info("Use --list para ver modelos disponíveis")
            return None
    
    # Seleção por --model NAME
    if args.model:
        for m in models:
            if args.model in m["name"]:
                return m["predictions_file"]
        logger.error(f"Modelo '{args.model}' não encontrado ou sem predições")
        logger.info("Use --list para ver modelos disponíveis")
        return None
    
    # Padrão: mais recente
    latest = models[-1]
    logger.info(f"Usando modelo mais recente: {latest['name']}")
    return latest["predictions_file"]


def load_predictions(filepath: Path) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Carrega predições do TSV gerado por inference.py.
    
    Returns:
        (words, predictions, references) — cada pred/ref é lista de fonemas
    """
    words = []
    predictions = []
    references = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()  # skip header
        if not header.startswith('word\t'):
            raise ValueError(f"Formato inesperado: {header.strip()}")
        
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            word = parts[0]
            pred = parts[1].split()
            ref = parts[2].split()
            words.append(word)
            predictions.append(pred)
            references.append(ref)
    
    return words, predictions, references


def calculate_per(predictions, references):
    """Calcula PER (Phoneme Error Rate) clássico."""
    total_errors = 0
    total_phonemes = 0
    for pred, ref in zip(predictions, references):
        dist = editdistance.eval(pred, ref)
        total_errors += dist
        total_phonemes += len(ref)
    return (total_errors / total_phonemes * 100) if total_phonemes > 0 else 0


def calculate_accuracy(predictions, references):
    """Calcula Word-level Accuracy."""
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return (correct / len(predictions) * 100) if predictions else 0


def analyze_length_distribution(words, predictions, references):
    """Analisa distribuição de comprimentos e detecta truncation/over-generation"""
    anomalies = {
        'truncated': [],      # pred muito menor que ref
        'overgenerated': [],  # pred muito maior que ref
        'stats': {}
    }
    
    len_diffs = []
    for i, (word, pred, ref) in enumerate(zip(words, predictions, references)):
        pred_len = len(pred)
        ref_len = len(ref)
        diff = pred_len - ref_len
        len_diffs.append(diff)
        
        # Truncation: pred < ref - 3 fonemas
        if diff <= -3:
            anomalies['truncated'].append({
                'word': word,
                'pred': pred,
                'ref': ref,
                'pred_len': pred_len,
                'ref_len': ref_len,
                'diff': diff
            })
        
        # Over-generation: pred > ref + 3 fonemas
        elif diff >= 3:
            anomalies['overgenerated'].append({
                'word': word,
                'pred': pred,
                'ref': ref,
                'pred_len': pred_len,
                'ref_len': ref_len,
                'diff': diff
            })
    
    # Estatísticas
    import numpy as np
    anomalies['stats'] = {
        'mean_diff': np.mean(len_diffs),
        'std_diff': np.std(len_diffs),
        'median_diff': np.median(len_diffs),
        'min_diff': min(len_diffs),
        'max_diff': max(len_diffs),
        'total_truncated': len(anomalies['truncated']),
        'total_overgenerated': len(anomalies['overgenerated'])
    }
    
    return anomalies


def _max_consecutive_ngram(seq, ngram_size):
    """Retorna dict {ngram: max_repetições_consecutivas} para repeats ≥ 2."""
    result = {}
    if len(seq) < ngram_size * 2:
        return result
    seen = set()
    for j in range(len(seq) - ngram_size + 1):
        ngram = tuple(seq[j:j+ngram_size])
        if ngram in seen:
            continue
        seen.add(ngram)
        consecutive = 1
        pos = j + ngram_size
        while pos + ngram_size <= len(seq):
            if tuple(seq[pos:pos+ngram_size]) == ngram:
                consecutive += 1
                pos += ngram_size
            else:
                break
        if consecutive >= 2:
            result[ngram] = consecutive
    return result


def _normalize_graphemes(word):
    """Normaliza grafemas: lowercase, remove acentos, filtra não-alfa."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', word.lower())
    return [c for c in nfkd if unicodedata.category(c) != 'Mn' and c.isalpha()]


def detect_hallucinations(words, predictions, references):
    """Detecta padrões de alucinação em RNNs (repetições excessivas/anômalas).
    
    Lógica: compara repetitividade da PALAVRA ORIGINAL (grafemas) com a PREDIÇÃO.
    Se a predição tem mais repetições consecutivas de n-grams que o nível natural
    da palavra, é provável alucinação (LSTM preso em loop).
    
    Pré-condição: pred != ref (predições corretas não são alucinações).
    
    Exemplos:
      "ururaí" → grafemas u,r,u,r,a,i: bigram (u,r)×2 natural.
                 pred u,ɾ,u,ɾ,a,ˈ,i: bigram (u,ɾ)×2 → 2 ≤ 2 → OK ✓
      "sussurrando" → pred == ref → skip (predição correta) ✓
      "político-administrativas" → grafemas sem repetição consecutiva (max=1).
                 pred tem trigram (t,ɾ,i)×9 → 9 > 1 → ALUCINAÇÃO ✗
    """
    hallucinations = []
    
    for i, (word, pred, ref) in enumerate(zip(words, predictions, references)):
        # Predição correta não é alucinação
        if pred == ref:
            continue
        
        pred_str = ' '.join(pred)
        patterns = []
        
        # 1. Comparar repetitividade: grafemas + ref vs pred
        #    Baseline = max(repetitividade dos grafemas, repetitividade da ref)
        #    Grafemas capturam repetições naturais da palavra (ururaí: u,r,u,r)
        #    Ref captura mapeamentos onde grafema→fonema cria repetições (digi→ʒiʒi)
        graphemes = _normalize_graphemes(word)
        
        for ngram_size in [2, 3]:
            # Baseline: nível natural de repetição
            word_repeats = _max_consecutive_ngram(graphemes, ngram_size)
            ref_repeats = _max_consecutive_ngram(ref, ngram_size)
            word_max = max(word_repeats.values(), default=1)
            ref_max = max(ref_repeats.values(), default=1)
            baseline = max(word_max, ref_max)
            
            # Repetições na predição
            pred_repeats = _max_consecutive_ngram(pred, ngram_size)
            
            for ngram, pred_count in pred_repeats.items():
                if pred_count > baseline:
                    ngram_str = ' '.join(ngram)
                    label = "bigram" if ngram_size == 2 else "trigram"
                    patterns.append(
                        f"{label}_loop:{ngram_str} ×{pred_count} "
                        f"(palavra: ×{word_max}, ref: ×{ref_max})"
                    )
        
        # 2. Explosão de fonema: pred tem um char muito mais vezes que ref
        #    Threshold: pred_count > 2×ref_count E pred_count - ref_count >= 4
        from collections import Counter
        pred_counts = Counter(pred)
        ref_counts = Counter(ref)
        for char in pred_counts:
            pc = pred_counts[char]
            rc = ref_counts.get(char, 0)
            if pc > 2 * max(rc, 1) and (pc - rc) >= 4:
                patterns.append(f"char_explosion:{char} ×{pc} (ref: ×{rc})")
        
        # 3. Over-generation severa: pred > 2× comprimento ref
        if len(pred) > 2 * len(ref) and len(ref) > 0:
            patterns.append(f"length_explosion: pred={len(pred)} ref={len(ref)}")
        
        if patterns:
            hallucinations.append({
                'word': word,
                'pred': pred,
                'ref': ref,
                'pred_str': pred_str,
                'ref_str': ' '.join(ref),
                'patterns': patterns,
                'pred_len': len(pred),
                'ref_len': len(ref),
            })
    
    return hallucinations


def main():
    parser = argparse.ArgumentParser(
        description='Análise fonética detalhada de erros G2P (PanPhon)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python src/analyze_errors.py              # Usa modelo mais recente
  python src/analyze_errors.py --list       # Lista modelos com predições
  python src/analyze_errors.py --index 0    # Analisa modelo com índice 0
  python src/analyze_errors.py --model exp2 # Analisa modelo específico
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='Lista todos os modelos com predições e sai')
    parser.add_argument('--index', type=int, metavar='N',
                       help='Usa o modelo com índice N da lista')
    parser.add_argument('--model', type=str, metavar='NAME',
                       help='Nome do modelo (busca parcial)')
    
    args = parser.parse_args()
    
    # Se --list, apenas lista e sai
    if args.list:
        list_available_models()
        return
    
    # Selecionar arquivo de predições
    pred_path = select_model(args)
    if pred_path is None:
        return

    base_name = get_base_name_from_path(pred_path)
    if not base_name:
        base_name = pred_path.stem
        if base_name.startswith("predictions_"):
            base_name = base_name[len("predictions_"):]
    metadata = load_model_metadata(MODELS_DIR / f"{base_name}.pt")
    
    start_time = time.perf_counter()
    
    # =========================================================================
    # 1. Carregar predições
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("ANÁLISE DE ERROS FONÉTICOS (PanPhon)")
    logger.info("=" * 70)
    logger.info(f"Arquivo: {pred_path.name}")
    
    words, predictions, references = load_predictions(pred_path)
    total_words = len(words)
    correct_count = sum(1 for p, r in zip(predictions, references) if p == r)
    error_count = total_words - correct_count
    
    logger.info(f"Total: {total_words} palavras | Corretas: {correct_count} | Erros: {error_count}")
    logger.info("")
    
    # =========================================================================
    # 2. Métricas clássicas (referência)
    # =========================================================================
    per = calculate_per(predictions, references)
    accuracy = calculate_accuracy(predictions, references)
    wer = 100.0 - accuracy
    
    logger.info("MÉTRICAS CLÁSSICAS (referência)")
    logger.info("-" * 40)
    logger.info(f"  PER:      {per:.2f}%")
    logger.info(f"  WER:      {wer:.2f}%")
    logger.info(f"  Accuracy: {accuracy:.2f}%")
    logger.info("")
    
    # =========================================================================
    # 3. Construir espaço fonético e calcular métricas graduadas
    # =========================================================================
    logger.info("Construindo espaço fonético (PanPhon)...")

    grapheme_encoding = "raw"
    keep_syllable_separators = False
    if metadata and 'config' in metadata:
        data_config = metadata['config'].get('data', metadata['config'])
        grapheme_encoding = data_config.get('grapheme_encoding', 'raw')
        keep_syllable_separators = data_config.get('keep_syllable_separators', False)

    logger.info("Codificação grafêmica: %s", grapheme_encoding)
    logger.info(
        "Separadores de sílaba: %s",
        "manter" if keep_syllable_separators else "remover",
    )

    # Reconstruir vocabulário a partir do dicionário
    corpus = G2PCorpus(
        DICT_PATH,
        grapheme_encoding=grapheme_encoding,
        keep_syllable_separators=keep_syllable_separators,
    )
    # Precisamos do split para reconstruir o vocab exato — mas o vocab é do corpus todo
    phoneme_vocab = corpus.phoneme_vocab
    
    ps = PhoneticSpace(phoneme_vocab)
    phoneme_map = load_phoneme_map()
    
    # Converter predictions/references de strings para índices
    pred_indices = [
        [phoneme_vocab.p2i.get(ph, 1) for ph in pred]
        for pred in predictions
    ]
    ref_indices = [
        [phoneme_vocab.p2i.get(ph, 1) for ph in ref]
        for ref in references
    ]
    
    # Calcular métricas graduadas
    grad = graduated_metrics(
        pred_indices, ref_indices, ps,
        phoneme_vocab=phoneme_vocab,
        phoneme_map=phoneme_map,
        words=words,
    )
    
    # =========================================================================
    # 4. Métricas graduadas (PanPhon)
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("MÉTRICAS GRADUADAS (PanPhon)")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"  WER Graduado:  {grad['wer_graduated']:.2f}%")
    logger.info(f"    → Ajustado por acertos parciais (classes A/B/C)")
    logger.info(f"  PER Ponderado: {grad['per_weighted']:.2f}%")
    logger.info(f"    → Ponderado por distância articulatória")
    logger.info("")
    
    # Comparação direta clássico vs graduado
    logger.info("  Comparação:")
    logger.info(f"    WER clássico {wer:.2f}% → WER graduado {grad['wer_graduated']:.2f}%  (delta {wer - grad['wer_graduated']:+.2f}%)")
    logger.info(f"    PER clássico {per:.2f}% → PER ponderado {grad['per_weighted']:.2f}%  (delta {per - grad['per_weighted']:+.2f}%)")
    logger.info("")
    
    # =========================================================================
    # 5. Distribuição de classes (por fonema)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("DISTRIBUIÇÃO DE CLASSES (por fonema)")
    logger.info("=" * 70)
    logger.info("  A = Exato (distância 0.000)")
    logger.info("  B = Quase-idêntico (≤1 feature, dist ≤ 0.125)")
    logger.info("  C = Mesma família (2-3 features, dist ≤ 0.250)")
    logger.info("  D = Distante (4+ features, dist > 0.250)")
    logger.info("")
    
    for cls in ['A', 'B', 'C', 'D']:
        count = grad['error_distribution'].get(cls, 0)
        pct = grad['class_proportions'].get(cls, 0.0)
        bar = '█' * int(pct / 2) if pct >= 1 else ''
        logger.info(f"  Classe {cls}: {count:6d} ({pct:5.2f}%) {bar}")
    logger.info("")
    
    # Qualidade dos erros
    total_errors_phoneme = sum(grad['error_distribution'].get(c, 0) for c in ['B', 'C', 'D'])
    if total_errors_phoneme > 0:
        smart = grad['error_distribution'].get('B', 0)
        medium = grad['error_distribution'].get('C', 0)
        grave = grad['error_distribution'].get('D', 0)
        logger.info("  Qualidade dos erros:")
        logger.info(f"    B (leves):  {smart:5d} ({smart/total_errors_phoneme*100:.1f}%)")
        logger.info(f"    C (médios): {medium:5d} ({medium/total_errors_phoneme*100:.1f}%)")
        logger.info(f"    D (graves): {grave:5d} ({grave/total_errors_phoneme*100:.1f}%)")
        
        if smart / total_errors_phoneme > 0.7:
            logger.info("    ✓ Modelo erra de forma INTELIGENTE (confunde fonemas similares)")
        elif smart / total_errors_phoneme > 0.5:
            logger.info("    → Modelo mostra conhecimento fonético parcial")
        else:
            logger.info("    ⚠ Erros parecem mais aleatórios")
    logger.info("")
    
    # =========================================================================
    # 6. WER segmentado por classe de erro (por palavra)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("WER SEGMENTADO POR CLASSE DE ERRO")
    logger.info("=" * 70)
    logger.info("  Cada palavra é classificada pela PIOR classe de erro entre seus fonemas.")
    logger.info("")
    
    word_scores = grad.get('graduated_word_scores', [])
    if word_scores:
        words_by_class = {'A': [], 'B': [], 'C': [], 'D': []}
        for ws in word_scores:
            words_by_class[ws['worst_class']].append(ws)
        
        class_labels = {
            'A': 'Exata (sem erro)',
            'B': 'Erro leve (quase-idêntico)',
            'C': 'Erro médio (mesma família)',
            'D': 'Erro grave (categorias diferentes)',
        }
        
        logger.info(f"  {'Classe':<8} {'Descrição':<35} {'Palavras':>8} {'%':>7} {'WER Parcial':>12}")
        logger.info("  " + "-" * 72)
        
        for cls in ['A', 'B', 'C', 'D']:
            n = len(words_by_class[cls])
            pct = (n / total_words * 100) if total_words > 0 else 0
            # WER parcial: contribuição desta classe ao WER total
            wer_contribution = (n / total_words * 100) if total_words > 0 else 0
            if cls == 'A':
                wer_contribution = 0.0  # exatas não contribuem ao WER
            logger.info(
                f"  {cls:<8} {class_labels[cls]:<35} {n:8d} {pct:6.2f}% {wer_contribution:11.2f}%"
            )
        
        logger.info("")
        logger.info(f"  Total WER (soma B+C+D):  {wer:.2f}%")
        
        # Score médio por classe
        logger.info("")
        logger.info("  Score médio por classe (1.0 = perfeito, 0.0 = totalmente errado):")
        for cls in ['A', 'B', 'C', 'D']:
            class_words = words_by_class[cls]
            if class_words:
                avg_score = sum(ws['score'] for ws in class_words) / len(class_words)
                logger.info(f"    Classe {cls}: {avg_score:.3f} ({len(class_words)} palavras)")
    logger.info("")
    
    # =========================================================================
    # 7. Top confusões fonéticas
    # =========================================================================
    logger.info("=" * 70)
    logger.info("TOP-10 CONFUSÕES FONÉTICAS (com distância articulatória)")
    logger.info("=" * 70)
    logger.info(f"  {'Ref':>4} → {'Pred':<4}  {'Count':>5}  {'Dist':>6}  {'Classe':>6}  Análise")
    logger.info("  " + "-" * 65)
    
    for conf in grad['phonetic_confusion'][:10]:
        analysis = ""
        if conf['class'] == 'B':
            analysis = "✓ Confusão esperada (muito similares)"
        elif conf['class'] == 'C':
            analysis = "→ Mesma categoria fonética"
        elif conf['class'] == 'D':
            analysis = "⚠ Confusão grave"
        
        logger.info(
            f"  {conf['ref']:>4} → {conf['pred']:<4}  {conf['count']:5d}  "
            f"{conf['distance']:6.4f}  {conf['class']:>6}  {analysis}"
        )
    logger.info("")
    
    # =========================================================================
    # 8. Análise de anomalias comportamentais
    # =========================================================================
    logger.info("=" * 70)
    logger.info("ANÁLISE DE ANOMALIAS COMPORTAMENTAIS")
    logger.info("=" * 70)
    
    # 8.1 Distribuição de comprimento
    length_anomalies = analyze_length_distribution(words, predictions, references)
    stats = length_anomalies['stats']
    
    logger.info("")
    logger.info("Estatísticas de comprimento (pred_len - ref_len):")
    logger.info(f"  Média:   {stats['mean_diff']:+.2f} fonemas")
    logger.info(f"  Desvio:  {stats['std_diff']:.2f} fonemas")
    logger.info(f"  Mediana: {stats['median_diff']:+.0f} fonemas")
    logger.info(f"  Range:   [{stats['min_diff']:+d}, {stats['max_diff']:+d}]")
    logger.info("")
    
    if stats['total_truncated'] > 0:
        logger.info(f"⚠  Truncation detectado: {stats['total_truncated']} palavras (diff ≤ -3)")
        truncated_sorted = sorted(length_anomalies['truncated'], key=lambda x: x['diff'])[:5]
        logger.info(f"  Top {len(truncated_sorted)} truncadas:")
        for item in truncated_sorted:
            logger.info(f"    {item['word']:<15} ref={item['ref_len']:2d} pred={item['pred_len']:2d} (Δ{item['diff']:+d})")
    else:
        logger.info("✓  Sem truncation detectado")
    
    logger.info("")
    if stats['total_overgenerated'] > 0:
        logger.info(f"⚠  Over-generation detectado: {stats['total_overgenerated']} palavras (diff ≥ +3)")
        overgen_sorted = sorted(length_anomalies['overgenerated'], key=lambda x: x['diff'], reverse=True)[:5]
        logger.info(f"  Top {len(overgen_sorted)} over-generated:")
        for item in overgen_sorted:
            logger.info(f"    {item['word']:<15} ref={item['ref_len']:2d} pred={item['pred_len']:2d} (Δ{item['diff']:+d})")
    else:
        logger.info("✓  Sem over-generation detectado")
    
    logger.info("")
    
    # 8.2 Detecção de alucinações
    hallucinations = detect_hallucinations(words, predictions, references)
    
    if hallucinations:
        logger.info(f"⚠  Alucinações detectadas: {len(hallucinations)} palavras")
        logger.info(f"  Top {min(5, len(hallucinations))} alucinações:")
        for item in hallucinations[:5]:
            patterns_str = ', '.join(item['patterns'])
            logger.info(f"    {item['word']:<15} patterns=[{patterns_str}]")
            logger.info(f"      pred: {item['pred_str']}")
            logger.info(f"      ref:  {item['ref_str']}")
    else:
        logger.info("✓  Sem alucinações detectadas (sem loops ou repetições anômalas)")
    
    logger.info("")
    
    # =========================================================================
    # 9. Exemplos por classe de erro
    # =========================================================================
    if word_scores:
        logger.info("=" * 70)
        logger.info("EXEMPLOS POR CLASSE DE ERRO")
        logger.info("=" * 70)
        
        for cls in ['B', 'C', 'D']:
            cls_words = words_by_class.get(cls, [])
            if not cls_words:
                continue
            
            # Ordenar por score (pior primeiro)
            cls_sorted = sorted(cls_words, key=lambda x: x['score'])
            show_count = min(5, len(cls_sorted))
            
            label = {'B': 'LEVE', 'C': 'MÉDIO', 'D': 'GRAVE'}[cls]
            logger.info("")
            logger.info(f"  Classe {cls} — Erro {label} ({len(cls_words)} palavras, mostrando {show_count}):")
            logger.info(f"  {'Palavra':<15} {'Score':>6}  {'Predito':<30} {'Referência':<30} {'Diffs'}")
            logger.info("  " + "-" * 95)
            
            for ws in cls_sorted[:show_count]:
                pred_str = ' '.join(ws['pred'])
                ref_str = ' '.join(ws['ref'])
                
                # Diferenças
                diffs = []
                for p, r in zip(ws['pred'], ws['ref']):
                    if p != r:
                        diffs.append(f"{r}→{p}")
                len_diff = len(ws['pred']) - len(ws['ref'])
                if len_diff > 0:
                    diffs.append(f"+{len_diff}ins")
                elif len_diff < 0:
                    diffs.append(f"{len_diff}del")
                
                diff_str = ', '.join(diffs[:4])
                if len(diffs) > 4:
                    diff_str += "..."
                
                logger.info(
                    f"  {ws['word']:<15} {ws['score']:6.3f}  {pred_str:<30} {ref_str:<30} {diff_str}"
                )
        
        logger.info("")
    
    # =========================================================================
    # 9. Substituições mais comuns (referência clássica)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("TOP-15 SUBSTITUIÇÕES (contagem bruta)")
    logger.info("=" * 70)
    
    substitutions = Counter()
    for pred, ref in zip(predictions, references):
        for p, r in zip(pred, ref):
            if p != r:
                substitutions[(r, p)] += 1
    
    logger.info(f"  {'Ref':>4} → {'Pred':<4}  {'Count':>5}")
    logger.info("  " + "-" * 20)
    for (ref_ph, pred_ph), count in substitutions.most_common(15):
        logger.info(f"  {ref_ph:>4} → {pred_ph:<4}  {count:5d}")
    logger.info("")
    
    # =========================================================================
    # 11. Salvar relatório
    # =========================================================================
    # Extrair base_name do predictions_file para manter rastreabilidade
    base_name = get_base_name_from_path(pred_path)
    if not base_name:
        # Fallback se não conseguir extrair
        base_name = pred_path.stem.replace("predictions_", "")
    
    report_file = RESULTS_DIR / f"error_analysis_{base_name}.txt"
    total_time = time.perf_counter() - start_time
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ANÁLISE DE ERROS FONÉTICOS (PanPhon)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Predições: {pred_path.name}\n")
        f.write(f"Total: {total_words} palavras | Corretas: {correct_count} | Erros: {error_count}\n")
        f.write(f"Tempo análise: {total_time:.2f}s\n")
        f.write("\n")
        
        f.write("MÉTRICAS CLÁSSICAS\n")
        f.write("-" * 40 + "\n")
        f.write(f"PER:      {per:.2f}%\n")
        f.write(f"WER:      {wer:.2f}%\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write("\n")
        
        f.write("MÉTRICAS GRADUADAS (PanPhon)\n")
        f.write("-" * 40 + "\n")
        f.write(f"WER Graduado:  {grad['wer_graduated']:.2f}%\n")
        f.write(f"PER Ponderado: {grad['per_weighted']:.2f}%\n")
        f.write("\n")
        
        f.write("DISTRIBUIÇÃO DE CLASSES (por fonema)\n")
        f.write("-" * 40 + "\n")
        for cls in ['A', 'B', 'C', 'D']:
            count = grad['error_distribution'].get(cls, 0)
            pct = grad['class_proportions'].get(cls, 0.0)
            f.write(f"Classe {cls}: {count:6d} ({pct:5.2f}%)\n")
        f.write("\n")
        
        if word_scores:
            f.write("WER SEGMENTADO (por pior classe na palavra)\n")
            f.write("-" * 40 + "\n")
            for cls in ['A', 'B', 'C', 'D']:
                n = len(words_by_class[cls])
                pct = (n / total_words * 100) if total_words > 0 else 0
                f.write(f"Classe {cls}: {n:6d} palavras ({pct:.2f}%)\n")
            f.write("\n")
        
        # Anomalias comportamentais
        f.write("ANOMALIAS COMPORTAMENTAIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Length stats: mean={stats['mean_diff']:+.2f} std={stats['std_diff']:.2f} median={stats['median_diff']:+.0f}\n")
        f.write(f"Truncated: {stats['total_truncated']} palavras\n")
        f.write(f"Over-generated: {stats['total_overgenerated']} palavras\n")
        f.write(f"Hallucinations: {len(hallucinations)} palavras\n")
        f.write("\n")
        
        if stats['total_truncated'] > 0:
            f.write("Top-10 Truncadas:\n")
            for item in sorted(length_anomalies['truncated'], key=lambda x: x['diff'])[:10]:
                f.write(f"  {item['word']:<15} ref={item['ref_len']:2d} pred={item['pred_len']:2d} (Δ{item['diff']:+d})\n")
            f.write("\n")
        
        if stats['total_overgenerated'] > 0:
            f.write("Top-10 Over-generated:\n")
            for item in sorted(length_anomalies['overgenerated'], key=lambda x: x['diff'], reverse=True)[:10]:
                f.write(f"  {item['word']:<15} ref={item['ref_len']:2d} pred={item['pred_len']:2d} (Δ{item['diff']:+d})\n")
            f.write("\n")
        
        if hallucinations:
            f.write("Top-10 Alucinações:\n")
            for item in hallucinations[:10]:
                patterns_str = ', '.join(item['patterns'])
                f.write(f"  {item['word']:<15} patterns=[{patterns_str}]\n")
                f.write(f"    pred: {item['pred_str']}\n")
                f.write(f"    ref:  {item['ref_str']}\n")
            f.write("\n")
        
        f.write("TOP-10 CONFUSÕES FONÉTICAS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Ref':>4} → {'Pred':<4}  {'Count':>5}  {'Dist':>6}  {'Class':>5}\n")
        for conf in grad['phonetic_confusion'][:10]:
            f.write(
                f"{conf['ref']:>4} → {conf['pred']:<4}  {conf['count']:5d}  "
                f"{conf['distance']:6.4f}  {conf['class']:>5}\n"
            )
        f.write("\n")
        
        if word_scores:
            f.write("EXEMPLOS POR CLASSE (até 10 por classe)\n")
            f.write("-" * 40 + "\n")
            for cls in ['B', 'C', 'D']:
                cls_sorted = sorted(words_by_class.get(cls, []), key=lambda x: x['score'])[:10]
                if cls_sorted:
                    f.write(f"\nClasse {cls}:\n")
                    for ws in cls_sorted:
                        pred_str = ' '.join(ws['pred'])
                        ref_str = ' '.join(ws['ref'])
                        f.write(f"  {ws['word']:<15} score={ws['score']:.3f}  pred={pred_str}  ref={ref_str}\n")
        
        f.write("\n")
        f.write("TOP-15 SUBSTITUIÇÕES\n")
        f.write("-" * 40 + "\n")
        for (ref_ph, pred_ph), count in substitutions.most_common(15):
            f.write(f"{ref_ph:>4} → {pred_ph:<4}  {count:5d}\n")
    
    logger.info("=" * 70)
    logger.info(f"Relatório salvo em: {report_file.name}")
    logger.info(f"Tempo total: {total_time:.2f}s")
    logger.info("=" * 70)
    logger.info("")


if __name__ == "__main__":
    main()
