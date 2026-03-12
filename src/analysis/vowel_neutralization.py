"""
Análise de Neutralização Vocálica (e ↔ ɛ) em PT-BR

Fenômeno linguístico: As vogais médias /e/ e /ɛ/ são fonêmicas distintas em PT-BR
em sílabas tônicas (péle vs pele), mas em sílabas átonas tendem a se neutralizar.

Este script quantifica:
1. Distribuição de e/ɛ por contexto grafêmico (te, de, se, etc.)
2. Padrão por posição silábica (tônica vs átona)
3. Impacto no aprendizado do LSTM (462 erros e↔ɛ no Exp104b = 18.9% de todos os erros)
4. Palavras "-este" (padrão específico: teste, veste, agreste, oeste, etc.)
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s"
)
logger = logging.getLogger("vowel_analysis")

# Paths
DICT_PATH = Path("dicts/pt-br.tsv")
OUTPUT_JSON = Path("results/vowel_analysis.json")
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def load_corpus():
    """Carrega corpus PT-BR (95.937 palavras) — retorna lista de (palavra, transcrição)"""
    corpus = []
    if not DICT_PATH.exists():
        logger.error(f"Corpus não encontrado: {DICT_PATH}")
        return corpus

    logger.info(f"Carregando corpus de {DICT_PATH}...")
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                word, phonemes = parts[0], parts[1]
                corpus.append((word, phonemes))

    logger.info(f"Carregado: {len(corpus)} palavras")
    return corpus


def analyze_bigram_patterns(corpus):
    """
    Analisa distribuição de e/ɛ para cada bigrama grafêmico + vogal

    Retorna dict: {
        "te": {"e": 1200, "ɛ": 340, ...},
        "de": {"e": 450, "ɛ": 120, ...},
        ...
    }
    """
    logger.info("Analisando padrões de bigramas grafêmicos...")
    bigram_vowels = defaultdict(lambda: defaultdict(int))

    for word, phonemes in corpus:
        # Procura por sequências "XY" (bigrama grafêmico) + "e" ou "ɛ"
        # Precisamos fazer matching entre grafemas (word) e fonemas (phonemes)
        # Aproximação: procurar por "te" em word e "e"/"ɛ" na transcrição próxima

        for i in range(len(word) - 1):
            bigram = word[i:i+2].lower()

            # Verificar se há vogal após este bigrama na transcrição
            # Buscar fonemas seguindo este contexto grafêmico
            if bigram[-1] in 'aeiouááàâãéêíóôõú' or bigram in ['ch', 'lh', 'nh', 'rr', 'ss', 'cc', 'çc', 'sc']:
                # Procura por vogal média na transcrição próxima
                # Abordagem simplificada: contar e/ɛ na transcrição inteira por contexto
                pass

        # Abordagem alternativa mais robusta: procurar sequências de fonemas "C+e" e "C+ɛ"
        # onde C é consoante, em contexto de bigrama grafêmico final
        phoneme_list = phonemes.split()

        for i in range(len(phoneme_list) - 1):
            current = phoneme_list[i]
            next_ph = phoneme_list[i + 1]

            # Se current é consoante e next é e/ɛ
            if current not in ['e', 'ɛ', 'a', 'ɔ', 'o', 'u', 'ĩ', 'ũ', 'õ', 'ə', 'ɾ', 'ɲ',
                              'w', 'j', 'ʃ', 'ʒ', 'x', 'ɣ', '.', 'ˈ', 'ã', 'ɪ', 'ʊ'] \
                    and next_ph in ['e', 'ɛ']:

                # Encontrar contexto grafêmico: procurar pela posição na palavra
                # Aproximação: usar último bigrama da palavra
                if len(word) >= 2:
                    last_bigram = word[-2:].lower()
                    vowel_key = next_ph

                    # Contar por contexto mais específico possível
                    if 'e' in last_bigram or 'a' in last_bigram:  # Contexto com vogal
                        bigram_vowels[last_bigram][vowel_key] += 1
                    else:
                        bigram_vowels[current][vowel_key] += 1

    # Filtra bigramas com pelo menos 10 ocorrências
    filtered = {
        k: dict(v) for k, v in bigram_vowels.items()
        if sum(v.values()) >= 5  # Menos rigoroso para ter mais dados
    }

    logger.info(f"Encontrados {len(filtered)} padrões de bigramas com e/ɛ")
    return filtered


def analyze_stress_position(corpus):
    """
    Analisa distribuição de e/ɛ por posição relativa ao acento primário (ˈ)

    Retorna dict: {
        "pre_tonic": {"e": count, "ɛ": count},     # Sílaba antes do acento
        "tonic": {"e": count, "ɛ": count},         # Na sílaba tônica
        "post_tonic": {"e": count, "ɛ": count}     # Sílaba depois do acento
    }
    """
    logger.info("Analisando posição de e/ɛ relativa ao acento primário...")
    positions = {
        "pre_tonic": defaultdict(int),
        "tonic": defaultdict(int),
        "post_tonic": defaultdict(int),
        "unstressed": defaultdict(int)
    }

    for word, phonemes in corpus:
        phoneme_list = phonemes.split()

        # Encontra posição do acento primário
        stress_idx = -1
        for i, ph in enumerate(phoneme_list):
            if ph == 'ˈ':
                stress_idx = i
                break

        for i, ph in enumerate(phoneme_list):
            if ph in ['e', 'ɛ']:
                if stress_idx == -1:
                    # Sem acento marcado (palavras não tônicas ou monossílabas)
                    positions["unstressed"][ph] += 1
                elif i < stress_idx:
                    positions["pre_tonic"][ph] += 1
                elif i == stress_idx + 1:
                    # Imediatamente após o marcador ˈ
                    positions["tonic"][ph] += 1
                else:
                    positions["post_tonic"][ph] += 1

    return {k: dict(v) for k, v in positions.items()}


def find_este_words(corpus):
    """
    Encontra palavras com padrão "-este" e verifica vogal usada

    Retorna lista de (palavra, transcrição, vogal_usada)
    """
    logger.info("Procurando palavras com padrão '-este'...")
    este_words = []

    for word, phonemes in corpus:
        if word.lower().endswith("este"):
            # Procura por padrão final "ɛ ʃ ." ou "e ʃ ."
            phoneme_list = phonemes.split()

            # Identifica o "e" ou "ɛ" que corresponde ao primeiro 'e' de "este"
            # Este é um match aproximado — procuramos vogal + ʃ
            vowel_used = None
            for i in range(len(phoneme_list) - 1):
                if phoneme_list[i] in ['e', 'ɛ'] and phoneme_list[i+1] == 'ʃ':
                    vowel_used = phoneme_list[i]
                    break

            este_words.append({
                "word": word,
                "phonemes": phonemes,
                "vowel": vowel_used,
                "expected": "ɛ"  # Todos devem ser ɛ em PT-BR
            })

    logger.info(f"Encontradas {len(este_words)} palavras com '-este'")
    return este_words


def analyze_te_distribution(corpus):
    """
    Análise focada em sequências "te" (o exemplo do usuário)

    Retorna estatísticas detalhadas sobre "te" → /e/ vs /ɛ/
    """
    logger.info("Analisando distribuição de 'te' especificamente...")

    te_pattern = {
        "total_te": 0,
        "te_e": 0,
        "te_epsilon": 0,
        "tonic_te_e": 0,
        "tonic_te_epsilon": 0,
        "atonic_te_e": 0,
        "atonic_te_epsilon": 0
    }

    for word, phonemes in corpus:
        word_lower = word.lower()

        if "te" not in word_lower:
            continue

        phoneme_list = phonemes.split()
        stress_idx = next((i for i, p in enumerate(phoneme_list) if p == 'ˈ'), -1)

        # Procura por padrões "t + e/ɛ"
        for i in range(len(phoneme_list) - 1):
            if phoneme_list[i] in ['t', 'ʃ', 'dʒ']:  # Inclui variações de "t"
                if phoneme_list[i+1] in ['e', 'ɛ']:
                    te_pattern["total_te"] += 1
                    vowel = phoneme_list[i+1]

                    if vowel == 'e':
                        te_pattern["te_e"] += 1
                        if stress_idx != -1 and i >= stress_idx:
                            te_pattern["tonic_te_e"] += 1
                        else:
                            te_pattern["atonic_te_e"] += 1
                    else:  # ɛ
                        te_pattern["te_epsilon"] += 1
                        if stress_idx != -1 and i >= stress_idx:
                            te_pattern["tonic_te_epsilon"] += 1
                        else:
                            te_pattern["atonic_te_epsilon"] += 1

    return te_pattern


def main():
    """Executa análise completa e salva resultados"""
    logger.info("=" * 70)
    logger.info("ANÁLISE DE NEUTRALIZAÇÃO VOCÁLICA (e ↔ ɛ) — PT-BR")
    logger.info("=" * 70)

    corpus = load_corpus()
    if not corpus:
        logger.error("Não foi possível carregar o corpus!")
        return

    # Análise 1: Padrões por bigrama
    bigram_analysis = analyze_bigram_patterns(corpus)

    # Análise 2: Posição relativa ao acento
    stress_analysis = analyze_stress_position(corpus)

    # Análise 3: Palavras "-este"
    este_words = find_este_words(corpus)

    # Análise 4: Foco em "te"
    te_analysis = analyze_te_distribution(corpus)

    # Compilar resultado
    results = {
        "metadata": {
            "analysis_date": "2026-02-26",
            "corpus_size": len(corpus),
            "phenomenon": "Neutralização vocálica PT-BR: e ↔ ɛ em sílabas átonas"
        },
        "bigram_patterns": bigram_analysis,
        "stress_position_analysis": stress_analysis,
        "te_specific_analysis": te_analysis,
        "este_words": {
            "total_found": len(este_words),
            "all_epsilon": sum(1 for w in este_words if w.get("vowel") == "ɛ"),
            "examples": este_words[:20]  # Primeiros 20 exemplos
        },
        "key_statistics": {
            "total_e": sum(v.get("e", 0) for v in stress_analysis.values() if isinstance(v, dict)),
            "total_epsilon": sum(v.get("ɛ", 0) for v in stress_analysis.values() if isinstance(v, dict)),
            "te_ratio_e_to_epsilon": f"{te_analysis['te_e']}:{te_analysis['te_epsilon']}",
            "model_error_context": {
                "exp104b_e_to_epsilon_errors": 265,
                "exp104b_epsilon_to_e_errors": 197,
                "exp104b_total_errors": 462,
                "percentage_of_all_errors": "18.9%"
            }
        }
    }

    # Salvar resultados
    logger.info(f"Salvando resultados em {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("RESUMO DOS RESULTADOS")
    logger.info("=" * 70)
    logger.info(f"Total de 'te': {te_analysis['total_te']}")
    logger.info(f"  → 'te' com /e/: {te_analysis['te_e']} ({100*te_analysis['te_e']/max(1, te_analysis['total_te']):.1f}%)")
    logger.info(f"  → 'te' com /ɛ/: {te_analysis['te_epsilon']} ({100*te_analysis['te_epsilon']/max(1, te_analysis['total_te']):.1f}%)")

    if te_analysis['total_te'] > 0:
        ratio = te_analysis['te_e'] / (te_analysis['te_epsilon'] + 0.001)
        logger.info(f"  → Ratio e:ɛ = {ratio:.1f}:1")

    logger.info(f"\nPalavras '-este': {len(este_words)} encontradas")
    all_epsilon_count = sum(1 for w in este_words if w.get("vowel") == "ɛ")
    logger.info(f"  → Todas com /ɛ/: {all_epsilon_count}/{len(este_words)}")

    logger.info("\nPadrões por posição tônica:")
    for pos, counts in stress_analysis.items():
        if isinstance(counts, dict) and counts:
            logger.info(f"  {pos}: e={counts.get('e', 0)}, ɛ={counts.get('ɛ', 0)}")

    logger.info(f"\n✓ Análise salva em: {OUTPUT_JSON}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
