"""
Análise de Homógrafos Heterófonos (Héteronímos) em PT-BR

Fenômeno: Palavras com a MESMA grafia mas DIFERENTES pronuncias dependendo de
categoria gramatical (substantivo vs verbo, etc.), distinguidas por qualidade vocálica
(aberta vs fechada).

Exemplos clássicos:
- "jogo" (substantivo) = /ˈʒɔgʊ/ vs "jogo" (verbo) = /ˈʒogʊ/
- "gosto" (substantivo) = /ˈgɔʃtʊ/ vs "gosto" (verbo) = /ˈgoʃtʊ/
- "seco" (adjetivo) = /ˈsɛkʊ/ vs "seco" (verbo) = /ˈsekʊ/

Este script verifica: O corpus PT-BR contém QUANTAS e QUAIS dessas palavras?
Para cada par heterófono, qual forma foi escolhida para ser armazenada?
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("homograph_analysis")

DICT_PATH = Path("dicts/pt-br.tsv")
OUTPUT_JSON = Path("results/homograph_analysis.json")
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# Lista de pares heterófonos clássicos do PT-BR
# Formato: (grafia, forma_A_descricao, forma_A_ipa, forma_B_descricao, forma_B_ipa)
HETEROPHONE_PAIRS = [
    # Pares com /o/ ↔ /ɔ/ (vogal média posterior)
    ("jogo", "substantivo", "ˈʒɔgʊ", "1ª p. verbo", "ˈʒogʊ"),
    ("gosto", "substantivo", "ˈgɔʃtʊ", "1ª p. verbo", "ˈgoʃtʊ"),
    ("molho", "substantivo (sauce)", "ˈmɔʎʊ", "1ª p. verbo", "ˈmoʎʊ"),
    ("torço", "substantivo (plant)", "ˈtɔɾsʊ", "1ª p. verbo", "ˈtoɾsʊ"),
    ("corte", "substantivo", "ˈkɔɾtʃɪ", "1ª p. verbo", "ˈkoɾtʃɪ"),
    ("sorte", "substantivo", "ˈsɔɾtʃɪ", "1ª p. verbo", "ˈsoɾtʃɪ"),
    ("porto", "substantivo", "ˈpɔɾtʊ", "1ª p. verbo", "ˈpoɾtʊ"),
    ("forma", "substantivo", "ˈfɔɾmə", "1ª p. verbo", "ˈfoɾmə"),
    ("morro", "substantivo", "ˈmɔxʊ", "1ª p. verbo", "ˈmoɾʊ"),
    ("torro", "substantivo", "ˈtɔxʊ", "1ª p. verbo", "ˈtoɾʊ"),
    ("socorro", "substantivo", "sʊˈkɔxʊ", "1ª p. verbo", "sʊˈkoɾʊ"),
    ("acordo", "substantivo", "aˈkɔɾdʊ", "1ª p. verbo", "aˈkoɾdʊ"),
    ("adorço", "substantivo", "aˈdɔɾsʊ", "1ª p. verbo", "aˈdoɾsʊ"),
    ("aperto", "substantivo", "aˈpɛɾtʊ", "1ª p. verbo", "aˈpeɾtʊ"),

    # Pares com /e/ ↔ /ɛ/ (vogal média anterior)
    ("seco", "adjetivo", "ˈsɛkʊ", "1ª p. verbo", "ˈsekʊ"),
    ("pego", "adjetivo/particípio", "ˈpɛgʊ", "1ª p. verbo", "ˈpegʊ"),
    ("cego", "adjetivo", "ˈsɛgʊ", "1ª p. verbo", "ˈsegʊ"),
    ("cedo", "advérbio", "ˈsɛdʊ", "1ª p. verbo", "ˈsedʊ"),
    ("neto", "substantivo", "ˈnɛtʊ", "1ª p. verbo", "ˈnetʊ"),
    ("veto", "substantivo", "ˈvɛtʊ", "1ª p. verbo", "ˈvetʊ"),
    ("dedo", "substantivo", "ˈdɛdʊ", "1ª p. verbo", "ˈdedʊ"),
    ("medo", "substantivo", "ˈmɛdʊ", "1ª p. verbo", "ˈmedʊ"),
    ("rego", "substantivo", "ˈɾɛgʊ", "1ª p. verbo", "ˈɾegʊ"),
    ("teco", "substantivo (rare)", "ˈtɛkʊ", "1ª p. verbo", "ˈtekʊ"),
]


def load_corpus():
    """Carrega corpus PT-BR — retorna dict {palavra: transcrição}"""
    corpus = {}
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
                corpus[word.lower()] = phonemes

    logger.info(f"Carregado: {len(corpus)} palavras")
    return corpus


def analyze_heterophone_pairs(corpus):
    """Analisa quais pares heterófonos existem no corpus e qual forma foi escolhida"""
    logger.info("Analisando pares heterófonos...")

    results = {
        "total_pairs": len(HETEROPHONE_PAIRS),
        "pairs_in_corpus": 0,
        "pairs_missing": 0,
        "pairs_with_one_form": 0,
        "pairs_with_both_forms": 0,
        "pairs": []
    }

    for grafia, forma_a_desc, forma_a_ipa, forma_b_desc, forma_b_ipa in HETEROPHONE_PAIRS:
        in_corpus = grafia in corpus
        stored_ipa = corpus.get(grafia, None)

        entry = {
            "word": grafia,
            "form_A": {
                "description": forma_a_desc,
                "ipa": forma_a_ipa
            },
            "form_B": {
                "description": forma_b_desc,
                "ipa": forma_b_ipa
            },
            "in_corpus": in_corpus,
            "corpus_entry": stored_ipa if in_corpus else None,
            "matches_A": stored_ipa == forma_a_ipa if in_corpus else None,
            "matches_B": stored_ipa == forma_b_ipa if in_corpus else None,
            "missing_form": None
        }

        if in_corpus:
            results["pairs_in_corpus"] += 1
            if stored_ipa == forma_a_ipa:
                entry["chosen_form"] = forma_a_desc
                entry["missing_form"] = forma_b_desc
            elif stored_ipa == forma_b_ipa:
                entry["chosen_form"] = forma_b_desc
                entry["missing_form"] = forma_a_desc
            else:
                entry["chosen_form"] = "NEITHER (unexpected)"
                entry["missing_form"] = "both"
        else:
            results["pairs_missing"] += 1
            entry["chosen_form"] = None

        results["pairs"].append(entry)

    # Contar estatísticas
    for p in results["pairs"]:
        if p["in_corpus"]:
            if p["chosen_form"] in [p["form_A"]["description"], p["form_B"]["description"]]:
                results["pairs_with_one_form"] += 1
                results["pairs_with_both_forms"] += 0
            else:
                results["pairs_with_both_forms"] += 1

    return results


def main():
    logger.info("=" * 70)
    logger.info("ANÁLISE DE HOMÓGRAFOS HETERÓFONOS (Héteronímos)")
    logger.info("=" * 70)

    corpus = load_corpus()
    if not corpus:
        logger.error("Não foi possível carregar o corpus!")
        return

    analysis = analyze_heterophone_pairs(corpus)

    # Salvar
    logger.info(f"Salvando resultados em {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # Resumo
    logger.info("\n" + "=" * 70)
    logger.info("RESUMO DA ANÁLISE")
    logger.info("=" * 70)
    logger.info(f"Total de pares heterófonos clássicos: {analysis['total_pairs']}")
    logger.info(f"Presentes no corpus: {analysis['pairs_in_corpus']}")
    logger.info(f"Ausentes no corpus: {analysis['pairs_missing']}")
    logger.info(f"Corpus escolheu 1 forma: {analysis['pairs_with_one_form']}")
    logger.info(f"Corpus tem ambas formas: {analysis['pairs_with_both_forms']}")

    logger.info("\nPares que estão NO CORPUS:")
    for p in analysis["pairs"]:
        if p["in_corpus"]:
            logger.info(f"  '{p['word']}' → {p['chosen_form']} armazenado")
            logger.info(f"    Forma ausente: {p['missing_form']}")

    logger.info("\nPares AUSENTES do corpus (modelo nunca aprenderá):")
    for p in analysis["pairs"]:
        if not p["in_corpus"]:
            logger.info(f"  '{p['word']}' (ambas formas ausentes)")

    logger.info(f"\n✓ Análise salva em: {OUTPUT_JSON}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
