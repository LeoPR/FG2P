#!/usr/bin/env python3
"""
analyze_divergences.py — Analisar diferenças entre dois corpus de G2P

Compara dois arquivos TSV (grafema TAB fonemas) e identifica:
  1. Palavras exclusivas de cada arquivo
  2. Palavras com IPA diferente (divergências reais)
  3. Categorias de divergência (vogais, consoantes, stress, silabificação)
  4. Exemplos representativos por categoria

Uso:
    python src/analysis/analyze_divergences.py
    python src/analysis/analyze_divergences.py --file-a dicts/pt-br.tsv --file-b dicts/pt_BR_converted.tsv
    python src/analysis/analyze_divergences.py --output results/divergence_report.json
    python src/analysis/analyze_divergences.py --top 30
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent.parent.parent


# =============================================================================
# Grupos fonéticos para categorizar divergências
# =============================================================================

VOWEL_CLOSED = {'e', 'o', 'i', 'u', 'a'}
VOWEL_OPEN = {'ɛ', 'ɔ'}
VOWEL_REDUCED = {'ə', 'ɪ', 'ʊ'}
NASAL_VOWELS = {'ã', 'ĩ', 'ũ', 'ẽ', 'õ', 'ỹ'}
ALL_VOWELS = VOWEL_CLOSED | VOWEL_OPEN | VOWEL_REDUCED | NASAL_VOWELS

RHOTIC = {'x', 'ɾ', 'ɣ', 'r'}
PALATAL = {'ɲ', 'ʎ'}
SIBILANT = {'ʃ', 'ʒ', 's', 'z'}
AFFRICATE = {'tʃ', 'dʒ'}  # pares (verificar como bigrama)


def load_tsv(path: Path) -> dict[str, str]:
    """Carrega TSV (grafema TAB ipa) → {grafema: ipa}"""
    d = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                d[parts[0]] = parts[1]
    return d


def categorize_diff(ipa_a: str, ipa_b: str) -> list[str]:
    """Classifica o tipo de diferença entre dois IPAs."""
    # Tokens são separados por espaço (formato atual)
    tokens_a = set(ipa_a.split())
    tokens_b = set(ipa_b.split())

    diff_a = tokens_a - tokens_b  # tokens só em A
    diff_b = tokens_b - tokens_a  # tokens só em B

    all_diff = diff_a | diff_b
    categories = []

    # Verificar tipo de diferença
    has_vowel = bool(all_diff & ALL_VOWELS)
    has_nasal = bool(all_diff & NASAL_VOWELS)
    has_rhotic = bool(all_diff & RHOTIC)
    has_palatal = bool(all_diff & PALATAL)
    has_stress = 'ˈ' in diff_a or 'ˈ' in diff_b
    has_syllable = '.' in diff_a or '.' in diff_b

    # Vogal média aberta/fechada (ε/ɔ vs e/o)
    if {'e', 'ɛ'} & all_diff or {'o', 'ɔ'} & all_diff:
        categories.append('vogal_media')
    elif has_nasal:
        categories.append('nasalizacao')
    elif has_vowel:
        categories.append('vogal')
    if has_rhotic:
        categories.append('rotico')
    if has_palatal:
        categories.append('palatal')
    if has_stress:
        categories.append('stress')
    if has_syllable and not has_stress:
        categories.append('silabificacao')

    # Redução vocálica (ə, ɪ, ʊ vs vogais plenas)
    if VOWEL_REDUCED & all_diff:
        categories.append('reducao_vocalica')

    # Diferença de comprimento (tokens extras ou faltando)
    len_a = len(ipa_a.split())
    len_b = len(ipa_b.split())
    if abs(len_a - len_b) > 2:
        categories.append('comprimento')

    return categories if categories else ['outro']


def compute_phoneme_diffs(ipa_a: str, ipa_b: str) -> tuple[list[str], list[str]]:
    """Retorna tokens que diferem entre os dois IPAs."""
    tokens_a = ipa_a.split()
    tokens_b = ipa_b.split()
    only_a = [t for t in tokens_a if t not in tokens_b]
    only_b = [t for t in tokens_b if t not in tokens_a]
    return only_a, only_b


def main():
    parser = argparse.ArgumentParser(
        description='Analisa divergências entre dois corpus G2P (TSV)'
    )
    parser.add_argument(
        '--file-a', type=Path,
        default=BASE_DIR / 'dicts' / 'pt-br.tsv',
        help='Corpus A (referência, default: dicts/pt-br.tsv)'
    )
    parser.add_argument(
        '--file-b', type=Path,
        default=BASE_DIR / 'dicts' / 'pt_BR_converted.tsv',
        help='Corpus B (backup convertido, default: dicts/pt_BR_converted.tsv)'
    )
    parser.add_argument(
        '--output', type=Path,
        default=BASE_DIR / 'results' / 'divergence_report.json',
        help='Relatório JSON de saída'
    )
    parser.add_argument(
        '--top', type=int, default=20, metavar='N',
        help='N exemplos por categoria nos relatórios (default: 20)'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Apenas exibir relatório, não salvar JSON'
    )
    args = parser.parse_args()

    # Verificar se arquivo B existe (pode precisar rodar convert_backup_format.py antes)
    if not args.file_b.exists():
        print(f"[AVISO] {args.file_b} não encontrado.")
        print("  Executar primeiro: python src/analysis/convert_backup_format.py")
        print("\n  Continuando com análise direta do backup bruto...")
        args.file_b = BASE_DIR / 'backups' / 'pt_BR.txt'
        if not args.file_b.exists():
            print(f"[ERRO] {args.file_b} também não encontrado. Abortando.")
            sys.exit(1)

    print(f"[INFO] Carregando corpus A: {args.file_a}")
    corpus_a = load_tsv(args.file_a)
    print(f"  {len(corpus_a)} palavras")

    print(f"[INFO] Carregando corpus B: {args.file_b}")
    corpus_b_raw = load_tsv(args.file_b)

    # Converter B se necessário (detectar formato /.../)
    import unicodedata as ud
    sample_b = next(iter(corpus_b_raw.values()))
    needs_conversion = sample_b.startswith('/')

    if needs_conversion:
        print("  Detectado formato /compacto/ — convertendo automaticamente...")
        corpus_b = {}
        for word, ipa in corpus_b_raw.items():
            if ipa.startswith('/') and ipa.endswith('/'):
                ipa = ipa[1:-1]
            ipa = ud.normalize('NFC', ipa)
            corpus_b[word] = ' '.join(ipa)
    else:
        corpus_b = corpus_b_raw

    print(f"  {len(corpus_b)} palavras")

    # ==========================================================================
    # Análise de vocabulário
    # ==========================================================================
    words_a = set(corpus_a.keys())
    words_b = set(corpus_b.keys())
    common = words_a & words_b
    only_a = words_a - words_b
    only_b = words_b - words_a

    print(f"\n{'='*60}")
    print("ANÁLISE DE VOCABULÁRIO")
    print(f"{'='*60}")
    print(f"  Corpus A ({args.file_a.name}): {len(words_a)} palavras")
    print(f"  Corpus B ({args.file_b.name}): {len(words_b)} palavras")
    print(f"  Palavras em comum:    {len(common)} ({100*len(common)/len(words_a):.1f}% do A)")
    print(f"  Só em A:              {len(only_a)}")
    print(f"  Só em B:              {len(only_b)}")

    if only_a:
        print(f"\n  [Palavras só em A (primeiras {min(10, len(only_a))})]")
        for word in sorted(only_a)[:10]:
            print(f"    {word:<25} → {corpus_a[word]}")

    if only_b:
        print(f"\n  [Palavras só em B (primeiras {min(10, len(only_b))})]")
        for word in sorted(only_b)[:10]:
            print(f"    {word:<25} → {corpus_b[word]}")

    # ==========================================================================
    # Análise de divergências
    # ==========================================================================
    print(f"\n{'='*60}")
    print("ANÁLISE DE DIVERGÊNCIAS")
    print(f"{'='*60}")

    matching = 0
    divergent = []

    for word in common:
        ipa_a = corpus_a[word]
        ipa_b = corpus_b[word]
        if ipa_a == ipa_b:
            matching += 1
        else:
            cats = categorize_diff(ipa_a, ipa_b)
            only_in_a, only_in_b = compute_phoneme_diffs(ipa_a, ipa_b)
            divergent.append({
                'word': word,
                'ipa_a': ipa_a,
                'ipa_b': ipa_b,
                'categories': cats,
                'only_in_a': only_in_a,
                'only_in_b': only_in_b,
            })

    pct_match = 100 * matching / len(common) if common else 0
    print(f"  Palavras idênticas:   {matching}/{len(common)} ({pct_match:.1f}%)")
    print(f"  Divergentes:          {len(divergent)} ({100-pct_match:.1f}%)")

    # Contar categorias
    cat_counter = Counter()
    for d in divergent:
        for cat in d['categories']:
            cat_counter[cat] += 1

    print("\n  [CATEGORIAS DE DIVERGÊNCIA]")
    for cat, count in cat_counter.most_common():
        print(f"    {cat:<25} {count:>6} ({100*count/len(divergent):.1f}%)")

    # Exemplos por categoria
    cat_examples = defaultdict(list)
    for d in divergent:
        for cat in d['categories']:
            if len(cat_examples[cat]) < args.top:
                cat_examples[cat].append(d)

    print("\n  [EXEMPLOS DE DIVERGÊNCIA]")
    for cat, count in cat_counter.most_common(5):
        examples = cat_examples[cat][:5]
        print(f"\n  === {cat.upper()} ({count} ocorrências) ===")
        print(f"  {'PALAVRA':<20} {'CORPUS A (referência)':<35} {'CORPUS B (backup)'}")
        print(f"  {'-'*20} {'-'*35} {'-'*35}")
        for ex in examples:
            print(f"  {ex['word']:<20} {ex['ipa_a']:<35} {ex['ipa_b']}")

    # ==========================================================================
    # Estatísticas de fonemas divergentes
    # ==========================================================================
    phoneme_changes = Counter()
    for d in divergent:
        for p_a in d['only_in_a']:
            for p_b in d['only_in_b']:
                if p_a != p_b and len(p_a) <= 2 and len(p_b) <= 2:
                    phoneme_changes[f"{p_a}→{p_b}"] += 1

    print("\n  [TOP MUDANÇAS FONÊMICAS]")
    for change, count in phoneme_changes.most_common(15):
        print(f"    {change:<10} {count:>5}x")

    # ==========================================================================
    # Salvar relatório JSON
    # ==========================================================================
    report = {
        'metadata': {
            'file_a': str(args.file_a),
            'file_b': str(args.file_b),
            'date': __import__('datetime').date.today().isoformat(),
        },
        'vocabulary': {
            'total_a': len(words_a),
            'total_b': len(words_b),
            'common': len(common),
            'only_a': sorted(only_a),
            'only_b': sorted(only_b)[:50],  # Limitar para não encher o JSON
        },
        'divergences': {
            'total': len(divergent),
            'matching': matching,
            'match_pct': round(pct_match, 2),
            'by_category': dict(cat_counter.most_common()),
            'top_phoneme_changes': dict(phoneme_changes.most_common(30)),
        },
        'examples_by_category': {
            cat: [
                {'word': ex['word'], 'ipa_a': ex['ipa_a'], 'ipa_b': ex['ipa_b']}
                for ex in examples[:15]
            ]
            for cat, examples in cat_examples.items()
        },
        'full_divergent_list': [
            {'word': d['word'], 'ipa_a': d['ipa_a'], 'ipa_b': d['ipa_b'],
             'categories': d['categories']}
            for d in sorted(divergent, key=lambda x: x['word'])
        ]
    }

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Relatório salvo: {args.output}")
        print(f"     {len(divergent)} divergências documentadas")
    else:
        print("\n[INFO] Modo no-save: relatório não salvo.")

    print("\n[PRONTO] Análise concluída.")
    print(f"  Compatibilidade: {pct_match:.1f}%")
    print("  Para testar o modelo no corpus B convertido:")
    print(f"    python src/inference_light.py --index 18 --test-file {args.file_b}")


if __name__ == '__main__':
    main()
