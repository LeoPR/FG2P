# FG2P — Índice de Documentação

> **Modelo neuronal BiLSTM Encoder-Decoder com atenção Bahdanau** para conversão automática de grafemas em fonemas IPA para o Português Brasileiro.

**Status**: Phase 6C Completa | PER 0.49% (Exp104b) | WER 4.96% (Exp9) | 95.937 palavras

---

## Resultados Principais

| **Exp** | **Loss** | **Sep** | **PER** | **WER** | **Nota** |
|---------|----------|---------|---------|---------|----------|
| **Exp9** | DA λ=0.2 | não | 0.58% | **4.96%** | **SOTA WER** |
| **Exp104b** | DA λ=0.2 + dist | sim | **0.49%** | 5.43% | **SOTA PER** |
| Exp106 | DA λ=0.2 + dist | sim | 0.58% | 6.12% | 2.58x speed |

---

## Documentação

### Artigo Científico → [article/](article/)

| Arquivo | Conteúdo |
|---------|----------|
| [ARTICLE.md](article/ARTICLE.md) | Documento principal (~913 linhas) — §1 Introdução, §2 Dados, §3 Arquitetura, §4 Loss, §5 Resultados, §6 Análise de Erros, §7 Generalização, §8 Ablações, §9 Limitações |
| [EXPERIMENTS.md](article/EXPERIMENTS.md) | Log completo Exp0–106 (~929 linhas) |
| [PIPELINE.md](article/PIPELINE.md) | Pipeline técnico: corpus → vocabulários → treino |
| [GLOSSARY.md](article/GLOSSARY.md) | Glossário: fonética PT-BR, ML, termos do projeto |
| [REFERENCES.bib](article/REFERENCES.bib) | Bibliografia completa — fonte única (BibTeX, suporte LaTeX/Zotero/Mendeley) |
| [ORIGINALITY_ANALYSIS.md](article/ORIGINALITY_ANALYSIS.md) | Pesquisa de originalidade da Distance-Aware Loss (fontes internas + externas) |

### Apresentação → [presentation/](presentation/)

| Arquivo | Conteúdo |
|---------|----------|
| [PRESENTATION.md](presentation/PRESENTATION.md) | Fonte dos slides PPTX (31 full / 20 compact) — com cross-refs para o artigo |
| [GENERATOR.md](presentation/GENERATOR.md) | Documentação do gerador PPTX |

### Relatório → [report/](report/)

| Arquivo | Conteúdo |
|---------|----------|
| [performance.json](report/performance.json) | Benchmarks + comparação SOTA |

### Dados de Avaliação → [data/](data/)

| Arquivo | Conteúdo |
|---------|----------|
| [generalization_test.tsv](data/generalization_test.tsv) | 31 palavras em 6 categorias (avaliação OOV) |
| [neologisms_test.tsv](data/neologisms_test.tsv) | 35 neologismos curados |

### Linguística → [linguistics/](linguistics/)

| Arquivo | Conteúdo |
|---------|----------|
| [PHONOLOGICAL_ANALYSIS.md](linguistics/PHONOLOGICAL_ANALYSIS.md) | Validação dos símbolos IPA (ɣ/x, distribuição complementar) — pesquisa fonológica detalhada com fontes primárias |

---

## Quick Start

```bash
# Inferência com modelo SOTA PER (Exp104b, index=18)
python src/inference_light.py --index 18 --word computador
# → k õ p u t a . ˈ d o x .

# Avaliar banco de generalização
python src/inference_light.py --index 18 --neologisms docs/data/generalization_test.tsv

# Modo interativo
python src/inference_light.py --index 18 --interactive
```

---

## Qual Modelo Usar

| Modelo | Index | PER | WER | Quando usar |
|--------|-------|-----|-----|-------------|
| **Exp9** | 11 | 0.58% | **4.96%** | Precisão por palavra — NLP, lookup, TTS |
| **Exp104b** | 18 | **0.49%** | 5.43% | Precisão por fonema — análise linguística |
| Exp106 | 20 | 0.58% | 6.12% | Latência crítica — 30.2 w/s (2.58x) |

---

## Leitura Recomendada

**Visão geral rápida** (~15 min): Este arquivo + [ARTICLE.md §1, §5, §9](article/ARTICLE.md)

**Artigo completo** (~2h): [ARTICLE.md](article/ARTICLE.md) — leitura linear, IMRaD

**Log de experimentos** (referência): [EXPERIMENTS.md](article/EXPERIMENTS.md) — Exp0–106

**Pipeline técnico** (implementação): [PIPELINE.md](article/PIPELINE.md) — corpus → treino

---

**Última atualização**: 2026-02-28
