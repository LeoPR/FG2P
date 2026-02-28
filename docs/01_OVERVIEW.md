# FG2P — Conversão Grafema-para-Fonema em Português Brasileiro

> **Escopo**: Modelo neuronal BiLSTM Encoder-Decoder com atenção Bahdanau para conversão automática de texto (grafemas) em transcrição fonética IPA (fonemas) para o Português Brasileiro.

**Data última atualização**: 2026-02-28
**Status**: ✅ Phase 6C Completa | Documentação consolidada em 8 arquivos
**Código-fonte**: [src/](../src/) | **Dados**: [data/](../data/) | **Modelos**: [models/](../models/) | **Resultados**: [results/](../results/)

---

## 🎯 Objetivo

Construir um modelo de alta precisão que converta palavras do Português Brasileiro em sua representação fonética IPA, com aplicações em:
- **Síntese de fala (Text-to-Speech)**: "casa" → [k-a-z-a]
- **Pesquisa linguística**: Análise de padrões fonológicos PT-BR
- **Processamento de linguagem natural**: Extração de features fonéticas

**Métrica focal**: **PER (Phoneme Error Rate)** minimizado; secundária: **WER (Word Error Rate)**.

---

## 📊 Resultados Principais

| **Exp** | **Params** | **Loss** | **Sep** | **PER ↓** | **WER ↓** | **Acc ↑** | **Nota** |
|---------|-----------|----------|---------|-----------|-----------|-----------|----------|
| Exp5 | 9,7M | CE | não | 0,63% | 5,38% | 94,62% | Baseline intermediário |
| **Exp9** | 9,7M | DA λ=0,2 | não | 0,58% | **4,96%** | **95,04%** | **SOTA WER** |
| Exp102 | 9,7M | CE | sim | 0,52% | 5,79% | 94,21% | Sep baseline |
| **Exp104b** | 9,7M | DA λ=0,2 + dist | sim | **0,49%** | 5,43% | 94,57% | **SOTA PER** |
| Exp105 | 9,7M | DA λ=0,2 + dist | sim | 0,54% | 5,87% | 94,13% | 50% dados — robustez |
| Exp106 | 9,7M | DA λ=0,2 + dist | sim | 0,58% | 6,12% | 93,88% | Sem hífen — 2,58× speed ⚡ |

**Descobertas-chave**:
- **SOTA WER**: Exp9 (4,96%) — DA Loss sem separadores
- **SOTA PER**: Exp104b (0,49%) — DA Loss + separadores + override de distância
- **Trade-off PER/WER**: Separadores melhoram PER mas impactam WER — trade-off Pareto fundamental
- **Ablações**: 50% dados → +0,05% PER (robusto); sem hífen → +0,04% PER, 2,58× speed

---

## 📚 Estrutura de Documentação (8 arquivos)

| Arquivo | Propósito | Tamanho |
|---------|-----------|---------|
| **[01_OVERVIEW.md](01_OVERVIEW.md)** | Este índice de navegação | ~150 linhas |
| **[16_SCIENTIFIC_ARTICLE.md](16_SCIENTIFIC_ARTICLE.md)** | Artigo científico completo | ~950+ linhas |
| **[04_EXPERIMENTS.md](04_EXPERIMENTS.md)** | Log completo de todos os experimentos | ~929 linhas |
| **[10_REFERENCES.md](10_REFERENCES.md)** | Bibliografia canônica | ~844 linhas |
| **[12_DATA_PIPELINE.md](12_DATA_PIPELINE.md)** | Pipeline técnico: corpus → vocabulários → treino | ~500 linhas |
| **[GLOSSARIO.md](GLOSSARIO.md)** | Glossário único: fonética, ML, termos do projeto | ~500 linhas |
| **[17_APRESENTACAO.md](17_APRESENTACAO.md)** | Fonte dos slides PPTX — **INTOCÁVEL** | ~272 linhas |
| **[REFACTORING_PRESENTATION_GENERATOR.md](REFACTORING_PRESENTATION_GENERATOR.md)** | Docs do gerador PPTX | ~192 linhas |

---

## 🚀 Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Inferência com modelo SOTA PER (Exp104b, index=18)
python src/inference_light.py --index 18 --word computador
# → k õ p u t a . ˈ d o x .

# Avaliação completa
python src/inference.py

# Relatório HTML
python src/reporting/report_generator.py

# Apresentação PPTX
python src/reporting/presentation_generator.py --compact   # → results/fg2p_presentation.pptx
```

---

## 📖 Leitura Recomendada

**Visão geral rápida** (~15 min):
→ Este arquivo + [16_SCIENTIFIC_ARTICLE.md](16_SCIENTIFIC_ARTICLE.md) (Seções 1, 5, 9)

**Artigo completo** (~2h):
→ [16_SCIENTIFIC_ARTICLE.md](16_SCIENTIFIC_ARTICLE.md) — leitura linear, IMRaD

**Log de experimentos** (referência):
→ [04_EXPERIMENTS.md](04_EXPERIMENTS.md) — tabela completa Exp0–106, análise detalhada

**Pipeline técnico** (implementação):
→ [12_DATA_PIPELINE.md](12_DATA_PIPELINE.md) — corpus → transformações → vocabulários → splits

**Termos e definições**:
→ [GLOSSARIO.md](GLOSSARIO.md) — fonética PT-BR, ML, termos do projeto

---

## 🏗️ Arquitetura

**BiLSTM Encoder-Decoder + Atenção Bahdanau**

```
Grafemas ("casa")
    ↓
[Embedding 192D aprendido]
    ↓
[BiLSTM Encoder 2 camadas, 384D hidden]
    ↓
[Atenção de Bahdanau]
    ↓
[LSTM Decoder 2 camadas]
    ↓
[Projeção Linear → Softmax]
    ↓
Fonemas IPA [k-a-z-a]
```

**Detalhes completos**: [16_SCIENTIFIC_ARTICLE.md § 3](16_SCIENTIFIC_ARTICLE.md) — Seção 3 (Arquitetura)

---

## 📝 Estrutura de Arquivos

```
FG2P/
├── README.md
├── requirements.txt
│
├── conf/                        ← Configurações (Exp0–Exp106)
│   └── config_exp106_no_hyphen_50split.json (e 25 mais)
│
├── docs/
│   ├── 01_OVERVIEW.md           ← Índice (você está aqui)
│   ├── 04_EXPERIMENTS.md        ← Log completo de experimentos
│   ├── 10_REFERENCES.md         ← Bibliografia canônica
│   ├── 12_DATA_PIPELINE.md      ← Pipeline de dados
│   ├── 16_SCIENTIFIC_ARTICLE.md ← Artigo científico completo
│   ├── 17_APRESENTACAO.md       ← Fonte dos slides PPTX (INTOCÁVEL)
│   ├── GLOSSARIO.md             ← Glossário único
│   ├── REFACTORING_PRESENTATION_GENERATOR.md ← Docs do gerador
│   │
│   ├── generalization_test.tsv  ← 31 palavras OOV curadas
│   ├── neologisms_test.tsv      ← Neologismos PT-BR
│   ├── performance.json         ← Benchmarks + comparação SOTA
│   └── REFERENCIAS.bib          ← BibTeX (suporte LaTeX/Word)
│
├── src/
│   ├── g2p.py                   ← Dataset, CharVocab, PhonemeVocab, modelo
│   ├── train.py                 ← Loop de treino + early stopping
│   ├── inference.py             ← Avaliação WER/PER sobre test set
│   ├── inference_light.py       ← G2PPredictor API mínima (produção/CLI)
│   ├── utils.py                 ← get_all_models_sorted(), CHAR_MAPPING
│   ├── losses.py                ← CrossEntropyLoss + DistanceAwareLoss
│   └── reporting/
│       ├── report_generator.py  ← Relatório HTML
│       └── presentation_generator.py ← Gerador PPTX (lê 17_APRESENTACAO.md)
│
├── data/
│   ├── train.txt / val.txt / test.txt
│   └── dataset_stats.json
│
└── models/
    └── *.pt + *_metadata.json
```

---

## 🔧 Qual modelo usar

| Modelo | Index | PER | WER | Quando usar |
|--------|-------|-----|-----|-------------|
| **Exp9** | 11 | 0,58% | **4,96%** | Precisão por palavra — NLP, lookup, TTS quando WER importa |
| **Exp104b** | 18 | **0,49%** | 5,43% | Precisão por fonema — análise linguística, síntese de fala |
| Exp106 | 20 | 0,58% | 6,12% | Latência crítica — 30,2 w/s (2,58× mais rápido) |

---

**Última atualização**: 2026-02-28
**Status**: Phase 6C concluída — SOTA PER 0,49% (Exp104b) | Documentação consolidada (25→8 arquivos)
