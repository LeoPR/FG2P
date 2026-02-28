# FG2P — Conversão Grafema-para-Fonema para Português Brasileiro

Modelo neural BiLSTM Encoder-Decoder + Atenção Bahdanau para converter texto PT-BR em transcrição fonética IPA, com Distance-Aware Loss customizada.

**🏆 SOTA PER: 0.49%** (Exp104b, 9.7M params, 28.782 palavras de teste)
**🏆 SOTA WER: 4.96%** (Exp9, sem separadores silábicos)
**Teste**: 57× maior que LatPhon (0.86%) com mais confiança estatística

---

## 🚀 Quick Start

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Usar modelo SOTA (Exp104b, index=18)
python src/inference_light.py --index 18 --word computador
# → k õ p u t a ˈ d o x .

# Modo interativo
python src/inference_light.py --index 18 --interactive

# Avaliar em banco de generalização
python src/inference_light.py --index 18 --neologisms docs/generalization_test.tsv

# Relatório HTML completo
python src/reporting/report_generator.py

# Gerar apresentação PPTX
python src/reporting/presentation_generator.py --mode full      # 31 slides
python src/reporting/presentation_generator.py --mode compact   # 20 slides (10 min)
```

---

## 📊 Resultados Principais (Phase 6C Completa)

| Exp | Params | Técnica | PER↓ | WER↓ | Uso recomendado |
|-----|--------|---------|------|------|-----------------|
| **Exp104b** | 9.7M | DA Loss + dist custom | **0.49%** | 5.43% | **SOTA PER — TTS, alinhamento** |
| **Exp9** | 9.7M | DA Loss λ=0.2 | 0.58% | **4.96%** | **SOTA WER — NLP, busca** |
| Exp106 | 9.7M | DA + sem hífen | 0.58% | 6.12% | **Velocidade: 30.2 w/s ⚡ (2.58×)** |
| Exp105 | 9.7M | DA + 50% dados | 0.54% | 5.87% | Deploy com menos dados |
| Exp102 | 9.7M | CE + separadores | 0.52% | 5.79% | Referência |
| Exp5 | 9.7M | CrossEntropy | 0.63% | 5.38% | Baseline |

**Descobertas-chave**:
- Split 60/10/30 supera 70/10/20 em **−41% PER**
- Distance-Aware Loss: pesa erros por distância articulatória (e→ɛ ≠ e→k)
- Separadores silábicos criam trade-off Pareto irredutível (PER↓, WER↑)
- 50% dados → apenas +0.05% PER — modelo robusto
- Sem hífen → 2.58× velocidade, apenas +0.04% PER

---

## 📚 Documentação

```
docs/
├── 01_OVERVIEW.md            ← Introdução, dataset, resultados completos
├── 02_ARCHITECTURE.md        ← BiLSTM + Atenção Bahdanau
├── 03_METRICS.md             ← PER, WER, métricas fonológicas
├── 04_EXPERIMENTS.md         ← Exp0-106, design e resultados
├── 05_THEORY.md              ← G2P, Loss functions, features articulatórias
├── 06_PREPROCESSING.md       ← Normalização, charset, filtros
├── 07_STRUCTURAL_ANALYSIS.md ← Problema d(.,ˈ)=0 e solução (Exp104b)
├── 09_CONTINUOUS_PHONETIC_SPACE.md ← Espaço fonético 7D (Phase 7 — futuro)
├── 11_CORPUS_AUDIT.md        ← Auditoria corpus: regra ɣ/x, NFD/NFC
├── 16_SCIENTIFIC_ARTICLE.md  ← Artigo acadêmico completo
└── 17_APRESENTACAO_MERGED.md ← Apresentação PPTX [modes: full, compact]
```

**Leitura recomendada**: [docs/01_OVERVIEW.md](docs/01_OVERVIEW.md)

---

## 🏗️ Arquitetura Resumida

```
"c a s a" → [Embedding 192D] → [BiLSTM Encoder 2×384D] → [Atenção Bahdanau]
                                                              ↓
                                          [LSTM Decoder 2×384D] → k a z a
```

**Loss**: `L = L_CE + λ · d(ŷ, y) · p(ŷ)` — CrossEntropy + penalidade articulatória
**Dataset**: 95.937 palavras PT-BR (dicts/pt-br.tsv) | Split: 60/10/30

---

## 🔧 Comandos Úteis

```bash
# Treinar experimento
python src/train.py --config conf/config_exp104b_intermediate_sep_da_custom_dist.json

# Avaliação completa (WER/PER no test set)
python src/inference.py

# Listar modelos treinados
python src/manage_experiments.py --list

# Benchmark de velocidade
python src/benchmark_inference.py
```

---

**Status**: Phase 6C Completa ✅ | Phase 7 Planejada (espaço fonético 7D contínuo)
**Documentação**: [docs/](docs/) | **Roadmap**: [TODO.md](TODO.md) | **Status**: [STATUS.md](STATUS.md)
