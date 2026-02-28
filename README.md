# FG2P — Conversão Grapheme-to-Phoneme para Português Brasileiro

Modelo neural BiLSTM Encoder-Decoder + Attention para converter texto PT-BR em transcrição fonética IPA. 

**🏆 SOTA**: **PER 0.58%** (Exp9, 9.7M params) | Exp2: 0.60% (17.2M) | Exp6: 0.63% (4.3M, budget)

**Breakthrough**: Exp9 (Intermediate + Distance-Aware Loss λ=0.2) alcança NOVO SOTA PT-BR G2P, superando LatPhon (0.86%) com test set 57× maior.

---

## 🚀 Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Treinar (Exp9 — SOTA recomendado)
python src\train.py --config conf/config_exp9_intermediate_distance_aware.json

# Avaliar (full evaluation)
python src\inference.py

# Teste rápido com neologismos
python src\inference_light.py --model-index 0 --test data/neologisms_test.tsv

# Validar dataset saúde
python src\dataset_health_check.py --input dicts/pt-br.tsv

# Relatório HTML
python src\reporting\report_generator.py

# Gestão de experimentos
python src\manage_experiments.py --list
python src\manage_experiments.py --prune-incomplete --dry-run
```

---

## 🎯 Capacidades Principais

### **1. SOTA G2P Model** (Exp9)
- PER 0.58% | WER 4.96% | Accuracy 95.04%
- 9.7M params (optimal ROI vs capacity)
- BiLSTM Encoder-Decoder + Attention + Distance-Aware Loss
- Production-ready com checkpointing automático

### **2. Neologisms & OOV Testing** (NEW - Phase 5A) 🆕
- `inference_light.py` — Teste rápido de palavras novas
- Detecção de palavras inventadas vs dicionário
- Confidence score + nearest match suggestions
- Uso: Avaliar performance em nomes, termos técnicos, loanwords

### **3. Dataset Quality Assurance** (NEW - Phase 5A) 🆕
- `dataset_health_check.py` — Valida dicionário
- Detecta duplicatas, typos, encoding issues
- HTML report com sugestões de correção
- Estatísticas de cobertura (phonemes, n-grams)

### **4. Comprehensive Analysis Pipeline**
- HTML reports com gráficos de convergência
- Métricas graduadas PanPhon (Classes A/B/C/D)
- Error analysis automático (confusões estruturadas)
- Comparação multi-modelo com SOTA literatura

---

## 📊 Resultados Destacados

| Exp | Params | Técnica | PER↓ | WER↓ | Acc↑ | ROI |
|-----|--------|---------|------|------|------|-----|
| **Exp9** | 9.7M | Intermediate + DA Loss λ=0.2 | **0.58%** | **4.96%** | **95.04%** | ⭐⭐⭐⭐⭐ **SOTA** |
| **Exp2** | 17.2M | Extended | 0.60% | 4.98% | 95.02% | ⭐⭐⭐ High capacity |
| **Exp6** | 4.3M | Baseline + DA Loss λ=0.1 | 0.63% | 5.35% | 94.65% | ⭐⭐⭐⭐ Budget |
| **Exp10** | 17.2M | Extended + DA Loss λ=0.2 | 0.61% | 5.25% | 94.75% | ⭐ Negative ROI |
| **Exp5** | 9.7M | Intermediate | 0.63% | 5.38% | 94.62% | ⭐⭐⭐ Sweet spot |
| **Exp1** | 4.3M | Baseline | 0.66% | 5.65% | 94.35% | ⭐⭐⭐ Simple |

**Key Insights**: 
- ✅ **Exp9 (9.7M) confirmado como SOTA**: Melhor PER/WER/Acc, optimal ROI
- ❌ **DA Loss não escala para high-capacity**: Exp10 (17.2M) pior que Exp2 e Exp9
- 💡 **Saturação em ~0.58% PER**: Limite alcançado com arquitetura atual
- 🎯 **Próxima fronteira**: Decomposed encoding (Exp11-13) para superar 0.58%

**Análise detalhada**: [docs/04_EXPERIMENTS.md](docs/04_EXPERIMENTS.md)

---

## 📚 Documentação (Estrutura de Artigo Científico)

**Leitura recomendada em ordem**:

1. **[docs/01_OVERVIEW.md](docs/01_OVERVIEW.md)** — Introdução, dataset, discovery 60/10/30
2. **[docs/02_ARCHITECTURE.md](docs/02_ARCHITECTURE.md)** — BiLSTM, Attention, Embeddings, tratamento sequências
3. **[docs/03_METRICS.md](docs/03_METRICS.md)** — PER, WER, métricas graduadas PanPhon (Classes A/B/C/D)
4. **[docs/04_EXPERIMENTS.md](docs/04_EXPERIMENTS.md)** — Exp0-9 design, resultados, RFC_EXP6, análise comparativa
5. **[docs/05_THEORY.md](docs/05_THEORY.md)** — Fundações G2P, Loss functions, Features articulatórias
6. **[docs/06_REFERENCES.md](docs/06_REFERENCES.md)** — Bibliography (SOTA, datasets, tools)

**Status & Roadmap**: [TODO.md](TODO.md) — Fonte única de status, Phase 3 schedule

**Benchmarks**: [docs/performance.json](docs/performance.json) — SOTA comparisons + hyperparameters
