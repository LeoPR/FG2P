# STATUS — FG2P Project Overview

**Data**: 2026-02-28
**Status**: Phase 6C COMPLETA ✅ | Docs reorganizados | Phase 7 iniciada (Exp107 planejado)
**SOTA WER**: **Exp9** (PER 0.58%, WER **4.96%**, Acc 95.04%, 9.7M params)
**SOTA PER**: **Exp104b** (PER **0.49%**, WER 5.43%, Acc 94.57%, 9.7M params)
**Finding Phase 6B**: Override de distâncias APÓS normalização → PER 0.49% NOVO SOTA (Exp104b)
**Finding Phase 6C**: 50% dados → +0.05% PER (robusto); sem hífen → +0.04% PER, 2.58× speed
**Próximo**: Exp107 (95% treino) — comparação com LatPhon 2025 (PER 0.89%, N=500)

---

## 🎯 Missão

Desenvolver modelo G2P (Grapheme-to-Phoneme) SOTA para Português Brasileiro usando arquitetura BiLSTM Encoder-Decoder + Attention, com foco em:
1. Minimizar PER (Phoneme Error Rate)
2. Métricas linguisticamente graduadas (PanPhon features)
3. Reproducibilidade científica total
4. ROI computacional (performance vs params)

---

## 🏆 Achievements

### **SOTA Alcançado** ✅
- **PER: 0.49%** (Exp104b) — SOTA PER final | Supera LatPhon 2025 (0.89%) em -45%
- **WER: 4.96%** (Exp9) — SOTA WER | Test set 28.782 palavras (57× maior que LatPhon)
- **Speed: 30.2 w/s** (Exp106) — 2.58× mais rápido sem perda fonológica significativa
- **Robusto**: 50% dados → PER 0.54% (+0.05%) — modelo não memoriza, generaliza
- **Generalização OOV PT-BR**: 5/5 palavras reais fora do vocabulário corretas (100%)

### **Descobertas Científicas** 🔬

#### 1. Split Ratio Impact (Phase 1)
- **60/10/30 > 70/10/20**: -41% PER improvement
- **Conclusão**: Mais dados de teste → validação estatística superior

#### 2. Capacity Ceiling (Phase 1-2)
```
4.3M params:  0.66% PER [baseline]
9.7M params:  0.58% PER [SWEET SPOT ✓]
17.2M params: 0.60% PER [diminishing returns]
```
- **Saturação em ~0.58% PER** com arquitetura LSTM atual

#### 3. Distance-Aware Loss (Phase 3-4)
- ✅ **Funciona em baseline/intermediate** (4.3M-9.7M params)
- ❌ **Falha em high-capacity** (17.2M params)
- **λ optimal = 0.2** (lambda sweep Exp7: 0.05 < 0.20 < 0.50)
- **Mecanismo**: Regularização implícita → contraprodutivo em modelos grandes

#### 4. PanPhon Features (Phase 2-3)
- **Neutral vs learned embeddings** em PT-BR (Exp3 ≈ Exp1)
- **Fixed features prejudicam** (Exp4 pior que Exp3)
- **Conclusão**: PT-BR ortografia regular → learned embeddings suficientes

#### 5. Error Pattern (consistente todos experimentos)
- **65-70% erros**: Confusões vogais médias (ɛ↔e, ɔ↔o)
- **Linguisticamente justificado**: Sem contexto semântico, ambiguidade inerente
- **Implicação**: 0.58% PER pode ser limite "natural" sem context awareness

---

## 📊 Experimentos Completos (20 modelos)

| Exp | PER | WER | Acc | Params | Key Feature | Status |
|-----|-----|-----|-----|--------|-------------|--------|
| **Exp104b** | **0.49%** | 5.43% | 94.57% | 9.7M | DA Loss + sep + override pós-norm | ✅ **SOTA PER** |
| Exp105 | 0.54% | 5.87% | 94.13% | 9.7M | DA Loss + sep, 50% treino | ✅ Ablação dados |
| **Exp102** | 0.52% | 5.79% | 94.21% | 9.7M | Intermediate + sep + CE | ✅ Sep baseline |
| Exp103 | 0.53% | 5.73% | 94.27% | 9.7M | Intermediate + sep + DA λ=0.2 | ✅ Phase 6A |
| **Exp9** | 0.58% | **4.96%** | **95.04%** | 9.7M | Intermediate + DA λ=0.2 | ✅ **SOTA WER** |
| Exp106 | 0.58% | 6.12% | 93.88% | 9.7M | DA Loss + sep, 50% treino, sem - | ✅ **30.2 w/s ⚡** |
| Exp101 | 0.53% | 5.99% | 94.01% | 4.3M | Baseline + sep | ✅ Sep. diagnóstico |
| Exp2 | 0.60% | 4.98% | 95.02% | 17.2M | Extended capacity | ✅ High-cap baseline |
| Exp10 | 0.61% | 5.25% | 94.75% | 17.2M | Extended + DA λ=0.2 | ✅ Negative ROI |
| Exp6 | 0.63% | 5.35% | 94.65% | 4.3M | Baseline + DA λ=0.1 | ✅ Budget option |
| Exp5 | 0.63% | 5.38% | 94.62% | 9.7M | Intermediate | ✅ Capacity test |
| Exp8 | 0.65% | 5.62% | 94.38% | 4.3M | PanPhon + DA λ=0.2 | ✅ Features test |
| Exp1 | 0.66% | 5.65% | 94.35% | 4.3M | Baseline 60/10/30 | ✅ Reference |
| Exp3 | 0.66% | 5.45% | 94.55% | 4.3M | PanPhon trainable | ✅ Linguistic features |
| Exp7 | 0.68-0.73% | varies | varies | 4.3M | Lambda sweep | ✅ Hyperopt |
| Exp4 | 0.71% | 6.02% | 93.98% | 4.3M | PanPhon fixed | ✅ Ablation |
| Exp11 | 0.97% | 7.53% | 92.47% | 4.3M | Baseline + decomposed NFD | ✅ NFD incompatível |
| Exp0 | 1.12% | 9.37% | 90.63% | 4.3M | Baseline 70/10/20 | ✅ Initial baseline |

**Progresso total**: -54% PER (Exp0 1.12% → Exp102 0.52%) | SOTA WER: Exp9 (4.96%) | 17 modelos treinados

---

## 🚀 Phase 6A CONCLUÍDA — Sep + DA Loss Combination

### Experimentos Planejados

#### **Exp11**: Baseline + Decomposed NFD
- **Config**: `config_exp11_baseline_decomposed.json`
- **Hipótese**: NFD Unicode (á→a+´) facilita aprendizado diacritics
- **Compara com**: Exp1 (0.66% PER)
- **Target**: 0.60-0.64% PER
- **Status**: Re-iniciado com `keep_syllable_separators=true`

#### **Exp101**: Baseline + Separadores (controle direto)
- **Config**: `config_exp101_baseline_60split_separators.json`
- **Hipótese**: Separadores mudam PER/WER mesmo com encoding raw
- **Compara com**: Exp1 (baseline raw sem separadores)

#### **Exp12**: PanPhon + Decomposed
- **Config**: `config_exp12_panphon_decomposed.json`
- **Hipótese**: Sinergia features linguísticas + diacritics explícitos
- **Compara com**: Exp3 (0.66%) e Exp11
- **Target**: 0.58-0.62% PER
- **Critical**: Se ≥ Exp9 → 4.3M params rivalizam 9.7M = OPTIMAL ROI

#### **Exp13**: SOTA + Decomposed (FRONTIER PUSH)
- **Config**: `config_exp13_intermediate_distance_aware_decomposed.json`
- **Hipótese**: NFD + SOTA architecture → **NEW ABSOLUTE SOTA**
- **Compara com**: Exp9 (0.58% PER)
- **Target**: **< 0.55% PER** (breakthrough PT-BR G2P)

**Estratégia**: Design fatorial 2×2 [raw/decomposed] × [learned/PanPhon]

---

## ✅ Hotfixes Recentes

- **BUG 1 (cache collision)**: resolvido com nomes sensíveis a `encoding + separadores + split + seed`
- **BUG 2 (separadores de sílaba)**: flag opcional `keep_syllable_separators`
	- Default: `false` (compatível com Exp0-10)
	- Exp11+ ativado para testes com separadores
- **Observação**: revalidar impacto direto via Exp101 (baseline raw + separadores)
- **Relatório HTML**: ordenação generalizada por índice de experimento (evita ordem lexicográfica `exp1, exp11, exp2`) e sort robusto para colunas numéricas (PER/WER/Accuracy e métricas graduadas)

### Impacto nos experimentos já feitos (avaliação retroativa)
- **Exp0–Exp10**: **sem impacto em métricas históricas** (PER/WER/Acc mantidos).
- Motivo: todos foram treinados antes da flag de separadores e permanecem válidos como baseline histórico.
- **Cache**: mudança afeta apenas arquivos de inspeção em `data/` (evita sobrescrita), não altera checkpoints já salvos em `models/`.
- **Comparabilidade**:
	- Exp11 run antigo (`20260222_161238`): decomposed sem separadores.
	- Exp11 run atual (`20260222_201314`): decomposed com separadores.
	- Esses dois runs devem ser tratados como condições experimentais diferentes.

---

## 📈 Métricas vs Literatura

| Sistema | Idioma | Test Size | PER | Params | Notas |
|---------|--------|-----------|-----|--------|-------|
| **FG2P Exp104b** | PT-BR | **28.8k** | **0.49%** | 9.7M | ✅ **SOTA PER** |
| **FG2P Exp9** | PT-BR | **28.8k** | 0.58% | 9.7M | ✅ **SOTA WER 4.96%** |
| LatPhon 2025 | PT-BR | 500 | 0.89% | ~7M | N=500 — menor comparabilidade |
| DeepPhonemizer | IT | ~77k | 0.40% | 229M | Romance: similar ao PT-BR |
| ByT5 Small | 100+ | varies | 8.90% | 299M | Multilingual zero-shot |

**Destaques**:
- ✅ **Supera LatPhon SOTA em -45% PER** (0.49% vs 0.89%)
- ✅ Test set **57× maior** que LatPhon (28.8k vs 500) — mais robusto
- ✅ Perto de DeepPhonemizer IT com **23× menos params**

---

## 🚀 Phase 7 — Próximos Experimentos

### Exp107 — MaxData 95% treino [PENDENTE]
- **Config**: `conf/config_exp107_maxdata_95train.json`
- **Split**: train=95% (~91.140 palavras), val=4%, test=1% (~960 palavras)
- **Hipótese**: Mais dados → PER < 0.49% (SOTA atual)
- **Objetivo estratégico**: Comparação metodológica com LatPhon 2025 (test N=500)
- **Executar**: `python src/train.py --config conf/config_exp107_maxdata_95train.json`
- **Após treino**: Atualizar tabela SOTA + slides + ARTICLE.md §5

---

## 📋 Log de Decisões de Documentação (2026-02-28)

Registro das decisões tomadas para não perder o raciocínio ao retomar o projeto.

### Reorganização docs/ (Fase 1)
- **Feito**: 12 arquivos movidos para subpastas `article/`, `presentation/`, `report/`, `data/`
- **Motivo**: 8 arquivos no mesmo nível era difícil de navegar
- **Resultado**: `docs/INDEX.md` como hub de entrada

### Slides — Decisões Tomadas (2026-02-28)
| Elemento | Decisão | Razão |
|----------|---------|-------|
| **Exp103 nos slides** | Removido de tabelas de ranking; aparece só como nota de rodapé | Não vence em nenhuma métrica — histórico apenas |
| **Slide "Distâncias"** | Exp103 → "Sem override (baseline)" | Anonimiza mas preserva a narrativa do bug |
| **"Ranking Final"** | Convertido de lista de bullets para tabela (PER/WER/Speed/Caso) | Mais informação em menos espaço |
| **ASCII chart** | Mantido intacto | Chart está limpo e conta a história bem |
| **Nota abaixo do chart** | Adicionada 1 linha com Phase 6C + Fase 7 hint | Contexto sem poluir o visual |

### O que NÃO foi tocado (e por quê)
- `src/reporting/presentation_generator.py` — slides hardcoded (attention, CE, DA example) não mudam sem risco
- Modelos existentes — nunca modificar arquivos .pt
- Configs existentes (Exp0–Exp106) — configs são imutáveis após treino

### Backups disponíveis
- `docs/presentation/PRESENTATION.md.bak_20260228` — estado dos slides antes das edições de 2026-02-28
- **Para restaurar**: `cp docs/presentation/PRESENTATION.md.bak_20260228 docs/presentation/PRESENTATION.md`
- **Para apagar o backup** (quando confirmar que está tudo OK): `rm docs/presentation/PRESENTATION.md.bak_20260228`

---

## 🛠️ Infraestrutura & Tools

### Core Pipeline
- ✅ **train.py**: Training loop com early stopping, checkpointing
- ✅ **inference.py**: Batch evaluation + metrics + error analysis
- ✅ **analysis.py**: Training history plots + convergence analysis
- ✅ **report_generator.py**: HTML reports com métricas graduadas
- ✅ **manage_experiments.py**: Experiment tracking + cleanup

### Data & Registry
- ✅ **G2PCorpus**: Stratified splits, caching, statistical validation
- ✅ **FileRegistry**: Timestamped artifacts, metadata tracking
- ✅ **dataset_stats.json**: Reproducible cache (phoneme coverage, chi-square)

### Quality Assurance
- ✅ Reproducibilidade total (seed=42, deterministic)
- ✅ Configs JSON versionados com todos hyperparameters
- ✅ Metadata JSON por modelo (architecture, training, metrics)
- ✅ Git-tracked configs + .gitignore models/results

---

## 📁 Repository Structure

```
FG2P/
├── src/                          # Source code
│   ├── train.py                  # Training pipeline
│   ├── inference.py              # Evaluation + metrics
│   ├── g2p.py                    # Model architecture + corpus
│   ├── analysis.py               # Training analysis
│   └── utils.py                  # Logging, paths, helpers
├── config_*.json                 # Experiment configs (reproducible)
├── models/                       # Model checkpoints .pt + metadata
├── results/                      # Evaluations, predictions, analysis
├── data/                         # Cached dataset splits
├── dicts/                        # Source dictionary pt-br.tsv
├── docs/                         # Documentation (scientific paper structure)
│   ├── 01_LITERATURA.md
│   ├── 02_ARQUITETURA.md
│   ├── 03_IMPLEMENTACAO.md
│   ├── 04_EXPERIMENTOS.md
│   ├── 05_BENCHMARKS.md
│   └── 06_ANALISE_LINGUISTICA.md
├── TODO.md                       # Roadmap + status (source of truth)
├── RESUMO_EXPERIMENTOS.md        # Consolidated experiment summary
├── EXPERIMENTOS_DECOMPOSED.md    # Phase 5 strategy
└── README.md                     # Quick start

```

---

## 🎯 Próximos Milestones

### Curto Prazo (1-2 semanas)
1. ✅ **Exp11-13 training** (decomposed encoding tests)
2. 📊 **Comparative analysis** Exp11-13 vs baselines
3. 🎯 **Decision**: Adoptar decomposed como default SE beneficial

### Médio Prazo (2-4 semanas)
4. 📝 **Paper draft** (estrutura já em docs/)
5. 🎨 **PowerPoint generator** para apresentações científicas
6. 🔬 **Métricas graduadas avançadas** (erosão cumulativa)

### Longo Prazo (backlog)
7. 🤖 **Transformer architecture** (compare vs LSTM)
8. 🌐 **Multi-task learning** (tonicidade, syllabification)
9. 📦 **Production API** (Flask/FastAPI deployment)

---

## 📚 Documentação

### Principal
- **[README.md](README.md)**: Quick start + resultados destacados
- **[TODO.md](TODO.md)**: Roadmap completo + tracking
- **[RESUMO_EXPERIMENTOS.md](RESUMO_EXPERIMENTOS.md)**: Análise consolidada Exp0-10

### Científica (paper structure)
- **[docs/](docs/)**: 6 documentos (Literatura → Análise linguística)
- **[EXPERIMENTOS_DECOMPOSED.md](EXPERIMENTOS_DECOMPOSED.md)**: Phase 5 strategy

### Configuration
- **[config_*.json](./config_exp*.json)**: 13 experiment configs (reproducible)
- **[CONFIG_README.md](CONFIG_README.md)**: Config file format specification

---

## 👥 Team & Contact

**Desenvolvedor Principal**: [Nome]  
**Orientação**: [Orientador]  
**Instituição**: [Universidade]  
**Curso**: [Programa]

---

**Última atualização**: 2026-02-22 16:40  
**Gerado automaticamente** a partir do estado atual do projeto.
