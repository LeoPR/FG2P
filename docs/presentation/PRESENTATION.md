---
marp: true
theme: default
paginate: true
backgroundColor: "#fff"
# [UNIFIED] FG2P Presentation — automatically generates full or compact
# Generator: python src/reporting/presentation_generator.py --mode full|compact
author: "Leonardo Marques de Souza"
professor: "Prof. Dr. André Eugênio Lazzaretti"
institution: "Universidade Tecnológica Federal do Paraná (UTFPR)"
course: "Deep Learning"
year: "Final de 2025"
sota_per: "0,49%"
sota_wer: "4,96%"
test_words: "28.782"
oov_accuracy: "100%"
style: |
  section { font-size: 22px; }
  h1 { color: #1a3a6b; }
  h2 { color: #1a3a6b; border-bottom: 2px solid #1a3a6b; padding-bottom: 4px; }
  table { font-size: 18px; width: 100%; }
  code { background: #f4f4f4; padding: 2px 4px; border-radius: 2px; }
  pre { background: #1e1e2e; color: #cdd6f4; padding: 16px; border-radius: 6px; }
---

[modes: full, compact]

# FG2P
## Conversão Grafema-Fonema para o Português Brasileiro

**Modelo BiLSTM com Distance-Aware Loss**

> Dado o texto `"computador"`, o modelo produz `k õ p u t a ˈ d o x`

**Resultados alcançados**
- PER 0,49% · WER 4,96% · 28.782 palavras de teste
- 100% de acerto em palavras PT-BR fora do vocabulário de treino

---

[modes: full, compact]

<!-- ref: article/ARTICLE.md §1 — Introdução -->
## O que é o problema G2P?

**Grafema → Fonema**

```
Entrada  :  "casa"
Saída    :  k  a  z  a
```

```
Entrada  :  "cena"
Saída    :  s  e  n  ə
```

**O mesmo grafema `c` → dois fonemas diferentes**
- `c` antes de vogal posterior/baixa → `/k/`
- `c` antes de vogal anterior alta → `/s/`

O modelo não memoriza palavras — aprende *regras* que generalizam.

---

[modes: full, compact]

<!-- ref: article/ARTICLE.md §1 + §2.3 — Desafios PT-BR, Regra ɣ/x -->
## Por que PT-BR é difícil?

### 3 desafios principais

| Desafio | Exemplo | Regra |
|---------|---------|-------|
| Ambiguidade grafêmica | "cama" vs "cena" | c→k ou c→s |
| Dependência de posição | "churrasco" (rr→x) · "computador" (r coda final→x) · "borboleta" (r+vozeada→ɣ) | r muda conforme posição silábica |
| Neutralização vocálica | "seco" (ɛ tônico) vs "secada" (e átono) | ɛ tônico → e/ɪ átono |

**Consequência prática**: ~60% dos erros do modelo são neutralizações vocálicas legítimas — ambiguidade irredutível sem contexto prosódico.

---

[modes: full, compact]

<!-- ref: article/ARTICLE.md §2.1 + §2.2 — Corpus e Split -->
## Os Dados

**95.937 pares** (palavra, transcrição IPA)
- Fonte: dicionário fonético PT-BR
- Charset treinado: `a–z` (exceto k, w, y) + `ç á à â ã é ê í ó ô õ ú ü`

**Divisão: 60% / 10% / 30%** (estratificada)

| Subconjunto | Tamanho |
|-------------|---------|
| Treino | 57.561 palavras |
| Validação | 9.594 palavras |
| Teste | **28.782 palavras** |

**Descoberta**: split 60/10/30 supera 70/10/20 em **−41% PER**
→ Mais dados de treino nem sempre = modelo melhor

---

[modes: full, compact]

<!-- ref: article/ARTICLE.md §3 — Arquitetura BiLSTM -->
## A Arquitetura

```
"c a s a"   ←— grafemas de entrada
    ↓
[ Embedding ]  192D: cada letra → vetor
    ↓
[ BiLSTM Encoder ]  2 camadas, 384D
  → lê a palavra nos dois sentidos
  → "c" em "cama" vs "cena" → representações diferentes
    ↓
[ Atenção Bahdanau ]
  → decoder "olha" para onde precisa na palavra
  → aprende: "nh" → foco nas duas letras juntas
    ↓
[ LSTM Decoder ]  gera fonema por fonema
    ↓
k  a  z  a   ←— fonemas de saída
```

---

[modes: full]

<!-- ref: article/ARTICLE.md §3.2 — Mecanismo de Atenção Bahdanau -->
## Por que Atenção?

### O Mecanismo de Bahdanau

**Atenção Bahdanau** = sistema que calcula pesos dinâmicos (0–1) para cada letra da palavra de entrada. O decoder usa esses pesos para decidir "onde olhar" ao gerar cada fonema:

```
Para cada fonema gerado, o decoder pergunta:
  "Qual letra da palavra é mais importante para gerar este som?"

Bahdanau responde com pesos:
  s=0.70 significa "preste atenção em 's' com 70% de confiança"
  c=0.20 significa "preste atenção em 'c' com 20% de confiança"
  o=0.02 significa "ignore 'o', é só 2% relevante"
```

Esses pesos são **aprendidos pelo modelo durante o treino**, não pré-definidos.

### Por que funciona:

**Sem atenção**: encoder cria 1 vetor resumido para TODA a palavra
```
"biscoito" → encoder processa: b-i-s-c-o-i-t-o
                                       ↓ (BiLSTM nas 2 direções)
            Comprime tudo em 1 vetor (384 números)
            Este vetor é um "resumo geral" da palavra inteira

Problema: 384 números precisam guardar TUDO sobre a palavra:
  "qual letra está na posição 3? qual na 4? padrão sc?"
  É como comprimir um livro inteiro em um parágrafo!

Resultado: decoder recebe esse resumo vago e gera cada fonema
  Ao gerar /k/: não consegue verificar "letra 3 é s, letra 4 é c"
  Resultado: trata como símbolos isolados → /s/ + /k/ (ERRADO!)
```

**Com atenção Bahdanau**: decoder pode consultar TODOS os estados intermediários
```
"biscoito" → encoder TAMBÉM comprime (384D por letra):
  b(384D)  i(384D)  s(384D)  c(384D)  o(384D)  i(384D)  t(384D)  o(384D)
  ↓ (cada um é um "resumo" dessa letra + contexto)

Ao gerar /k/ na posição 3:
  Atenção calcula pesos dinâmicos:
    b=0.02  i=0.03  s=0.70  c=0.20  o=0.02  i=0.02  t=0.01  o=0.00
  ↓
  Decoder acessa: 0.70 × s(384D) + 0.20 × c(384D) + ...
  Resultado: "olha fortemente para s e c" → reconhece padrão "sc"
  ↓
  Gera /k/ corretamente!
```

Essencial para padrões de alinhamento n:1 como:
- `ch` → `/ʃ/` (2 grafemas → 1 fonema)
- `nh` → `/ɲ/` (2 grafemas → 1 fonema)
- `x` → `/ʃ/` ou `/ks/` dependendo do contexto

---

[modes: full]

<!-- ref: article/ARTICLE.md §4.1 — Limitação da CrossEntropy -->
## O Problema da CrossEntropy

**CrossEntropy: intervalo [0, +∞) matematicamente — [0, ~16] computacionalmente — SEM gradação fonológica**

**Fórmula**: `L_CE = -log(p_correct)` onde `p_correct ∈ (0, 1]`

```
Matematicamente:
  • p = 1.0 (predição correta)        →  L_CE = 0
  • p = 0.5 (incerteza)               →  L_CE = 0.69
  • p → 0 (predição muito errada)     →  L_CE → +∞

Computacionalmente (com log-sum-exp trick do PyTorch):
  • p nunca é exatamente 0            →  L_CE máx ~16 típico
  • Numericamente estável (sem overflow)
```

```
Cenário A — Erro de 1 feature (quase certo):
  Predição: ɛ (aberta)    Correto: e (fechada)
  Diferença: mesma posição, só altura da língua
  CrossEntropy: penalidade ≈ 0.7 (exemplo)

Cenário B — Erro catastrófico (completamente errado):
  Predição: k (oclusiva velar)    Correto: a (vogal baixa)
  Diferença: classes fonológicas opostas
  CrossEntropy: penalidade ≈ 3.5 (exemplo) ← PROBLEMA: SEM relação à distância fonológica!
```

**O Problema**: Ambos erros recebem penalidades contínuas mas **sem discriminação fonológica**!
→ Modelo não consegue aprender que ɛ→e (quase acerto) é fundamentalmente diferente de a→k
→ Trata erros de "quase acerto" e "catastróficos" como pertencentes à mesma escala de penalidade

---

[modes: full, compact | label: da_loss_full, da_loss_compact]

<!-- ref: article/ARTICLE.md §4.2 — Distance-Aware Loss -->
## A Distance-Aware Loss

**Solução**: Adicionar um sinal fonológico ao erro

```
L = L_CE + λ · d(ŷ, y) · p(ŷ)
```

**O que cada parte faz:**
- **L_CE**: Penalidade base do CrossEntropy
- **λ (lambda)**: Peso do sinal fonológico — empiricamente ótimo = **0,20**
- **d(ŷ, y)**: Distância articulatória entre predito e correto
- **p(ŷ)**: Confiança do modelo na sua predição (softmax)

→ Permite ao modelo aprender que `ɛ→e` é "quase OK" enquanto `k→a` é "crítico"

---

[modes: full]

<!-- ref: article/ARTICLE.md §4.2 — Função de distância fonológica -->
## DA Loss — Explicação de d(ŷ, y)

**A "régua fonológica" que mede distância entre fonemas**

```
Tabela de Distâncias (PanPhon 24 features articulatórios):

d(e, ɛ) = 0.10    ← mesma posição (média anterior), só abertura
d(p, b) = 0.05    ← oclusivas bilabiais, só vozeamento
d(s, ʃ) = 0.15    ← fricativas, ponto diferente (alveolar vs palatal)
d(a, k) = 0.90    ← vogal baixa vs. oclusiva velar — classe oposta!
d(., ˈ) = 1.0     ← símbolos estruturais (máxima distância)
```

**Insight**: Erros "próximos" (d pequeno) recebem penalidade fraca
→ Erros "distantes" (d grande) recebem penalidade forte

---

[modes: full]

<!-- ref: article/ARTICLE.md §4.2 — Exemplo numérico DA Loss -->
## DA Loss — Exemplo Passo a Passo

**Palavra**: "cena" · **Posição**: grafema "c" · **Correto**: `/s/`

```
PASSO 1 — Modelo produz distribuição de probabilidades (softmax):
  s=42%   ʃ=35%   z=12%   t=6%   ...
  ↓ argmax (escolhe o maior)
  Predição: ʃ  (errado! esperado era s)

PASSO 2 — CrossEntropy (só olha para o correto, que é s):
  p(s) = 0.42   ← a probabilidade que o modelo deu para o fonema CORRETO
  L_CE = −log(0.42) = 0.87   ← penaliza por não ter escolhido s

PASSO 3 — Distância articulatória (ʃ ≠ s, mas próximos):
  d(ʃ, s) = 0.15   ← fricativas, mesmo modo, só ponto diferente

PASSO 4 — Term extra da DA Loss:
  λ × d × p(pred) = 0.20 × 0.15 × 0.35 = 0.010
                    (lambda) (distância) (confiança do modelo em ʃ)

TOTAL: 0.87 + 0.01 = 0.88
```

**Comparação**: Se tivesse predito `k` (completamente errado)?
→ d(k,s) = 0.80 → extra = 0.056 → total = 0.93  (penalidade 5% maior!)

**Detalhe importante**: CE só penaliza "não escolheu s", mas DA Loss também penaliza "escolheu algo muito distante de s"

---

[modes: full]

## Por que multiplicar por $p(\hat{y})$?

**O fator de confiança evita punir incerteza honesta:**

```
Época 5 — modelo incerto:
  Prediz k com 52% → penalidade moderada
  "Ainda aprendendo, só corrigindo a direção"

Época 40 — modelo confiante:
  Prediz k com 91% → penalidade alta
  "Você está MUITO certo de algo completamente errado"
```

Análogo ao **label smoothing**, mas proporcional à distância fonológica — não uniforme.

---

[modes: full]

<!-- ref: article/ARTICLE.md §8.1 — Separadores silábicos -->
## Separadores Silábicos — O Trade-off

**Adicionando `.` como token de fronteira silábica:**

```
Exp9   (sem sep):  k õ p u t a ˈ d o x
Exp102 (com sep):  k . õ p . u . t a . ˈ d o x .
```

**Efeito nas métricas:**

| Experimento | PER | WER | Comentário |
|---|-----|-----|---|
| Sem separadores (Exp9) | 0,58% | **4,96%** | SOTA WER ← menos erros de palavra |
| Com separadores (Exp102/104b) | **0,49–0,52%** | 5,43–5,79% | SOTA PER ← menos erros de fonema |

**Por quê?** Cada separador mal-posicionado = palavra inteira errada (WER binário)
→ Trade-off Pareto fundamental: *não* se resolve ajustando hiperparâmetros
→ DA Loss + separadores testados (Exp103/104b) — trade-off persiste independente da loss function

---

[modes: full]

<!-- ref: article/ARTICLE.md §4.3 — Override de distância -->
## Distâncias Customizadas — O Bug que Virou Feature

**Problema**: `.` e `ˈ` têm vetor **zero** no PanPhon

```
d(., ˈ) = 0.0   ← Loss não penaliza confusão entre eles!
d(., a) = d(ˈ, a)   ← mesmo valor — indistinguíveis
```

**Solução (Exp104b)**: override pós-normalização

```python
for sym in {'.', 'ˈ'}:
    for other in vocab:
        distance[sym][other] = 1.0  # distância máxima
```

**Resultado:**

| Exp | PER | WER | Erros .↔ˈ |
|-----|-----|-----|-----------|
| Sem override (baseline) | 0,53% | 5,73% | ~107 |
| Exp104 (bug: pré-normalização) | 0,54% | 5,88% | ~119 |
| **Exp104b (pós-normalização ✓)** | **0,49%** | **5,43%** | ~106 |

→ **NOVO SOTA PER: 0,49%** mesmo sem eliminar as confusões estruturais

---

[modes: full, compact | label: experiments_full, experiments_compact]

## Todos os Experimentos — Evolução

```
PER (%)
1.12 │ ● Exp0 (70/10/20)
0.66 │         ● Exp1 (60/10/30) −41%
0.63 │                 ● Exp5 (9.7M)  ● Exp6 (DA Loss)
0.61 │                         ● Exp7 (λ=0.20)
0.60 │                 ● Exp2 (17.2M)
0.58 │                                 ● Exp9 ← SOTA WER (4.96%)
0.52 │                                         ● Exp102 (sep)
0.49 │                                                 ● Exp104b ← SOTA PER
     └───────────────────────────────────────────────────────────────
      Fase1   Fase2    Fase3    Fase4    Fase5    Fase6
```

Phase 6C (ablações): Exp105 0,54% · Exp106 0,58% + **30.2 w/s** ⚡ · Phase 7: Exp107 em preparo (95% treino)

---

[modes: full]

## Ranking Final (Phase 6C Completa)

| Modelo | PER | WER | Speed | Melhor para |
|--------|-----|-----|-------|-------------|
| **Exp104b** | **0,49%** | 5,43% | 11.7 w/s | TTS, alinhamento fonético |
| **Exp9** | 0,58% | **4,96%** | 11.7 w/s | NLP, busca, indexação |
| **Exp106** | 0,58% | 6,12% | **30.2 w/s ⚡** | Latência crítica (2.58×) |
| Exp105 | 0,54% | 5,87% | 11.7 w/s | Validação com menos dados |
| Exp1 | 0,66% | 5,65% | — | Baseline histórico |

Todos: 9.7M parâmetros · BiLSTM 2 camadas · Embedding 192D

---

[modes: full]

<!-- ref: article/ARTICLE.md §8.4 — Ablação Exp105/Exp106 -->
## Phase 6C: Ablação de Dados e Ortografia (Exp105 + Exp106)

**Exp105 — Redução de Dados (50% treino)**
- Pergunta: Quanto dados é suficiente? Modelo robusto com 10% menos?
- Resultado: PER 0,54% (vs 0,49% Exp104b) — +0.05% modesto
- Conclusão: ✅ Modelo escala robustamente com menos dados

**Exp106 — Filtro de Hífen (50% treino, sem `-`)**
- Pergunta: Hífen (-) é ortográfico ou fonológico? Impacto?
- Entrada: "abaixo-assinado" → sem-hífen: "abaixoassinado"
- Resultado: PER 0,58% (vs 0,54% Exp105) — +0.04% negligenciável
- **🚀 Descoberta**: Velocidade 2,58x mais rápida! (30.2 w/s vs 11.7 w/s)
- Conclusão: ✅ Hífen não afeta fonologia. CharVocab menor (38 vs 39) permite speedup prático

**Impactos Resumidos:**
- 10% menos dados (Exp105): +0.05% PER — Aceitável ✅
- Sem hífen (Exp106): +0.04% PER, +2.58x Speed — Use para latência crítica ⚡

---

[modes: full, compact | label: generalization_full, generalization_compact]

<!-- ref: article/ARTICLE.md §7 — Avaliação de Generalização -->
## Avaliação de Generalização — Design

**Pergunta central**: o modelo *memorizou* ou *aprendeu regras*?

**31 palavras em 6 categorias:**

- **Generalização PT-BR** (9): Neologismos e portmanteaux com chars no vocab
- **Consoantes duplas** (5): lazzaretti, cappuccino → redução de geminadas
- **Anglicismos** (5): mouse, site → fonologia inglesa
- **Chars OOV** (3): wifi, yoga → falha esperada e documentada
- **PT-BR reais OOV** (5): puxadinho, zunido → **prova de generalização**
- **Controles** (4): biscoito, computador → sanidade

---

[modes: full, compact]

<!-- ref: article/ARTICLE.md §7.3 + §7.5 — OOV 5/5 -->
## Resultado que mais importa

### Palavras PT-BR reais fora do vocabulário: **5/5 ✓**

**Exemplos testados (todos corretos):**
- `puxadinho` → p u ʃ a ˈ d ʒ ĩ ɲ ʊ ✓
- `malcriado` → m a w k ɾ i ˈ a d ʊ ✓
- `arrombado` → a x õ ˈ b a d ʊ ✓
- `abacatada` → a b a k a ˈ t a d ə ✓
- `zunido` → z u ˈ n i d ʊ ✓

**O modelo aprendeu regras produtivas do PT-BR:**
→ Palatalização (x→ʃ, dʒ), coda (l→w, rr→x), nasalização (om→õ)
→ **Não memorizou, aprendeu regras!**

---

[modes: full]

## Resultados de Generalização — Visão Geral

**Resumo por categoria de teste:**

- **PT-BR reais OOV**: 5/5 (100%) | Score 100% → Aprendeu regras, não memorizou ✓
- **Controles**: 4/4 (100%) | Score 100% → Regra ɣ/x aprendida corretamente ✓
- **Generalização PT-BR**: 4/9 (44%) | Score 97% → Near-misses articulatórios
- **Consoantes duplas**: 1/5 (20%) | Score 81% → Geminadas estrangeiras: gap
- **Anglicismos**: 1/5 (20%) | Score 71% → Fonologia inglesa é OOV
- **Chars OOV** (k/w/y): 0/3 (0%) | Score 68% → Falha esperada e documentada

**Score fonológico** = métrica de 0–100% independente do G2P
→ "97%" = mesmo quando erra, o fonema está na família articulatória certa

---

[modes: full, compact]

<!-- ref: article/ARTICLE.md §2.3 — Distribuição complementar ɣ/x -->
## Descoberta Fonológica — Regra ɣ/x no PT-BR

### O que é Coda?

**Coda silábica** = consoante(s) que fecham a sílaba (após vogal)
```
com-pu-ta-dor
           └─ "r" em coda final = /x/ (r no fim da palavra)

bor-bo-le-ta
└─ "r" em coda interna = /ɣ/ (r não-final, antes de consoante vozeada)
```

### Distribuição Complementar Perfeita (0 exceções no corpus)

```
/r/ em coda FINAL de palavra          → x   (fricativa surda)
/r/ em coda INTERNA antes C vozeada   → ɣ   (fricativa sonora — assimilação)
```

**Exemplos**:
- computador → x (coda final) ✓
- churrasco → x (rr) ✓
- borboleta → ɣ (antes de b vozeada) ✓
- açucarzão → ɣ (antes de z vozeada) ✓

**Dados**: 19.730 × x (final) · 5.449 × ɣ (pré-vozeado) · **zero exceções**

→ O modelo aprendeu **assimilação regressiva de vozeamento** (fenômeno universal) por puro contato com dados — validação da capacidade de generalização

---

[modes: full]

## A Métrica Fonológica (Score)

**Problema**: métricas binárias (certo/errado) não mostram *o quanto* se errou

**Solução implementada** — score 0–100% independente do G2P:

```python
_group_dist("ɣ", "x") = 0.1   # mesma família FR_velar
_group_dist("a",  "k") = 0.9   # V_baixo vs OC_velar — opostos
```

| Score | Rótulo | Significado |
|-------|--------|-------------|
| 100% | exato | Transcrição idêntica |
| 90–99% | muito próximo | 1 sub. na mesma família |
| 70–89% | próximo | Subs. em famílias relacionadas |
| 50–69% | parcial | Diferenças estruturais |
| < 50% | distante | Falha substancial |

**Uso**: `G2PPredictor._phonological_score(pred, ref)` → independente de treino

---

[modes: full, compact]

## Como Usar — 3 comandos essenciais

**1. Testar uma palavra:**
```bash
python src/inference_light.py --index 18 --word computador
# → k õ p u t a ˈ d o x .
```

**2. Avaliar banco de generalização:**
```bash
python src/inference_light.py --index 18 --neologisms docs/data/generalization_test.tsv
```

**3. Modo interativo:**
```bash
python src/inference_light.py --index 18 --interactive
```

---

[modes: full]

## Escolha do Modelo na Prática

```
Você quer...
    ├─ Acertar mais palavras inteiras?
    │   └─ Use Exp9 (index=11) — WER 4.96%
    │       ex: indexação de texto, busca fonética, NLP
    │
    └─ Acertar mais fonemas individuais?
        └─ Use Exp104b (index=18) — PER 0.49%
            ex: TTS, alinhamento fonético, análise linguística
```

**Para palavras PT-BR "novas"**: ambos generalizam bem.
**Para anglicismos** e **geminadas**: espere erros — estão além do corpus.
**Para k/w/y**: mapeamento para UNK — avise o usuário.

---

[modes: full, compact]

## Comparação com Estado da Arte

| Sistema | PER | WER | Idioma | Teste (N) | Params |
|---------|-----|-----|--------|-----------|--------|
| **FG2P Exp104b** | **0,49%** | 5,43% | PT-BR | 28.782 | 9,7M |
| **FG2P Exp9** | 0,58% | **4,96%** | PT-BR | 28.782 | 9,7M |
| LatPhon 2025 | 0,86% | — | PT-BR | 500 | n/d |
| ByT5-Small | 8,9% | — | 100 idiomas | ~500/lang | 299M |

**FG2P usa 9,7M params** — ByT5-Small usa 299M (30× maior, zero-shot)

---

[modes: full, compact]

<!-- ref: article/ARTICLE.md §9 — Limitações -->
## Limites Bem-Definidos

| Limite | Causa | Solução Futura |
|--------|-------|----------------|
| Homógrafos heterófonos (jogo, gosto) | Ambiguidade léxico-sintática sem contexto | Pipeline NLP→G2P em série |
| Geminadas (zz, pp, tt) | Sem exemplos no corpus | Ampliar corpus com empréstimos |
| Fonologia inglesa (site, mouse) | OOV fonológico | Corpus bilíngue adaptado |
| k, w, y | OOV de charset | Re-treinar com charset ampliado |
| ɣ→x em coda | Sem restrição fonotática | Integrar fonotática PT-BR |
| `.`↔`ˈ` confusão | Posicional, não apenas distância | Métricas separadas + beam restrito |

---

[modes: full, compact]

## Próximos Passos

### Curto prazo
- **Stress Accuracy**: % de acentos na sílaba certa
- **Boundary F1**: precisão/revocação das fronteiras silábicas
- Corpus ampliado com empréstimos e geminadas

### Médio prazo
- **Espaço articulatório 7D contínuo**: substituir PanPhon binário
  - Símbolos estruturais (`.`, `ˈ`) distinguíveis intrinsecamente
  - Suporta interpolação entre sons

### Longo prazo
- **Universalização multilíngue**: o espaço 7D é universal
  - Transfer learning para novos idiomas com corpus reduzido
  - PT-BR → Espanhol → Inglês via fine-tuning

---

[modes: full, compact]

## Resumo em 5 Pontos

1. **BiLSTM + Atenção** — arquitetura clássica, bem compreendida, SOTA em PT-BR com 9,7M params

2. **Distance-Aware Loss** — sinal fonológico que ensina o modelo a "preferir erros inteligentes"; λ=0,20 ótimo empírico

3. **Separadores criam trade-off Pareto** — PER↓ + WER↑ de forma irredutível; escolha depende da aplicação

4. **Distâncias customizadas corrigem vetor zero** — bug no PanPhon para símbolos estruturais; fix pós-normalização → SOTA PER 0,49%

5. **O modelo generaliza regras PT-BR** — 100% em palavras reais OOV; falhas têm limites bem-definidos (charset, corpus, fonologia estrangeira)

---

[modes: full, compact]

# Obrigado

**FG2P** · PT-BR G2P BiLSTM

```
"biscoitinhozão" → b i s k o y t ʃ ĩ ɲ ɔ ˈ z ã ʊ̃
```

Código: `src/` · Dados: `dicts/` · Docs: `docs/article/ARTICLE.md`

```bash
# Experimente agora:
python src/inference_light.py --index 18 --interactive
```

---

[modes: full, compact]

# APÊNDICE A: Articulações Vocálicas

## Como os Sons Nascem na Boca

**Ponto** (Onde?)
| Zona | Exemplos | Sensação |
|------|----------|----------|
| Labial | /p/, /b/, /m/ | Lábios tocam |
| Alveolar | /t/, /d/, /s/, /z/, /n/ | Língua nos dentes |
| Palatal | /ʃ/, /ʒ/, /ɲ/, /j/ | Língua no céu da boca |
| Velar | /k/, /ɡ/ | Língua na garganta |

**Modo** (Como passa o ar?)
- **Oclusiva**: Bloqueia completamente (/p/, /b/, /t/, /d/, /k/, /ɡ/)
- **Fricativa**: Ar com atrito (/f/, /s/, /ʃ/, /x/, /z/, /ʒ/)
- **Nasal**: Ar pela nariz (/m/, /n/, /ɲ/)
- **Lateral**: Ar pelos lados da língua (/l/)

**Vozeamento** (Cordas vibram?)
- **Vozeadas**: /b/, /d/, /ɡ/, /v/, /z/, /ʒ/ — vibra
- **Desvozeadas**: /p/, /t/, /k/, /f/, /s/, /ʃ/ — não vibra

**Vogais** (Onde a língua fica)
- **Alto**: /i/, /u/ — língua perto do palato
- **Médio**: /e/, /o/, /ə/ — posição intermédia
- **Baixo**: /a/ — boca bem aberta
- **Frente**: /i/, /e/ — língua para frente
- **Trás**: /u/, /o/ — língua para trás

---

[modes: full, compact]

# APÊNDICE B: Termos de Algoritmos

## ML Básico

| Termo | O Quê |
|-------|-------|
| **Modelo** | Função que aprende padrões dos dados |
| **Treino** | Processo de ajustar parâmetros para minimizar erro |
| **Validação** | Dados para monitorar progresso (não treina) |
| **Teste** | Dados nunca vistos para avaliar resultado final |

## Losses (Funções de Erro)

- **Cross Entropy (CE)**: Erro base — `L_CE = -log(p)` — intervalo [0, +∞) sem discriminação fonológica
  - Matematicamente ilimitada superiormente, computacionalmente [0, ~16] com log-sum-exp
- **Distance-Aware (DA)**: Nosso sinal — penalidade proporcional à distância articulatória
- **Fórmula**: `L = L_CE + λ · d(ŷ, y) · p(ŷ)` onde λ=0.2 (empiricamente ótimo)

## Arquitetura

- **RNN**: Rede que processa sequências "lembrando" do passado
- **LSTM**: Versão melhorada de RNN com memória de longo prazo
- **Attention**: Mecanismo que permite focar em partes importantes
- **Embedding**: Converte símbolos (letras) em vetores numéricos

## Métricas

| Métrica | Foco |
|---------|------|
| **PER (Phoneme Error Rate)** | % de fonemas errados (importante para TTS) |
| **WER (Word Error Rate)** | % de palavras inteiras erradas (importante para busca) |
| **Accuracy** | % de acertos simples |

---

[modes: full, compact]

# APÊNDICE C: Termos do Projeto

## Conceitos Principais

- **G2P**: Converter grafemas (letras) em fonemas (sons) — objetivo principal
- **PT-BR**: Português brasileiro — variante modelada
- **SOTA**: State-of-the-Art — melhor resultado até agora
- **Trade-off**: Situação onde melhorar uma métrica piora outra

## Fenômenos Linguísticos

- **Coda**: Posição final de sílaba — consoantes sofrem mudanças aqui
- **Stress**: Acentuação — sílaba tônica vs. átona
- **Redução vocálica**: Vogal átona muda de som
- **Palatalização**: /t/, /d/ → /tʃ/, /dʒ/ antes de /i/

## Avaliação

- **Overfitting**: Memoriza dados de treino em vez de aprender regras
- **Generalização**: Capacidade de acertar em dados novos e desconhecidos
- **OOV (Out-of-Vocabulary)**: Símbolos/palavras nunca vistos antes
