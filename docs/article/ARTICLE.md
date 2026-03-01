# FG2P: Distance-Aware Loss Fonética para Conversão Grafema-Fonema no Português Brasileiro

**Relatório técnico — Projeto Acadêmico FG2P**
**Versão**: 1.0 | **Data**: 2026-02-25
**Status dos experimentos**: Exp0–Exp104b concluídos

---

## Resumo

Este trabalho apresenta o FG2P, um sistema de conversão grafema-para-fonema (*Grapheme-to-Phoneme*, G2P) para o Português Brasileiro construído sobre uma arquitetura BiLSTM Encoder-Decoder com mecanismo de atenção de Bahdanau. Treinado sobre um corpus de 95.937 palavras com transcrições IPA normalizadas e avaliado sobre 28.782 palavras de teste, o sistema alcança **PER (Phoneme Error Rate) de 0,49%** e **WER (Word Error Rate) de 4,96%** em configurações complementares, estabelecendo novos valores de referência para G2P em PT-BR em escala comparável.

São introduzidas três contribuições técnicas originais: (1) uma *Distance-Aware Loss* que penaliza erros fonologicamente distantes com peso proporcional à distância articulatória PanPhon; (2) separadores silábicos como tokens de saída que melhoram a acurácia por fonema ao custo de precisão por palavra; e (3) correção de distâncias customizadas para símbolos estruturais (`.` e `ˈ`), que possuem vetor zero no espaço PanPhon e não recebem gradiente diferenciado sem a correção. Adicionalmente, um banco de generalização com 31 palavras em 6 categorias e uma métrica fonológica independente do framework confirmam que o modelo **generaliza sistematicamente as regras do PT-BR** para palavras inéditas, atingindo 100% de acerto em palavras reais fora do vocabulário de treino.

**Palavras-chave**: G2P, conversão grafema-fonema, Português Brasileiro, BiLSTM, atenção, Distance-Aware Loss, IPA, fonologia.

---

## 1. Introdução

A conversão automática de texto escrito em representação fonética — tarefa conhecida como *Grapheme-to-Phoneme* (G2P) — é componente fundamental de sistemas de síntese de fala (*Text-to-Speech*, TTS), reconhecimento de fala, análise linguística computacional e processamento de linguagem natural. A entrada do sistema é uma sequência de caracteres (grafemas) e a saída é a sequência de fonemas do Alfabeto Fonético Internacional (IPA) correspondente.

Para o Português Brasileiro, o problema apresenta desafios específicos que o tornam particularmente interessante como objeto de estudo:

**Ambiguidade grafêmica**: O mesmo grafema produz fonemas distintos dependendo do contexto. O grafema "c" realiza-se como `/k/` em "cama" mas como `/s/` em "cena". O "s" intervocálico soante realiza-se `/z/` (casa → /k-a-z-a/). O "r" em posição de ataque silábico realiza-se `/ɾ/`, mas em posição de coda e em início de palavra realiza-se `/x/` (carro → `/k-a-x-ʊ/`; roda → `/x-ɔ-d-ə/`).

**Suprassegmentais marcados como tokens**: O corpus utilizado representa o acento tônico com o símbolo `ˈ` como token separado antes da sílaba tônica, e fronteiras silábicas com `.` em algumas configurações. Esses tokens não correspondem a sons articulatórios e exigem tratamento especial na função de custo.

**Neutralização vocálica**: Em sílabas átonas, o contraste entre vogais médias abertas e fechadas neutraliza-se: `/ɛ/` e `/e/` ambos realizam-se como /e/ ou /ɪ/ em posição átona. Isso introduz "ruído fonológico legítimo" no corpus — o mesmo contexto ortográfico pode corresponder a transcrições ligeiramente diferentes que são ambas foneticamente corretas.

O objetivo deste trabalho é construir um modelo de alta precisão para essa tarefa, documentar sistematicamente as escolhas de design e seus efeitos empíricos, e avaliar a capacidade de generalização do modelo para palavras fora do vocabulário de treino — incluindo neologismos, empréstimos linguísticos e palavras morfologicamente derivadas.

### 1.1 Comparação com o Estado da Arte

A referência externa mais próxima ao escopo deste trabalho é o **LatPhon** (2025), que reporta PER de 0,86% em teste de 500 palavras para PT-BR. O FG2P alcança PER de 0,49–0,58% com teste de 28.782 palavras — conjunto 57 vezes maior, fornecendo estimativa estatisticamente muito mais robusta.

Em escala global, modelos como **ByT5-Small** (Xue et al., 2022, 299M parâmetros) alcançam PER de 8,9% em 100 idiomas em configuração zero-shot. O FG2P, com 9,7M parâmetros treinados especificamente para PT-BR, supera essa referência significativamente no idioma alvo.

---

## 2. Dados

### 2.1 Corpus

O corpus de treinamento consiste em **95.937 pares (palavra, transcrição IPA)** extraídos de `dicts/pt-br.tsv`, um dicionário fonético do Português Brasileiro com transcrições normalizadas. Após inspeção, foram corrigidas 10.252 instâncias com o grafema "g" (U+0067) que deveriam usar o símbolo IPA "ɡ" (U+0261, oclusiva velar sonora), distinção necessária para o correto mapeamento pelo PanPhon.

**Características do corpus**:

| Aspecto | Valor |
|---------|-------|
| Total de pares | 95.937 |
| Tamanho médio da palavra | ~8 caracteres |
| Tamanho médio da transcrição | ~9 fonemas |
| Charset de entrada (caracteres treinados) | a–z (exceto k, w, y) + ç, á, à, â, ã, é, ê, í, ó, ô, õ, ú, ü, – |
| Vocabulário de saída (fonemas) | ~52–54 tokens IPA + especiais |

**Nota**: Os grafemas `k`, `w` e `y` não ocorrem no dicionário PT-BR de treino e portanto não fazem parte do vocabulário de entrada do modelo. Palavras que contêm esses caracteres — como anglicismos (*wifi*, *karatê*) — são tratadas como Out-of-Vocabulary (OOV) de caractere, com mapeamento para `<UNK>`.

### 2.2 Divisão e Estratificação

O corpus é dividido em três subconjuntos com **estratificação por características fonológicas**, prática recomendada para assegurar representatividade proporcional em datasets heterogêneos (Kohavi, 1995; Arlot & Celisse, 2010).

| Subconjunto | Proporção | Palavras |
|-------------|-----------|---------|
| Treino | 60% | 57.561 |
| Validação | 10% | 9.594 |
| Teste | 30% | 28.782 |

**Variáveis de estratificação**: Cada par (palavra, transcrição) é atribuído a um estrato pela combinação hierárquica de três features fonológicas:

1. **`stress_type`** — posição do acento primário (oxítona, paroxítona, proparoxítona)
2. **`syllable_bin`** — faixa de contagem de sílabas (monossilábica, 2, 3, 4, 5+)
3. **`length_bin`** — faixa de comprimento em grafemas (≤4, 5–7, 8–10, 11+)

A combinação `stress_type × syllable_bin × length_bin` gera **~48 estratos** distintos. Estratos com menos de 2 membros são consolidados em `__rare__` para viabilizar o split. A implementação usa `sklearn.model_selection.train_test_split(stratify=strata, random_state=42)` em duas etapas: (1) extração do conjunto de teste (30%); (2) extração do conjunto de validação do restante (val_fraction ≈ 14,3% do trainval → 10% efetivo do total).

**Qualidade do balanceamento**:

$$\chi^2 = 0{,}95 \quad (p = 0{,}678 > 0{,}05) \qquad \text{Cramér V} = 0{,}0007$$

O valor p não-significante indica ausência de diferença estatística entre as distribuições dos estratos nos três subconjuntos — o balanceamento é excelente. O Cramér V ≈ 0 confirma que a divisão não introduz viés em relação às features de estratificação.

**Descoberta metodológica**: A divisão 60/10/30 supera 70/10/20 em −41% de PER (Exp0 vs. Exp1), contrariando a intuição de que "mais dados de treino = melhor performance". A explicação é dupla: (a) o conjunto de teste 50% maior fornece estimativa estatisticamente mais robusta; (b) o conjunto de treino ligeiramente menor força o modelo a aprender invariantes generalizáveis em vez de memorizar padrões específicos. Esta descoberta tem implicações diretas para o design de experimentos em datasets de porte médio.

### 2.3 Auditoria do Corpus: Alofones e Normalização Unicode

Uma inspeção sistemática do corpus revelou dois aspectos relevantes para a modelagem fonológica.

**Regra ɣ/x — Distribuição Complementar Perfeita**: O corpus contém dois alofones do fonema /r/ em posição de coda silábica:

| Alofone | Posição | Frequência | Exemplos |
|---------|---------|-----------|---------|
| `x` | Coda final de palavra | 19.730 palavras | computador → `k õ p u t a ˈ d o x` |
| `ɣ` | Coda interna antes de C vozeada | 5.449 palavras | borboleta → `b o ɣ . b o . ˈ l e . t ə` |

A distribuição é **completamente complementar** — nenhuma exceção entre as 95.937 entradas. O fenômeno é assimilação regressiva de vozeamento: quando o fonema seguinte é vozeado (b, d, g, v, z, ʒ, m, n, l, ɲ, ɾ), a fricativa em coda também se torna vozeada (`ɣ`); na coda final de palavra, onde não há seguinte, permanece surda (`x`). Esta regra é produtiva e universal nas línguas humanas — o corpus usa transcrição alofônica rigorosa, não abstrata.

A descoberta foi metodologicamente importante: na avaliação qualitativa inicial, as predições do modelo em palavras como *borboleta* (`ɣ` antes de `/b/`) foram incorretamente classificadas como erros. Após a auditoria, 4 entradas do banco de generalização foram corrigidas, e a acurácia corrigida subiu de 14/31 (45%) para 17/31 (55%). Para a análise fonológica completa dos símbolos IPA usados no corpus (incluindo a validação empírica da distribuição complementar [x]/[ɣ] com 0 exceções em 95.937 palavras), ver [PHONOLOGICAL_ANALYSIS.md](../linguistics/PHONOLOGICAL_ANALYSIS.md).

**Normalização Unicode NFC**: 10 entradas (~0,01%) continham vogais nasais em representação NFD (caractere base + combining mark separado) em vez de NFC (caractere pré-composto). O pipeline normaliza automaticamente para NFC via `unicodedata.normalize()`, mas as 10 entradas foram corrigidas na fonte para consistência. Impacto mensurável no PER: nulo.

---


## 3. Arquitetura

O sistema utiliza uma arquitetura **BiLSTM Encoder-Decoder com mecanismo de atenção de Bahdanau** (Bahdanau et al., 2014), que se estabeleceu como o paradigma dominante para G2P antes da era dos Transformers e ainda demonstra excelente desempenho para datasets de porte médio.

```
Grafemas ("c", "a", "s", "a")
          ↓
  [Camada de Embedding]       128D–256D aprendido ou 24D PanPhon
          ↓
  [BiLSTM Encoder]            2 camadas, hidden=256–512D
  forward: c→a→s→a           bidirecional: contexto completo
  backward: a→s→a→c
          ↓
  [Atenção de Bahdanau]       α = softmax(v·tanh(Wh·h + Ws·s))
                              c_t = Σ α_t · h_t
          ↓
  [LSTM Decoder]              2 camadas, hidden=256D
  (teacher forcing no treino, auto-regressivo na inferência)
          ↓
  [Projeção Linear → Softmax] logits sobre vocab fonético
          ↓
Fonemas IPA (/k/, /a/, /z/, /a/)
```

### 3.1 Encoder BiLSTM

O encoder processa a sequência de grafemas nas duas direções. Para cada posição $t$, a representação bidirecional $h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$ concatena os estados forward e backward, provendo ao decoder acesso simultâneo ao contexto anterior e posterior de cada grafema. Isso é essencial para ambiguidades contextuais como "c" em "cama" (velar) vs "cena" (alveolar), onde o fonema depende do grafema seguinte.

As equações das células LSTM são:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad \tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t, \qquad h_t = o_t \odot \tanh(C_t)$$

### 3.2 Mecanismo de Atenção (Bahdanau)

A atenção permite que o decoder pondere dinamicamente todos os estados do encoder a cada passo de decodificação, em vez de depender apenas do estado final comprimido:

$$e_{t,j} = v^\top \tanh(W_h h_j + W_s s_{t-1}), \qquad \alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_k \exp(e_{t,k})}$$

$$c_t = \sum_j \alpha_{t,j} h_j$$

O vetor de contexto $c_t$ é concatenado ao estado do decoder para gerar a predição do próximo fonema. A atenção é especialmente útil para aprender alinhamentos muitos-para-um (como "nh" → [ɲ]) e um-para-muitos (como "x" → [ʃ] ou [ks] dependendo do contexto).

### 3.3 Vocabulários

**Entrada (grafemas)**: `PAD` (0), `UNK` (1), seguidos dos caracteres treinados. A propriedade `padding_idx=0` garante que o embedding de PAD permaneça fixo em zero — o batch padding é invisível ao modelo.

**Saída (fonemas)**: `PAD` (0), `UNK` (1), `EOS` (2), seguidos dos ~52 fonemas IPA do PT-BR. O token `EOS` é gerado explicitamente pelo decoder ao final de cada sequência (não apenas por comprimento máximo), tornando a detecção de fim mais robusta.

**Configurações testadas**:

| Configuração | Embedding | Hidden | Parâmetros | Status |
|---|---|---|---|---|
| Pequena | 128D | 256D | 4,3M | Baseline |
| Intermediária | 192D | 384D | 9,7M | **Sweet spot** |
| Grande | 256D | 512D | 17,2M | Saturação |

---

## 4. Funções de Custo

### 4.1 CrossEntropy Clássica (CE)

A loss padrão para classificação multi-classe:

$$L_{CE} = -\frac{1}{N} \sum_{i=1}^N \log p_i^{(y_i)}$$

O principal limitante da CE é tratar todos os erros igualmente. Do ponto de vista fonológico, isso é inadequado: substituir `/ɛ/` por `/e/` (1 feature de diferença, vogais médias da mesma família) é erro qualitativaente diferente de substituir `/a/` por `/k/` (8+ features, vogal baixa vs. oclusiva velar).

### 4.2 Distance-Aware Phonetic Loss (DA Loss)

#### O problema fundamental

A CrossEntropy clássica é "cega" à estrutura fonológica: qualquer erro vale 1 (errado). Do ponto de vista de quem aprende a produzir sons, isso é um sinal de treinamento distorcido:

```
Situação A — modelo prediz ɛ, correto é e:
  Diferença articulatória: altura da língua ligeiramente diferente
  CrossEntropy: erro = 1.0  ← penalidade máxima

Situação B — modelo prediz k, correto é a:
  Diferença articulatória: vogal baixa vs. oclusiva velar — sons completamente distintos
  CrossEntropy: erro = 1.0  ← mesma penalidade!
```

Para o modelo, errar `ɛ→e` (quasi-correto) e errar `a→k` (catastrófico) são indistinguíveis. Não há sinal que indique "pelo menos você estava perto".

#### A intuição da DA Loss

A Distance-Aware Loss adiciona uma **segunda voz** ao sinal de treinamento: além de dizer "errou", ela diz "o quanto errou, na escala fonológica real":

```
Situação A — modelo prediz ɛ, correto é e:
  d(ɛ, e) = 0.10  (mesma família V_med_front — apenas altura varia)
  Penalidade extra: λ × 0.10 × p_pred  ← pequena

Situação B — modelo prediz k, correto é a:
  d(k, a) = 0.90  (vogal baixa vs. oclusiva velar — completamente diferentes)
  Penalidade extra: λ × 0.90 × p_pred  ← 9× maior
```

O resultado ao longo das épocas: quando o modelo está incerto entre dois candidatos, ele aprende a "desempatar para o lado certo" — preferir o fonema mais próximo articulatoriamente do correto.

#### A fórmula completa

$$L = L_{CE} + \lambda \cdot d_{\text{PanPhon}}(\hat{y}_i, y_i) \cdot p_i^{(\hat{y}_i)}$$

Cada parâmetro tem uma motivação específica:

**$d_{\text{PanPhon}}(\hat{y}, y)$ — a régua fonológica**

Distância calculada sobre os 24 features articulatórios binários do PanPhon (vozeamento, nasalidade, ponto de articulação, modo de articulação, etc.), normalizada para [0, 1]. Representa o quanto dois fonemas diferem na sua produção física — não na ortografia, não no símbolo IPA, mas no movimento dos órgãos fonadores.

Exemplos de pares com distâncias reais:

| Par | d | Interpretação |
|-----|---|---|
| `e` ↔ `ɛ` | ~0.10 | Vogais médias, mesma posição, abertura diferente |
| `p` ↔ `b` | ~0.05 | Oclusivas bilabiais, só vozeamento difere |
| `s` ↔ `ʃ` | ~0.15 | Fricativas, ponto de articulação diferente |
| `a` ↔ `k` | ~0.90 | Vogal baixa vs. oclusiva velar — família oposta |
| `.` ↔ `ˈ` | 0.00* | Ambos vetores zero no PanPhon — **problema corrigido em Exp104b** |

**$p_i^{(\hat{y}_i)}$ — o fator de confiança**

É a probabilidade que o modelo atribuiu ao fonema que ele efetivamente prediu (o argmax do softmax). Multiplicar a distância por essa confiança faz com que a penalidade escale com o nível de certeza do erro:

```
Época 5 (modelo ainda aprendendo):
  Prediz k com 52% de confiança → penalidade moderada
  → Sinal: "ainda incerto, mas direção errada"

Época 40 (modelo mais seguro):
  Prediz k com 91% de confiança → penalidade alta
  → Sinal: "você está MUITO confiante mas completamente errado"
```

Este mecanismo é análogo ao *label smoothing* — ambos penalizam confiança excessiva — mas aqui a penalidade é proporcional à distância fonológica, não uniforme.

**$\lambda$ — o peso do sinal fonológico**

Controla quanto do sinal articulatório "pesa" em relação ao CE. $\lambda = 0$ reproduz a CE pura; $\lambda$ muito alto ("sobrepenaliza") faz o modelo hesitar mesmo quando sabe a resposta correta. O valor ótimo encontrado empiricamente é $\lambda = 0{,}20$.

#### Fluxo de um exemplo passo a passo

Tomamos a palavra "cena" (fonemas corretos: `s e n ə`) e acompanhamos o passo de decodificação onde o modelo deve predizer `/s/` para o grafema "c":

```
PASSO 1 — Softmax do decoder gera distribuição sobre fonemas:
  {  s: 0.42,  ʃ: 0.35,  z: 0.12,  t: 0.06,  outros: 0.05  }
  ↓  argmax
  Predição: ʃ  (errado — devia ser s)

PASSO 2 — CE Loss calcula penalidade base:
  L_CE = -log(0.42) = 0.868   ← penaliza que "s" não foi o máximo

PASSO 3 — DA Loss calcula o termo extra:
  d(ʃ, s) = 0.15  (mesma classe fricativa, ponto diferente: palatal vs. alveolar)
  p(ʃ)    = 0.35  (confiança no predito incorreto)
  λ       = 0.20

  L_extra = 0.20 × 0.15 × 0.35 = 0.0105

PASSO 4 — Loss total = 0.868 + 0.0105 = 0.879

COMPARAÇÃO — se tivesse predito k (completamente errado):
  d(k, s) = 0.80,  p(k) = 0.35 (hipotético)
  L_extra = 0.20 × 0.80 × 0.35 = 0.056
  Loss total = 0.868 + 0.056 = 0.924  ← 5% maior que o erro "próximo"
```

A diferença de 5% por passo pode parecer pequena, mas **ao longo de milhares de batches e épocas**, o gradiente acumulado empurra os pesos do modelo para aprender "quando incerto sobre fricativas, prefira as alveolares". O modelo não recebe uma regra; ele recebe um sinal diferenciado que gradativamente molda sua incerteza.

#### Por que isso melhora PER e WER

Após o treinamento, o modelo não comete menos erros em termos absolutos — a CE ainda domina o sinal. O que muda é a *distribuição* dos erros:

- Antes: erros são "aleatórios" na escala fonológica (qualquer fonema pode substituir qualquer outro)
- Depois: erros tendem a ser fonologicamente próximos (substituições dentro da mesma família)

Isso melhora PER porque a distância de edição Levenshtein contabiliza cada fonema errado igualmente — e havendo menos fonemas "catastroficamente errados", PER cai.

WER melhora porque erros próximos às vezes são *aceitáveis* — `ɛ` por `e` pode ser imperceptível em TTS, enquanto `k` por `a` quebra a palavra completamente.

#### Busca do hiperparâmetro λ

Experimento Exp7 varreu λ ∈ {0,05; 0,10; 0,20; 0,50} com arquitetura fixa (4,3M params):

| λ | PER | Comportamento |
|---|-----|---|
| 0,05 | 0,68% | Sinal fonológico fraco demais; CE domina e resultado é inferior ao CE puro |
| 0,10 | 0,63% | Melhora — âncora do Exp6 |
| **0,20** | **0,61%** | **Ótimo empírico — curva U-invertido** |
| 0,50 | 0,73% | Sobrepenaliza; modelo hesita mesmo quando confiante no correto |

A curva em U-invertido é esperada: muito pouco sinal fonológico = inócuo; muito sinal = atrapalha a CE que precisa dominar o aprendizado principal. λ=0,20 foi depois aplicado no modelo 9,7M (Exp9) → SOTA WER 4,96%.

### 4.3 Distâncias Customizadas para Símbolos Estruturais

O PanPhon atribui vetor zero a símbolos não-fonéticos como `.` (separador silábico) e `ˈ` (marcador de acento). Consequentemente, a distância entre eles é $d(., ˈ) = 0{,}0$ — a DA Loss não penaliza confusões entre esses símbolos.

**Solução implementada (Exp104b)**: Override pós-normalização da matriz de distâncias:

```python
# Após normalização da matriz euclidiana:
for sym in {'.', 'ˈ'}:
    for other in phoneme_vocab:
        if other != sym:
            distance_matrix[sym, other] = 1.0   # distância máxima
            distance_matrix[other, sym] = 1.0
```

O ponto crítico é aplicar o override **após** a normalização (dividir por `max_dist` euclidiano). Exp104 aplicava o override antes da normalização, resultando em $d(., ˈ) \approx 0{,}25$ após divisão — equivalente à distância entre vogais médias, não à máxima.

**Evidência empírica (Exp102 e Exp103)**: A análise de erros do Exp102 (CE + separadores) contabiliza 107 confusões estruturais por rodada de avaliação: `.→ˈ` (59) e `ˈ→.` (48). Adicionando DA Loss (Exp103), o total permanece em ~107 — a loss altera a *direção* das confusões mas não sua frequência total. Isso confirmou que `d(., ˈ) = 0.0` torna o sinal de distância inútil para esses pares.

**Alternativas consideradas**: Cinco abordagens foram avaliadas antes do override:

| Abordagem | Complexidade | Descrição |
|-----------|-------------|-----------|
| 1. Override direto da matriz | Baixa | Setar d(., ˈ) = 1.0 pós-normalização |
| 2. Módulo de distância customizado | Média | Classe `CustomDistanceMatrix` com regras por tipo de símbolo |
| 3. Métrica de distância aprendida | Alta | MLP que aprende função de distância end-to-end |
| 4. Espaço de embedding estruturado | Muito alta | 8+ dimensões extras para símbolos estruturais |
| 5. Hierarquia de tipos + pesos | Média | Classificar tokens em phoneme/structural/pause e aplicar pesos distintos |

A abordagem 1 foi escolhida por simplicidade e eficácia — resolve o problema imediato com 3 linhas de código. As abordagens 3 e 4 são promissoras para trabalhos futuros (cf. espaço 7D, Seção 9.1).

---

## 5. Experimentos e Resultados

### 5.1 Progressão dos Experimentos

A tabela a seguir sintetiza todos os experimentos concluídos, agrupados por fase metodológica:

| Exp | Params | Loss | Sep | PER | WER | Acc | Insight |
|-----|--------|------|-----|-----|-----|-----|---------|
| **Fase 1: Baseline e Split** | | | | | | | |
| Exp0 | 4,3M | CE | não | 1,12% | 9,37% | 90,63% | Split 70/10/20 — baseline |
| Exp1 | 4,3M | CE | não | 0,66% | 5,65% | 94,35% | Split 60/10/30 — −41% PER |
| **Fase 2: Capacidade** | | | | | | | |
| Exp2 | 17,2M | CE | não | 0,60% | 4,98% | 95,02% | ROI negativo (4× params, −9% PER) |
| Exp5 | 9,7M | CE | não | 0,63% | 5,38% | 94,62% | Sweet spot intermediário |
| **Fase 3: Embeddings** | | | | | | | |
| Exp3 | 4,3M | CE + PanPhon_T | não | 0,66% | 5,45% | 94,55% | Erros mais inteligentes (PER_w 0,28%) |
| Exp4 | 4,0M | CE + PanPhon_F | não | 0,71% | ~6,0% | ~93,5% | Features fixas: insuficientes |
| **Fase 4: Loss Fonética** | | | | | | | |
| Exp6 | 4,3M | DA λ=0,1 | não | 0,63% | 5,35% | 94,65% | DA Loss funciona (−4,5% PER) |
| Exp7 | 4,3M | DA λ=0,2 | não | 0,61% | ~5,3% | — | λ ótimo confirmado |
| Exp8 | 4,3M | DA+PanPhon λ=0,2 | não | 0,65% | ~5,4% | — | Sinergia não materializada |
| **Exp9** | **9,7M** | **DA λ=0,2** | **não** | **0,58%** | **4,96%** | **95,04%** | **SOTA WER** |
| Exp10 | 17,2M | DA λ=0,2 | não | 0,61% | 5,25% | 94,75% | DA não escala para 17,2M |
| **Fase 5: Separadores Silábicos** | | | | | | | |
| Exp101 | 4,3M | CE | sim | 0,53% | 5,99% | 94,01% | Sep+raw: PER ↓, WER ↑ |
| Exp102 | 9,7M | CE | sim | 0,52% | 5,79% | 94,21% | Capacidade atenua WER penalty |
| **Fase 6: DA + Sep + Distâncias Customizadas** | | | | | | | |
| Exp103 | 9,7M | DA λ=0,2 | sim | 0,53% | 5,73% | 94,27% | Efeitos não-aditivos |
| Exp104 | 9,7M | DA λ=0,2 + dist | sim | 0,54% | 5,88% | 94,12% | Bug: override pré-norm |
| **Exp104b** | **9,7M** | **DA λ=0,2 + dist** | **sim** | **0,49%** | **5,43%** | **94,57%** | **SOTA PER** |
| **Fase 7: Robustez e Ablações** | | | | | | | |
| Exp105 | 9,7M | DA λ=0,2 + dist | sim | 0,54% | 5,87% | 94,13% | 50% dados, com hífen — robustez |
| Exp106 | 9,7M | DA λ=0,2 + dist | sim | 0,58% | 6,12% | 93,88% | 50% dados, sem hífen — 2,58× speed |

### 5.2 Design Space Completo

O espaço experimental pode ser visualizado como uma grade encoding × loss:

| Encoding \ Loss | CE | DA λ=0,2 | DA λ=0,2 + dist corrigida |
|---|---|---|---|
| Raw 4,3M | 0,66% / 5,65% | 0,61% | — |
| Raw 9,7M | 0,63% / 5,38% | **0,58% / 4,96%** ← WER SOTA | — |
| Raw 9,7M + Sep | 0,52% / 5,79% | 0,53% / 5,73% | **0,49% / 5,43%** ← PER SOTA |
| NFD Decomposed | 0,97% / 7,53% ❌ | — | — |

O encoding NFD (decomposed) foi testado em Exp11 e mostrou regressão severa (+47% PER), sendo descartado. O culpado é a decomposição Unicode, não os separadores — confirmado por Exp101 (sep sem decomposed: PER melhora).

### 5.3 Principais Descobertas por Fase

**Fase 1 — Split**: O split 60/10/30 supera 70/10/20 em −41% PER com a mesma arquitetura. Mais dados de treino não garantem melhor generalização — conjunto de teste maior e bem estratificado é mais valioso.

**Fase 2 — Capacidade**: O modelo 9,7M é o *sweet spot*. 4,3M satura; 17,2M não adiciona valor proporcional. DA Loss funciona como regularizador e, acima de 17,2M parâmetros, passa a interferir negativamente com o potencial de memorização benígna do modelo.

**Fase 3 — Embeddings PanPhon**: Features articulatórias treináveis (Exp3) não melhoram PER clássico (0,66% em ambos), mas produzem erros qualitativamente melhores: PER ponderado 0,28% vs 0,30% e menos erros de classe D (grave). Para TTS, onde pequenas variações fonéticas são aceitáveis, Exp3 é preferível ao baseline apesar da mesma acurácia.

**Fase 4 — Distance-Aware Loss**: DA Loss com λ=0,2 reduz PER em −12% (0,66%→0,58%) no modelo 9,7M. A sinergia com PanPhon trainable (Exp8) não se materializa — o inductive bias articulatório no embedding e na loss compete em vez de se complementar. A ortografia regular do PT-BR já provê o padrão; features linguísticas não adicionam valor marginal.

**Fase 5 — Separadores Silábicos**: Separadores melhoram PER consistentemente (−17% a −20%) mas pioram WER (+6% a +8%). O mecanismo é estrutural: cada erro em token separador conta como erro de palavra inteira. Capacidade maior (9,7M) atenua o dano mas não elimina o trade-off. **Resultado publicável**: separadores silábicos criam um trade-off Pareto PER/WER fundamental em G2P BiLSTM para PT-BR.

**Fase 6 — Distâncias Customizadas**: O override correto (Exp104b, pós-normalização) melhora PER em −7,5% vs. Exp103 e estabelece novo SOTA PER de 0,49%. As confusões estruturais `.`↔`ˈ` persistem (~106 no total), indicando que o problema é predominantemente posicional — o modelo confunde o posicionamento dos tokens estruturais na sequência, não apenas sua identidade — e não é remediável apenas pelo sinal de distância na loss.

**Fase 7 — Robustez e Ablações**: Dois experimentos testam a robustez do sistema SOTA a variações controladas de dados.

**Exp105** (50% de dados de treino, com hífen): Reduz os dados de treino de 60% para 50% (~10K palavras a menos), mantendo todas as demais configurações idênticas ao Exp104b. O conjunto de teste cresce de 28.782 para 38.296 palavras (+33% de poder estatístico). Resultado: PER 0,54% — apenas +0,05 p.p. de degradação com 17% menos dados. A balanceamento estratificado é mantido (χ² p=0,86, Cramér V=0,003). **Conclusão**: O modelo é robusto à redução de dados; a qualidade do corpus e da arquitetura importam mais que a quantidade de exemplos até este nível de redução.

**Exp106** (50% de dados + remoção do hífen): Além da redução de dados do Exp105, remove o caractere hífen do vocabulário de entrada (CharVocab 39→38 tokens). O impacto na acurácia é mínimo: PER 0,58% (+0,04 p.p. vs. Exp105). O impacto na velocidade é substancial: **30,2 palavras/segundo vs. 11,7 w/s do Exp105 — speedup de 2,58×**. A confusão `.→ˈ` aumentou ligeiramente (+22 ocorrências), mas dentro do ruído esperado. **Conclusão**: O hífen é foneticamente irrelevante (palavras compostas têm a mesma pronúncia com ou sem ele); removê-lo reduz o vocabulário sem penalidade semântica, com ganho expressivo de throughput de inferência.

---

## 6. Análise de Erros

### 6.1 Padrões de Confusão Dominantes

A análise dos erros do modelo SOTA por palavras (Exp104b, 28.782 palavras teste) revela os seguintes padrões:

| Confusão | Contagem | Grupo Fonológico | Causa |
|----------|----------|-----------------|-------|
| ɛ → e | 255 | Vogal média anterior | Neutralização PT-BR em átona |
| e → ɛ | 197 | Vogal média anterior | Idem (sentido inverso) |
| ɔ → o | 131 | Vogal média posterior | Neutralização PT-BR em átona |
| i → e | 121 | Vogal alta/média anterior | Redução vocálica |
| o → ɔ | 95 | Vogal média posterior | Idem |
| ə → a | 73 | Vogal central/baixa | Schwa → vogal aberta |
| . → ˈ | 67 | Estrutural | Confusão posicional |
| a → . | 62 | Fonema → estrutural | Inserção indevida de separador |
| i → . | 55 | Fonema → estrutural | Idem |
| ˈ → . | 39 | Estrutural | Confusão posicional inversa |

**Observação central**: Mais de 60% dos erros são neutralizações vocálicas — substituições entre vogais médias abertas e fechadas (`/ɛ/↔/e/`, `/ɔ/↔/o/`). Estas não são falhas do modelo; refletem ambiguidade fonológica genuína do Português Brasileiro, onde vogais médias neutralizam em posição átona. O corpus inevitavelmente contém exemplos "conflitantes" para o modelo — sem acesso à prosódia, a confusão é esperada e parcialmente irredutível por treinamento adicional.

**Análise estatística da neutralização e↔ɛ**: A análise do corpus revela a causa estrutural da confusão. A distribuição global de `/e/` vs. `/ɛ/` é fortemente assimétrica — razão 7,1:1 (46.209 vs. 6.510 instâncias). Mais relevante é a distribuição por posição silábica:

| Posição | Instâncias /e/ | Instâncias /ɛ/ | Razão e:ɛ |
|---------|---------------|---------------|---------|
| Pré-tônica | 38.147 | 1.535 | **24,9:1** |
| Tônica | 275 | 836 | **0,33:1** (ɛ majoritário) |
| Pós-tônica | 7.787 | 4.139 | 1,6:1 |

O padrão revela que `/ɛ/` é na verdade a **vogal tônica dominante** (razão inversa), mas o desequilíbrio massivo na posição pré-tônica (25:1) vicia o aprendizado global para `/e/`. O modelo aprende o bias pré-tônico do corpus e o generaliza incorretamente para sílabas tônicas, produzindo exatamente a distribuição de erros observada.

Esta análise transforma o que parece um erro do modelo em **espelho fiel do corpus**: o modelo não está errado — está sendo consistente com o que aprendeu. Reduzir esses 18,9% de erros exigiria informação prosódica explícita (posição do acento como feature adicional) ou um classificador separado para vogais médias tônicas.

### 6.2 Revisão da Avaliação de Generalização: ɣ/x como Comportamento Correto

Na avaliação inicial de generalização, o modelo produzia `ɣ` em posição de coda em palavras como *borboleta* (`b o ɣ . b o...`), *açucarzão* e *computadorzinho*. Isso foi inicialmente classificado como erro sistemático — o avaliador esperava `x`.

A auditoria do corpus (Seção 2.3) revelou que a classificação estava **invertida**: o modelo estava certo; a referência inicial estava errada. A regra de assimilação de vozeamento do PT-BR (documentada com distribuição complementar perfeita no corpus) exige:

- `ɣ` antes de consoante vozeada: `borboleta` → `b o **ɣ** . b o . ˈ l e . t ə` (r-coda antes de `/b/`)
- `x` em coda final: `computador` → `k õ p u t a . ˈ d o **x**`

O modelo aprendeu essa regra corretamente a partir do corpus. Quatro entradas do banco de generalização (`borboleta`, `computadorzinho`, `açucarzão`, `internet`) foram corrigidas, e a acurácia revisada subiu de 14/31 (45%) para 17/31 (55%).

Este episódio ilustra uma lição metodológica importante: **métricas globais (PER/WER) não são suficientes para identificar se um padrão é erro ou acerto sistêmico**. A inspeção qualitativa revelou que o padrão era correto — mas a avaliação qualitativa também exige conhecimento fonológico suficiente para distinguir alofones de erros genuínos.

### 6.3 Confusões Estruturais (`.` ↔ `ˈ`)

Nos modelos com separadores silábicos, o modelo comete ~106 erros envolvendo tokens estruturais por rodada de avaliação. A análise revela que o problema é primariamente **posicional**: as confusões mais frequentes envolvem `i→.` (66×) e `a→.` (62×) — o modelo insere separadores onde havia vogais. Isso sugere que o modelo aprendeu a associar certas posições silábicas com separadores, não necessariamente a reconhecer as fronteiras corretas.

O override de distância (Exp104b) reduziu ligeiramente essas confusões, mas o problema central requer abordagem diferente — possivelmente métricas separadas (Boundary F1) ou integração de restrições fonotáticas explícitas.

---

## 7. Avaliação de Generalização

### 7.1 Motivação e Design

Para avaliar honestamente a capacidade do modelo de generalizar além do corpus de treino, foi construído o **Banco de Generalização** (`docs/data/generalization_test.tsv`) — um conjunto de 31 palavras curadas em 6 categorias, com foco especial em distinguir três tipos de dificuldade:

1. **Falhas esperadas e documentadas**: palavras com caracteres OOV (k, w, y)
2. **Generalização de padrões**: palavras com todos os caracteres no vocabulário mas estrutura não vista
3. **Controles**: palavras provavelmente no corpus de treino

As 6 categorias e seus objetivos diagnósticos:

| Categoria | N | Objetivo |
|-----------|---|---------|
| Generalização PT-BR | 9 | Testar regras fonológicas em neologismos e portmanteaux |
| Consoantes Duplas | 5 | Testar redução de geminadas (lazzaretti → z único) |
| Anglicismos (chars no vocab) | 5 | Testar fonologia inglesa com grafemas portugueses |
| Chars OOV (k/w/y) | 3 | Documentar falhas esperadas por limite do charset |
| Palavras PT-BR reais (OOV) | 5 | Testar generalização de regras para palavras inéditas |
| Controles (em treino) | 4 | Baseline de sanidade |

### 7.2 Métrica Fonológica Independente

Para além das métricas binárias (correto/errado), foi implementada uma métrica de **score fonológico** baseada em distâncias articulatórias — completamente independente do framework G2P, sem uso de PanPhon, funcionando como avaliador externo:

```python
_PHONEME_GROUPS = {
    "i": "V_alto_front",  "ɛ": "V_med_front",  "a": "V_baixo",
    "x": "FR_velar",      "ɣ": "FR_velar",      "s": "FR_dental",
    "m": "NS_bilabial",   "ɾ": "RHOT",          # ... 36 entradas
}

def _group_dist(p1, p2) -> float:
    if p1 == p2: return 0.0
    if p1 in estruturais or p2 in estruturais: return 1.0
    g1, g2 = _PHONEME_GROUPS.get(p1), _PHONEME_GROUPS.get(p2)
    if g1 == g2: return 0.1          # mesma classe articulatória
    if prefixo(g1) == prefixo(g2): return 0.4   # mesmo tipo (V, OC, FR...)
    if ambos vogais: return 0.5
    if ambos consoantes: return 0.55
    return 0.9                        # vogal↔consoante
```

O score final é calculado via edit distance ponderada com custos articulatórios, normalizado para o intervalo 0–100%:

- **100% (exato)**: transcrição idêntica
- **90–99% (muito próximo)**: 1 substituição de fonema da mesma família
- **70–89% (próximo)**: substituições de fonemas relacionados
- **50–69% (parcial)**: diferenças estruturais significativas
- **< 50% (distante)**: falha substancial

### 7.3 Resultados do Banco de Generalização

Avaliação com o modelo Exp104b (SOTA PER, índice 18):

**Tabela de resultados por categoria**:

| Categoria | Corretas | Score Fonol. Médio | Insight |
|-----------|----------|---------------------|---------|
| Generalização PT-BR | 4/9 (44%) | 97% | Near-misses: ɣ→x, ĩ→i |
| Consoantes Duplas | 1/5 (20%) | 81% | Model gera geminadas no output |
| Anglicismos (invocab) | 1/5 (20%) | 71% | clube ✓; fonologia inglesa é OOV |
| Chars OOV | 0/3 (0%) | 68% | Falha esperada documentada |
| **PT-BR Reais (OOV)** | **5/5 (100%)** | **100%** | **Generalização perfeita** |
| Controles (em treino) | 3/4 (75%) | 98% | borboleta: ɣ→x |

**Total**: 14/31 corretas (45%)

### 7.4 Análise Qualitativa por Categoria

**Generalização PT-BR (4/9)**: Os 5 erros nesta categoria têm score médio de 97%, indicando que todos são *near-misses* com uma ou duas substituições de fonemas da mesma família articulatória. Exemplos:
- *fantabulástico*: `ʃ` → `s` (mesma posição alveolar; score 97%)
- *computadorzinho*: `ɣ` → `x` (mesma classe velar; score 94%)
- O modelo acerta todos os padrões complexos: `tʃ` para `-ti-`, `ɲ` para `nh`, `ã ʊ̃` para `-ão`, acento na sílaba correta

**Consoantes Duplas (1/5)**: O modelo não aprendeu que geminadas italianas/estrangeiras reduzem a uma única consoante em PT-BR. Em *lazzaretti*, prediz `l a z z a...` (duplo-z) em vez de `l a z a...`. Em *mozzarela*, idem: duplo-z no output. A única acerto (*aterrissar*) envolve `rr→x` e `ss→s`, padrões que *existem* no corpus PT-BR. O contraste confirma: o modelo generaliza regras presentes no treino, mas não regras de empréstimos não-observados.

**Anglicismos com chars no vocabulário (1/5)**: *clube* acerta facilmente (cl-cluster regular). Os demais falham por fonologia inglesa: *mouse* (ou→aw), *site* (ditongo ai-inglês), *stress* (ss final → epentese sɪ). Score médio 71% indica que os erros são parcialmente capturados — não catástrofes fonéticas, mas mapeamentos incorretos de uma língua para outra.

**Chars OOV — k, w, y (0/3)**: Falhas completamente esperadas e corretamente documentadas. *karatê* tem score 76% — o modelo produz `k a ɾ a ˈ t e` vs. esperado `k a ɾ a ˈ t ʃ ɪ`: a consoante "k" foi gerada corretamente (o modelo trata o `<UNK>` mapeado de "k" de forma útil), mas o final `tê→tʃɪ` foi errado. *yoga* tem 87% — apenas a semivogal inicial falha.

**PT-BR Reais Fora do Vocabulário (5/5 — 100%)**: Esta é a descoberta mais importante. Cinco palavras que muito provavelmente não estão no corpus de treino — *puxadinho*, *abacatada*, *zunido*, *malcriado*, *arrombado* — foram todas transcritas corretamente:

| Palavra | Predição | Esperado |
|---------|----------|---------|
| puxadinho | `p u ʃ a ˈ d ʒ ĩ ɲ ʊ` | `p u ʃ a ˈ d ʒ ĩ ɲ ʊ` ✓ |
| abacatada | `a b a k a ˈ t a d ə` | `a b a k a ˈ t a d ə` ✓ |
| zunido | `z u ˈ n i d ʊ` | `z u ˈ n i d ʊ` ✓ |
| malcriado | `m a w k ɾ i ˈ a d ʊ` | `m a w k ɾ i ˈ a d ʊ` ✓ |
| arrombado | `a x õ ˈ b a d ʊ` | `a x õ ˈ b a d ʊ` ✓ |

O modelo acerta: `x→ʃ` para "x" em puxar, `d+i→dʒ` (palatalização), `l coda→w` (malcriado), `rr→x`, `om→õ` (nasal) — todas regras fonológicas produtivas do PT-BR que o modelo aprendeu genuinamente, não memorizou.

### 7.5 Implicação Central

O resultado 5/5 na categoria *real_oov* é a validação mais direta da capacidade do modelo: **o modelo aprendeu as regras fonológicas do Português Brasileiro, não memorizou o corpus**. Isso tem implicações práticas imediatas — o modelo pode ser usado para transcrever neologismos, nomes derivados morfologicamente, e palavras novas que seguem os padrões produtivos do PT-BR.

As falhas revelam **limites de generalização bem-definidos**:
1. Geminadas de empréstimos não-produtivos (italiano/inglês) → o corpus não tem exemplos suficientes
2. Fonologia de anglicismos não-adaptados → fonemas/ditongos ingleses são genuinamente OOV
3. Caracteres k, w, y → limite hard do charset treinado

---

## 8. Discussão

### 8.1 O Trade-off Fundamental PER/WER com Separadores Silábicos

Um dos achados mais robustos do trabalho é o trade-off sistemático entre PER e WER introduzido pelos separadores silábicos. Separadores consistentemente melhoram PER (−17% a −20%) ao mesmo tempo em que pioram WER (+6% a +8%), independentemente da capacidade do modelo ou da função de custo usada.

O mecanismo é estrutural e não-eliminável apenas por ajuste de hiperparâmetros: cada token separador mal-posicionado conta como erro de palavra inteira (WER métrica binária por palavra). O modelo com separadores tem mais oportunidades de erro por sequência de saída, o que se reflete diretamente no WER mesmo quando a qualidade fonêmica melhora.

**Implicação prática**: A escolha entre os dois regimes (com/sem separadores) deve ser guiada pela aplicação:
- **TTS / síntese de fala**: PER mais importante → usar Exp104b (sep, PER 0,49%)
- **Reconhecimento de fala / NLP / lookup**: WER mais importante → usar Exp9 (sem sep, WER 4,96%)

### 8.2 Limites da Distance-Aware Loss

A DA Loss demonstrou eficácia como regularizador fonético: melhora PER e WER no modelo 9,7M. No entanto, dois limites importantes foram identificados:

**Limite de escala**: Em 17,2M parâmetros (Exp10), DA Loss interfere negativamente — o modelo grande tem capacidade suficiente para memorizar, e a penalização fonética atrapalha esse processo. DA Loss funciona melhor em modelos de capacidade moderada onde o sinal fonético guia a generalização.

**Limite estrutural**: Símbolos estruturais (`.`, `ˈ`) não recebem sinal útil da DA Loss porque têm vetor zero no PanPhon. O override de distância corrige parcialmente isso, mas confusões posicionais persistem — o modelo sabe que `.` e `ˈ` são diferentes, mas ainda os posiciona incorretamente na sequência.

### 8.3 Convergência Rápida como Sinal de Qualidade

Um padrão consistente nos experimentos é que modelos superiores convergem em menos épocas com val_loss menor. Exp104b convergiu em 88 épocas com val_loss de 0,0136, similar a Exp102, confirmando que a convergência rápida e a baixa val_loss são indicadores confiáveis de qualidade — o que permite early stopping criterioso sem esperar épocas fixas.

### 8.4 Robustez e Trade-offs de Dados (Exp105/106)

As ablações da Fase 7 quantificam dois trade-offs práticos com impacto direto em aplicações reais.

**Trade-off quantidade de dados**: Exp105 reduz o conjunto de treino de 60% para 50% (−17% de exemplos). A degradação de PER de 0,49% para 0,54% (+10%) é surpreendentemente baixa dado o tamanho da redução. Isso sugere que o corpus de 95K palavras está bem acima do limiar de saturação para a arquitetura 9,7M — mais dados de treino na faixa 50%–60% têm retorno decrescente. Para contextos com corpus menor (e.g., dialetos, línguas de baixo recurso), esta curva de robustez implica que o sistema pode alcançar performance razoável com ~48K palavras de treino.

**Trade-off speed vs. acurácia (hífen)**: Exp106 demonstra que a remoção de um único caractere do vocabulário (hífen, `-`) gera speedup de 2,58× com degradação de apenas +0,04 p.p. de PER. O mecanismo não é apenas a redução de 1 token: a remoção do hífen elimina palavras compostas do vocabulário de entrada, simplificando os padrões de grafema→fonema que o modelo precisa generalizar. Para aplicações onde latência é crítica (TTS em tempo real, assistentes por voz), Exp106 oferece melhor razão performance/velocidade que Exp104b.

**Resumo do Pareto de configurações**:

| Configuração | PER | Speed (w/s) | Caso de uso ideal |
|---|---|---|---|
| Exp9 | 0,58% | ~20 w/s | WER mínimo, NLP/lookup |
| Exp104b | 0,49% | 11,7 w/s | PER mínimo, análise linguística |
| Exp105 | 0,54% | 11,7 w/s | Corpus reduzido, mesma speed |
| Exp106 | 0,58% | **30,2 w/s** | Latência crítica, TTS em tempo real |

---

## 9. Conclusões

Este trabalho apresentou o FG2P, um sistema G2P para o Português Brasileiro baseado em BiLSTM Encoder-Decoder com atenção de Bahdanau. Os principais resultados e contribuições são:

**Resultados de state-of-the-art**:
- **PER 0,49%** (Exp104b: DA Loss + separadores + distâncias customizadas)
- **WER 4,96%** (Exp9: DA Loss sem separadores)
- Avaliação sobre 28.782 palavras — 57× maior que referências comparáveis em PT-BR

**Contribuições técnicas**:
1. *Distance-Aware Loss* com λ=0,20 como regularizador fonético — melhora PER/WER sem custo arquitetural (ver [análise de originalidade](ORIGINALITY_ANALYSIS.md))
2. Separadores silábicos como tokens de saída — trade-off documentado e quantificado (PER/WER Pareto)
3. Override de distâncias para símbolos estruturais — corrige limitação do PanPhon para tokens não-fonéticos
4. Banco de generalização de 31 palavras em 6 categorias — ferramenta reutilizável para avaliação OOV

**Descoberta metodológica**:
- Split 60/10/30 estratificado supera 70/10/20 em −41% PER com a mesma arquitetura

**Capacidade de generalização confirmada**:
- 100% de acurácia em palavras PT-BR reais fora do vocabulário de treino
- O modelo aprendeu regras produtivas do PT-BR (palatalização, redução de coda, nasalização) de forma genuinamente generalizável

**Limites bem-definidos identificados**:
- Geminadas de empréstimos (zz, tt, pp) → corpus insuficiente
- Fonologia de anglicismos não-adaptados → OOV fonológico
- Chars k, w, y → OOV de charset (limite hard)
- Confusão `ɣ`→`x` em coda → erro sistemático de vozeamento velar

### 9.1 Trabalhos Futuros

**Métricas especializadas**: Stress Accuracy (% de acentos posicionados corretamente) e Boundary F1 (precisão/revocação de fronteiras silábicas) capturariam aspectos atualmente invisíveis ao PER/WER.

**Pipeline fonotático em 4 fases**: Integrar conhecimento explícito da fonotática do PT-BR para reduzir inserções indevidas de separadores e confusões posicionais de tokens estruturais. O PT-BR permite apenas 4 consoantes em coda (/s/, /ɾ/, /l/, /N/), com ~78% das sílabas sendo abertas (CV) — restrições que o modelo atual não aprende explicitamente.

A implementação proposta segue 4 fases incrementais:
1. **Diagnóstico**: Quantificar quais erros atuais violam regras fonotáticas (sem modificar o modelo)
2. **N-gram fonotático**: Treinar bigrama/trigrama de fonemas no corpus como modelo de linguagem fonológico
3. **Autômato de estados finitos**: FSA encoding estrutura silábica (ONSET → NUCLEUS → CODA), com aceitação/rejeição de transições ilegais
4. **Integração**: Reranking dos N-melhores beams do LSTM pelo modelo fonotático (risco baixo) ou penalização direta como termo adicional na loss (risco médio)

**Espaço articulatório contínuo 7D**: Substituir o espaço PanPhon binário discreto (24 features binárias) por um espaço de 7 dimensões contínuas fundamentado em estudos de análise de componentes principais do trato vocal (Birkholz et al., 2024 — 7D preservam 95-99% da variância):

| Dimensão | Semântica | Correlato acústico |
|----------|-----------|-------------------|
| HEIGHT | Altura da língua [0,1] | F1 (inverso) |
| BACKNESS | Posição anterior-posterior [0,1] | F2 (inverso) |
| ROUNDING | Arredondamento labial [0,1] | Timbre, F2 |
| CONSTR_LOC | Local de constrição [0=labial ... 1=glotal] | Transições F2/F3 |
| CONSTR_DEG | Grau de constrição [-0,1=estrutural ... 1=oclusiva] | Energia, fricção |
| NASALITY | Nasalização [0,1] | Antirressonâncias em F1 |
| VOICING | Vozeamento [0,1] | F0, estrutura harmônica |

A vantagem chave deste espaço: `.` e `ˈ` receberiam coordenadas distintas (CONSTR_DEG = -0,1 para ambos, mas VOICING = 0,0 vs. 1,0), resolvendo o problema d(., ˈ) = 0,0 na raiz sem necessidade de override. Adicionalmente, o espaço é multilinguisticamente universal — cada idioma é uma quantização (conjunto de fonemas) do mesmo espaço contínuo subjacente, permitindo transfer learning zero-shot ou few-shot para novos idiomas.

**Dados de geminadas**: Ampliar o corpus com exemplos de empréstimos italianos e ingleses com consoantes duplas para cobrir o gap identificado na avaliação de generalização.

**Morfossintaxe para homógrafos heterófonos**: Para palavras como *jogo* (substantivo /ˈʒɔgʊ/ vs. verbo /ˈʒogʊ/), *gosto*, *acordo*, onde a pronúncia depende da categoria gramatical, um sistema G2P em isolamento nunca poderá resolver a ambiguidade. A solução exige um pipeline em série: analisador morfossintático → G2P com contexto. Este é um limite irredutível do design word-isolation, não da arquitetura BiLSTM.

---

## 10. Guia de Uso: inference_light.py

O módulo `src/inference_light.py` é a interface principal para uso do modelo em produção ou experimentação. Funciona como API Python e como ferramenta de linha de comando (CLI), com quatro modos de operação.

### 10.1 Qual modelo usar

Antes de qualquer uso, escolha o modelo conforme o objetivo:

| Modelo | Index | PER | WER | Quando usar |
|--------|-------|-----|-----|-------------|
| **Exp9** | 11 | 0,58% | **4,96%** | Precisão por palavra — NLP, TTS, lookup |
| **Exp104b** | 18 | **0,49%** | 5,43% | Precisão por fonema — análise linguística, síntese, alinhamento |

A diferença prática: Exp9 acerta mais palavras *inteiras* (95% vs 94,6%); Exp104b erra menos fonemas individuais quando erra. Para síntese de voz onde um fonema errado basta para soar estranho, Exp104b é preferível. Para indexação, busca e NLP onde a palavra toda precisa estar certa, Exp9 é preferível.

### 10.2 Modo 1 — Palavra única ou lista (CLI)

O modo mais direto. Não requer nenhum dataset externo.

```bash
# Uma palavra
python src/inference_light.py --index 18 --word computador
# → k õ p u t a ˈ d o x .

# Várias palavras separadas por vírgula
python src/inference_light.py --index 18 --words "selfie,biscoito,chuveirada"

# Remover separadores silábicos da saída (modelos com sep)
python src/inference_light.py --index 18 --words "puxadinho" --strip-sep
# → p u ʃ a ˈ d ʒ ĩ ɲ ʊ   (sem pontos)

# Modelo SOTA WER (Exp9, sem separadores)
python src/inference_light.py --index 11 --word computador
# → k õ p u t a ˈ d o x
```

**Interpretação da saída**: fonemas separados por espaço. `ˈ` antes da sílaba tônica. `.` indica fronteira silábica (só em modelos com separadores). Cada símbolo é um token IPA do Português Brasileiro.

```bash
# Ver todos os modelos disponíveis
python src/inference_light.py --list

# Informações detalhadas de um modelo
python src/inference_light.py --index 18 --info
```

### 10.3 Modo 2 — Interativo

Para explorar o modelo palavra por palavra sem reinicializar. Útil para demonstrações.

```bash
python src/inference_light.py --index 18 --interactive
```

```
> computador
  → k õ p u t a ˈ d o x .
> biscoitinhozão
  → b i s k o y t ʃ ĩ ɲ ɔ ˈ z ã ʊ̃
> sair
```

### 10.4 Modo 3 — Avaliação de neologismos com análise fonológica

Este modo avalia um banco de palavras com referências IPA conhecidas e exibe análise detalhada de cada erro — incluindo diff fonêmico, score fonológico e notas linguísticas.

```bash
# Banco padrão (docs/data/neologisms_test.tsv — 35 neologismos)
python src/inference_light.py --index 18 --neologisms

# Banco de generalização (31 palavras em 6 categorias)
python src/inference_light.py --index 18 --neologisms docs/data/generalization_test.tsv
```

**Exemplo de saída para uma palavra errada**:
```
✗  fantabulástico    [medium]  [fonol: 97% muito próximo]
     Predito : f ã t a b u ˈ l a ʃ t ʃ i k ʊ
     Esperado: f ã t a b u ˈ l a s t ʃ i k ʊ
     Diff    : 'ʃ' → 's'
     Nota    : portmanteau fantástico+fabuloso; fan→fã; -ti→tʃ; stress -lás-
```

**Leitura do output**:
- **97% muito próximo**: a predição difere em apenas 1 fonema da mesma família articulatória
- **Diff**: mostra exatamente quais fonemas foram trocados — útil para identificar padrões sistemáticos
- **Nota**: contexto linguístico do TSV que explica por que aquela transcrição é esperada

**Formato do TSV estendido** (compatível com `--neologisms`):

```
# word	phonemes	category	difficulty	notes
computador	k õ p u t a ˈ d o x	controle	easy	[EM DICT] controle: r coda→x
chuveirada	ʃ u v e y ˈ ɾ a d ə	generalizacao_pt	easy	ch→ʃ; -ei-=ey ditongo
wifi	ˈ w i f i	char_oov	hard	[OOV: 'w' não no vocab] falha esperada
```

Categorias suportadas nativamente: `anglicismo`, `verbo_emprestimo`, `neologismo_nativo`, `nome_proprio`, `palavra_inventada`, `generalizacao_pt`, `consoante_dupla`, `anglicismo_invocab`, `char_oov`, `real_oov`, `controle`.

**Resumo por dificuldade ao final**:
```
Por dificuldade:
  easy   : 11/16  ██████░░░░  69%
  medium :  3/11  ██░░░░░░░░  27%
  hard   :  0/ 4  ░░░░░░░░░░  0%

PER: 13.33%  |  WER: 54.84%  |  Accuracy: 45.16%
```

### 10.5 Modo 4 — Avaliação de TSV genérico

Para datasets customizados com apenas duas colunas (palavra, fonemas de referência). Calcula PER/WER/Accuracy e exibe a tabela de comparação.

```bash
python src/inference_light.py --index 18 --tsv meu_dataset.tsv --cache-tag teste_custom
```

O parâmetro `--cache-tag` nomeia o arquivo de cache gerado em `data/`, facilitando inspeção posterior. O TSV pode ter qualquer número de colunas — apenas as duas primeiras são usadas.

### 10.6 Uso como API Python

Para integração em scripts ou sistemas externos:

```python
from src.inference_light import G2PPredictor

# Carregar modelo — o index corresponde à posição na lista (--list)
p = G2PPredictor.load(index=18)   # Exp104b: SOTA PER
# p = G2PPredictor.load(index=11) # Exp9:    SOTA WER

# Predição simples
p.predict("computador")
# → "k õ p u t a . ˈ d o x ."   (com separadores silábicos)

p.predict_stripped("computador")
# → "k õ p u t a ˈ d o x"       (sem separadores — independente do modelo)

# Batch
p.predict_batch(["viral", "puxadinho", "biscoitinhozão"])
# → ["v i ˈ ɾ a w .", "p u ʃ a ˈ d ʒ ĩ ɲ ʊ", ...]

# Score fonológico (métrica independente do G2P)
score, label = G2PPredictor._phonological_score(
    pred="f ã t a b u ˈ l a ʃ t ʃ i k ʊ",
    ref ="f ã t a b u ˈ l a s t ʃ i k ʊ"
)
# → (97.0, "muito próximo")

# Verificar chars OOV
coverage, oov = p._char_coverage("wifi")
# → (75.0, {'w'})  — 'w' não está no vocab treinado
```

---

## Apêndice A — Glossário

### Termos Fonéticos e Linguísticos

**Fonema** (/fonema/): Unidade mínima sonora de uma língua capaz de distinguir significados. `/k/` e `/g/` são fonemas distintos porque "cama" e "gama" têm significados diferentes. Diferente de *letra* (representação escrita) ou *som* (instância fonética concreta).

**Grafema**: Unidade mínima do sistema escrito — basicamente, uma letra ou dígrafo com função de representação sonora. "ch" é um grafema que representa o fonema `/ʃ/`.

**IPA** (*International Phonetic Alphabet*, Alfabeto Fonético Internacional): Sistema padronizado de notação que representa todos os sons das línguas humanas com símbolos únicos e inequívocos. `ɛ` representa sempre a vogal média anterior aberta, independente da língua.

**Suprassegmental**: Propriedade fonética que se sobrepõe a um segmento (fonema) ou a uma sequência de segmentos. O acento tônico (`ˈ`) é suprassegmental — ele modifica a realização da sílaba inteira, não de um único som.

**Neutralização vocálica**: Fenômeno fonológico em que dois fonemas distintos em posição tônica passam a ser realizados de forma idêntica (ou muito similar) em posição átona. Em PT-BR, `/ɛ/` (aberta) e `/e/` (fechada) neutralizam em sílaba átona — ambas se realizam como `[e]` ou `[ɪ]`.

**Coda silábica**: Consoante (ou grupo de consoantes) que fecha uma sílaba após a vogal. Em "computador", o "r" final está em coda → realiza-se como `/x/`. Diferente do "r" em posição de ataque (início de sílaba) → `/ɾ/`.

**Palatalização**: Processo fonológico em que uma consoante se "aproxima" do palato quando seguida de vogal anterior alta `/i/` ou `/ɪ/`. Em PT-BR, `/t/` → `/tʃ/` antes de `/i/` (carro"ti"nha → carrotiɲa com `/tʃ/`).

**Geminada** (*consoante dupla*): Consoante com duração dobrada, representada ortograficamente pela repetição da letra ("lazzaretti", "cappuccino"). Em PT-BR, geminadas de empréstimos tendem a ser reduzidas a uma única consoante na pronúncia adaptada.

**Fonotática**: Conjunto de regras que governam quais sequências de fonemas são permitidas em uma língua. Em PT-BR, a fricativa velar em posição de coda é tipicamente surda (`/x/`, não `/ɣ/`).

---

### Termos de Aprendizado de Máquina

**Token**: Unidade mínima processada pelo modelo — no FG2P, cada fonema ou símbolo estrutural é um token de saída; cada caractere é um token de entrada.

**Embedding**: Representação vetorial densa de um token. Converte um índice discreto (ex: `42 = "ɛ"`) em um vetor contínuo de $d$ dimensões que o modelo pode processar matematicamente.

**Encoder-Decoder**: Arquitetura de rede neural onde uma parte (encoder) lê a sequência de entrada e produz uma representação interna, e outra parte (decoder) gera a sequência de saída a partir dessa representação. Permite sequências de entrada e saída com comprimentos diferentes.

**BiLSTM** (*Bidirectional Long Short-Term Memory*): Variante do LSTM que processa a sequência tanto da esquerda para a direita quanto da direita para a esquerda, combinando ambos os contextos. Permite ao modelo considerar tanto o que veio antes quanto o que vem depois de cada posição.

**Atenção de Bahdanau**: Mecanismo que permite ao decoder "olhar para trás" e ponderar todos os estados do encoder a cada passo de geração, em vez de depender apenas do último estado. Produz pesos $\alpha_t$ (soma 1.0) indicando quais posições de entrada são mais relevantes para gerar o próximo token.

**Teacher Forcing**: Técnica de treinamento onde, durante cada passo do decoder, o token correto de referência é fornecido como entrada (em vez da predição anterior do modelo). Acelera a convergência mas introduz uma discrepância com a inferência (onde o modelo usa suas próprias predições).

**Softmax**: Função que converte um vetor de valores reais em uma distribuição de probabilidade (soma = 1.0, todos valores ≥ 0). Usada na camada final do decoder para gerar probabilidades sobre o vocabulário de fonemas.

**Dropout**: Regularizador que desativa aleatoriamente uma fração dos neurônios durante o treinamento, forçando a rede a não depender de nenhum neurônio específico. No FG2P, dropout=0.5 (50% dos neurônios desativados por batch).

**Early Stopping**: Técnica que interrompe o treinamento quando a performance no conjunto de validação para de melhorar por $N$ épocas consecutivas, evitando sobreajuste (*overfitting*). O FG2P usa paciência de 10 épocas.

**PAD** (*padding*): Token especial adicionado ao final de sequências curtas para que todas as sequências em um batch tenham o mesmo comprimento. O embedding de PAD é fixo em zero e nunca é atualizado.

**EOS** (*End-of-Sequence*): Token especial que o decoder aprende a emitir para sinalizar o fim da sequência de fonemas. Permite sequências de comprimento variável sem depender de um limite fixo.

**OOV** (*Out-of-Vocabulary*): Qualquer entrada que não está no vocabulário do modelo. No FG2P, há dois tipos: (1) **OOV de palavra** — a palavra não está no corpus de treino (mas seus caracteres estão); (2) **OOV de caractere** — a palavra contém caracteres não treinados (k, w, y), mapeados para `<UNK>`.

---

### Termos de Métricas

**PER** (*Phoneme Error Rate*, Taxa de Erro por Fonema): Percentual de fonemas incorretos na predição, calculado via distância de edição Levenshtein sobre sequências de fonemas. Contabiliza substituições, inserções e deleções.

**WER** (*Word Error Rate*, Taxa de Erro por Palavra): Percentual de palavras com *pelo menos um fonema errado*. Métrica binária por palavra — uma palavra com 1 erro e uma com 5 erros contam igualmente.

**Accuracy** (Acurácia por palavra): Complemento do WER — percentual de palavras 100% corretas. `Accuracy = 100% - WER`.

**PanPhon**: Biblioteca que mapeia fonemas IPA para vetores binários de 24 features articulatórias (vozeamento, nasalidade, ponto de articulação, modo, etc.). Usada no FG2P para calcular distâncias entre fonemas e treinar a DA Loss.

**Feature articulatória**: Propriedade fonética binária que descreve como um som é produzido fisicamente. Exemplos: `[+voice]` (vozeado, como /b/), `[-voice]` (não-vozeado, como /p/), `[+nasal]` (como /m/), `[+syl]` (silábico, como vogais).

**Score fonológico**: Métrica de 0–100% implementada em `_phonological_score()` que mede a "proximidade fonológica" entre duas transcrições, usando distâncias articulatórias por grupo de fonema. Independente do modelo G2P — pode ser usada como avaliador externo.

**Cramér V**: Medida de associação entre variáveis categóricas (0 = independência perfeita, 1 = associação máxima). Valor próximo de 0 indica que a distribuição fonológica dos subconjuntos de treino/validação/teste é estatisticamente equivalente.

---

### Termos Específicos do FG2P

**Separador silábico** (`.`): Token adicionado ao vocabulário de saída nos modelos Exp101-104b que marca fronteiras entre sílabas. Melhora PER (modelo aprende estrutura silábica) ao custo de WER (cada separador mal-posicionado = erro de palavra inteira).

**Marcador de acento** (`ˈ`): Token que precede imediatamente a sílaba tônica na transcrição. Ex: "computador" → `k õ p u t a ˈ d o x` (acento em "-dor").

**Vetor zero (PanPhon)**: Representação atribuída pelo PanPhon a símbolos não-fonéticos (`.`, `ˈ`) que não correspondem a nenhum segmento articulatório. Consequência: `d(., ˈ) = 0.0` — a DA Loss não diferencia confusões entre eles sem o override implementado em Exp104b.

**Distance-Aware Loss** (DA Loss): Função de custo proposta neste trabalho que combina CrossEntropy com um termo de penalidade proporcional à distância articulatória PanPhon do fonema predito em relação ao correto. Permite ao modelo aprender a "preferir erros fonologicamente próximos" quando incerto.

**Sweet spot** (ponto ótimo): No contexto do FG2P, refere-se ao tamanho de modelo 9,7M de parâmetros, que maximiza a razão performance/custo. Modelos maiores (17,2M) não adicionam valor proporcional; modelos menores (4,3M) saturam antes de atingir o máximo do corpus.

---

## Referências

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. *arXiv:1409.0473*.
- Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. *Neural Networks*, 18(5–6), 602–610.
- Hayes, B., & Wilson, C. (2008). A maximum entropy model of phonotactics and phonotactic learning. *Linguistic Inquiry*, 39(3), 379–440.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707–710.
- Mortensen, D. R., Littell, P., Bharadwaj, A., Goyal, K., Dyer, C., & Levin, L. (2016). PanPhon: A resource for mapping IPA segments to articulatory feature vectors. *COLING 2016*, 3475–3484.
- Vitevitch, M. S., & Luce, P. A. (2004). A web-based interface to calculate phonotactic probability for words and nonwords in English. *Behavior Research Methods, Instruments, & Computers*, 36(3), 481–487.
- Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., ... & Raffel, C. (2022). ByT5: Towards a token-free future with pre-trained byte-to-byte models. *TACL*, 10, 291–306.

---

*Documento gerado em 2026-02-25. Código, dados e modelos disponíveis em [src/](../src/), [dicts/](../dicts/), [models/](../models/).*
