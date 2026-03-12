# Distance-Aware Loss

**Arquivo**: `docs/article/DA_LOSS_ANALYSIS.md`
**Relacionado a**: `docs/article/ARTICLE.md §4.2` · `src/losses.py`
**Status**: Análise pós-experimentos · base para trabalhos futuros

---

## 1. A Fórmula Real

A implementação exata em `src/losses.py:286`:

```python
combined_loss = ce_loss_per_token + self.distance_lambda * distance_loss_per_token
```

Onde:

```python
ce_loss_per_token    = nn.CrossEntropyLoss(logits, target)   # -log(p_correct)
distance_loss_per_token = distances * pred_probs              # d(pred,correct) * p(pred)
```

Expandido:

```
L = -log(p_correct) + λ · d(pred, correct) · p(pred)
        ↑                    ↑                  ↑
   CE clássica          distância            confiança no
   (penaliza o que      fonológica           fonema predito
    deveria ter saído)  (0..1 normaliz.)     (argmax, não
                                              o correto)
```

**Detalhe crítico**: CE olha para `p_correct` (fonema **correto**). DA olha para `p_pred` (fonema **predito**, o argmax). São variáveis independentes — daí a complementaridade.

---

## 2. Intervalo Matemático e Computacional

### 2.1 Cross-Entropy

| Aspecto | Matemático | Computacional (PyTorch) |
|---------|-----------|------------------------|
| Fórmula | `L_CE = -log(p_correct)` | `log_softmax + nll_loss` (log-sum-exp trick) |
| Mínimo | 0 — quando p_correct = 1.0 | ~0 (+ ruído FP32) |
| Máximo | +∞ — quando p_correct → 0 | ~16 — softmax nunca chega a 0 |
| Intervalo prático (treino) | — | **[0, ~5]** na maioria dos batches |

O log-sum-exp trick garante estabilidade numérica: PyTorch **não** computa `softmax` seguido de `log`, mas sim `logsoftmax` diretamente, evitando underflow quando logits têm magnitudes altas.

### 2.2 Distance-Aware Term

| Componente | Intervalo | Origem |
|-----------|----------|--------|
| `d(pred, correct)` | [0, 1] | normalizado por `max_dist` euclidiano (losses.py:117-119) |
| `p(pred)` | (0, 1] | softmax — nunca exatamente 1 |
| `d × p_pred` | [0, ~0.99] | produto de dois termos em [0,1] |
| `λ × d × p_pred` | **[0, ~0.198]** | λ=0.20 é o teto absoluto |

**DA é always bounded por ~0.20.** CE pode ir a 16. Essa assimetria é fundamental para entender onde DA é efetivo.

---

## 3. Exemplos Numéricos Completos

Todos os valores calculados com λ=0.20 (valor do Exp104b).

### 3.1 Cenários Fundamentais

**CASO 1 — Acerto perfeito**
```
argmax = correto = /s/,  p(s) = 0.95
CE = -log(0.95) = 0.051
DA =  0.00 × 0.95 = 0.000   (d=0 porque pred=correto)
L  = 0.051 + 0.20 × 0.000 = 0.051
```

**CASO 2 — Erro leve** (ɛ→e, distância d=0.10)
```
argmax = /ɛ/,  correto = /e/,  p(e) = 0.40,  p(ɛ) = 0.45
CE = -log(0.40) = 0.916
DA =  0.10 × 0.45 = 0.045 → λ × 0.045 = 0.009
L  = 0.916 + 0.009 = 0.925
```

**CASO 3 — Erro grave** (k→s, distância d=0.80), **mesmo p_correct do Caso 2**
```
argmax = /k/,  correto = /s/,  p(s) = 0.40,  p(k) = 0.45
CE = -log(0.40) = 0.916   ← IDÊNTICO ao Caso 2!
DA =  0.80 × 0.45 = 0.360 → λ × 0.360 = 0.072
L  = 0.916 + 0.072 = 0.988
```

> **Insight fundamental**: CE é **cega à distância**. Com a mesma distribuição de probabilidades (`p_correct=0.40`), CE gera o mesmo gradiente independente de o erro ser leve ou grave. DA é o único componente que diferencia os dois casos: +0.063 de L no erro grave.

### 3.2 Quando CE é alto, DA ainda ajuda?

**Situação: modelo muito confiante no fonema errado** — p(correto)=0.02, p(errado)=0.90

| Tipo de erro | CE | DA | L total | DA como % |
|-------------|----|----|---------|-----------|
| Leve  (d=0.10) | 3.912 | 0.018 | **3.930** | 0.5% |
| Grave (d=0.80) | 3.912 | 0.144 | **4.056** | 3.5% |
| Diferença | 0 | 0.126 | 0.126 | +3.2% |

**Situação: modelo na dúvida** — p(correto)=0.40, p(errado)=0.45

| Tipo de erro | CE | DA | L total | DA como % |
|-------------|----|----|---------|-----------|
| Leve  (d=0.10) | 0.916 | 0.009 | **0.925** | 1.0% |
| Grave (d=0.80) | 0.916 | 0.072 | **0.988** | 7.3% |
| Diferença | 0 | 0.063 | 0.063 | +7.0% |

**Situação: zona de transição** — p(correto)=0.70, p(errado)=0.25

| Tipo de erro | CE | DA | L total | DA como % |
|-------------|----|----|---------|-----------|
| Leve  (d=0.10) | 0.357 | 0.005 | **0.362** | 1.4% |
| Grave (d=0.80) | 0.357 | 0.040 | **0.397** | 10.0% |
| Diferença | 0 | 0.035 | 0.035 | +9.7% |

**Conclusão**: DA é mais efetivo na zona de transição (CE 0.3–1.5), onde o modelo está **ativamente aprendendo** qual fonema escolher. Quando CE é alto (modelo muito errado), DA representa menos de 5% do sinal total — contribui, mas é ruído sobre o domínio da CE.

---

## 4. O Papel de λ (lambda)

λ controla a **força relativa do sinal fonológico**. Com λ=0.20, a contribuição máxima de DA é ~0.20 (teto absoluto). Os resultados do Exp7 mostraram:

| λ | PER | Interpretação |
|---|-----|---------------|
| 0.05 | 0.68% | DA muito fraco; CE domina completamente — abaixo do CE puro |
| 0.10 | 0.63% | Melhora moderada |
| **0.20** | **0.61%** | **Ótimo empírico — curva U-invertido** |
| 0.50 | 0.73% | DA sobrepenaliza — modelo hesita mesmo quando confiante no correto |

A curva em U-invertido explica o mecanismo: λ muito baixo = DA inócuo; λ muito alto = DA competindo com CE e causando instabilidade no gradiente.

### 4.1 Por que o ótimo está em 0.20 e não em 0.50 ou maior?

Com λ=0.50 e um erro grave (d=0.80, p_pred=0.90):
```
L_DA = 0.50 × 0.80 × 0.90 = 0.360
```

Se a CE for 0.5 (zona de aprendizado ativo), DA representa:
```
0.360 / (0.5 + 0.360) = 42% do sinal total
```

Com 42% do gradiente vindo de distância fonológica, o modelo começa a **otimizar para minimizar distância** em vez de **maximizar p_correct**. Isso é análogo a label smoothing excessivo: o modelo distribui probabilidade entre candidatos próximos ao invés de concentrar no correto.

### 4.2 λ Ótimo vs. Capacidade do Modelo

Uma hipótese não testada (ver §6): λ ótimo pode variar com a capacidade do modelo.

| Capacidade | Raciocínio | λ hipotético |
|-----------|-----------|-------------|
| 4.3M (Exp6/7) | Menor memorização, mais dependente de generalização | 0.20 |
| 9.7M (Exp9/104b) | Mais capacidade, CE mais forte — DA pode ser maior | 0.20–0.30 |
| 17.2M (Exp10) | DA Loss já interfere negativamente — modelo memoriza | < 0.10 ou desativar |

---

## 5. Interação com BiLSTM + Atenção

A DA Loss não opera isoladamente — ela interage com o gradiente que flui pelo BiLSTM encoder e o mecanismo de atenção.

### 5.1 O que o Decoder Recebe

No passo de decodificação `t`, o decoder recebe:

```
input_t = concat(embedding(ŷ_{t-1}), context_t)

context_t = Σ α_{t,i} · h_i       (Atenção de Bahdanau)
            i=1..T

α_{t,i} = softmax(e_{t,i})
e_{t,i} = v^T · tanh(W_s · s_t + W_h · h_i)
```

Onde `h_i` são os estados ocultos do BiLSTM encoder e `s_t` é o estado do decoder.

### 5.2 Como DA Loss Modifica o Gradiente

**Com CE pura**, o gradiente no decoder penaliza apenas `1 - p_correct`:

```
∂L_CE / ∂logit_j = p_j - 1(j == correct)   (softmax + NLL gradiente)
```

Isso empurra `logit_correct` para cima e todos os outros igualmente para baixo.

**Com DA adicionado**, aparece um termo extra:

```
∂L_DA / ∂logit_j ≈ λ · d(j, correct) · ∂(p_j)/∂logit_j
                  = λ · d(j, correct) · p_j · (1 - p_j)   (para j=pred)
```

Isso significa que fonemas **mais distantes** do correto recebem um empurrão extra de gradiente "negativo" — seus logits são mais fortemente penalizados. Fonemas **próximos** ao correto recebem penalidade fraca.

**Efeito acumulado nos pesos do BiLSTM**: ao longo de milhares de batches, os pesos do encoder aprendem representações onde grafemas em contextos de ambiguidade (ex: "c" antes de "e/i" vs. "a/o") produzem `h_i` que naturalmente favorecem fonemas da mesma família fonológica. A atenção aprende a focar nos grafemas que *resolvem* a ambiguidade porque os gradientes DA penalizam mais quando o modelo foca nos grafemas errados e prediz fonemas distantes.

### 5.3 Atenção e DA Loss

O mecanismo de atenção é especialmente afetado pela DA Loss em casos de **ambiguidade contextual**:

**Exemplo: "c" em "cena" vs "cama"**

Sem DA: o decoder aprende apenas `p(s) = 1, p(k) = 0` — gradiente binário sobre o contexto.

Com DA: quando o modelo erra `c→k` em "cena" (deveria ser `c→s`):
```
d(k, s) = 0.80    ← grande distância
penalidade extra: λ × 0.80 × p(k)   ← sinal forte
```

Esse sinal forte se propaga via backpropagation até os pesos de atenção `W_s` e `W_h`, ensinando que "olhar para o grafema seguinte e/i é crítico para evitar penalidades grandes". Com CE pura, qualquer erro tem o mesmo gradiente independente de o contexto ser mais ou menos informativo.

---

## 6. Limitações Identificadas e Sugestões de Melhoria

### 6.1 Limitação Principal: DA é Bounded, CE é Unbounded

Como visto na §3.2, DA representa menos de 5% do sinal quando CE é alto. Isso é intencional (CE deve dominar), mas significa que **DA não consegue diferenciar bem erros graves de erros leves quando o modelo já está muito errado**.

Durante as primeiras épocas (épocas 1–20), quando CE é tipicamente alto (modelos inicializa randomicamente, CE ≈ log(vocab_size) ≈ 3.9 para 50 fonemas), DA quase não contribui. O sinal fonológico só começa a ter peso relevante quando CE cai para a zona 0.3–1.5 — geralmente épocas 30–80.

### 6.2 Sugestão A: DA Loss Escalada por Confiança na CE

Uma alternativa que daria mais impacto à distância é normalizar o termo DA pelo valor atual da CE, mantendo-o como percentagem constante do sinal total:

```
L_prop = CE × (1 + λ · d(pred, correct) · p(pred))
```

Com essa fórmula:
- CE=0.5, erro grave (d=0.80, p=0.45): `L = 0.5 × (1 + 0.20 × 0.80 × 0.45) = 0.5 × 1.072 = 0.536`
- CE=4.0, erro grave (d=0.80, p=0.45): `L = 4.0 × (1 + 0.20 × 0.80 × 0.45) = 4.0 × 1.072 = 4.288`

DA agora escala **proporcionalmente com a CE** — sempre representando a mesma percentagem do sinal independente de o modelo estar bem ou mal.

**Desvantagem**: perde interpretabilidade; λ deixa de ter escala absoluta comparável entre diferentes configurações.

### 6.3 Sugestão B: DA Loss com Threshold de Distância

Penalizar apenas erros acima de um limiar de distância mínima, ignorando "quase-acertos" fonológicos:

```
L_thresh = CE + λ · max(0, d - δ) · p(pred)
```

Com δ=0.25 (fonemas da mesma classe mas features diferentes):
- Erro leve  d=0.10: `max(0, 0.10-0.25) = 0` — ignorado
- Erro médio d=0.40: `max(0, 0.40-0.25) = 0.15` — penalidade pequena
- Erro grave d=0.80: `max(0, 0.80-0.25) = 0.55` — penalidade forte

**Vantagem**: foca o sinal fonológico apenas nos erros que realmente importam (evita "desperdiçar" sinal em confusões fonologicamente aceitáveis como ɛ/e).

### 6.4 Sugestão C: λ Adaptativo por Época

λ fixo em 0.20 é subótimo em fases diferentes do treino:
- Épocas 1–30: CE alta, DA inócuo → λ ideal seria maior para forçar sinal fonológico
- Épocas 30–80: zona de transição → λ=0.20 é ótimo
- Épocas 80+: modelo ajustando detalhes → λ pode ser reduzido gradualmente (annealing)

```python
# Schedule hipotético:
λ(epoch) = λ_max × exp(-k × epoch)     # exponential decay
# ou
λ(epoch) = λ_max × (1 - epoch/total)   # linear decay
```

Isso é análogo ao learning rate scheduling e pode ser implementado sem mudanças na fórmula base.

### 6.5 Sugestão D: DA Loss Assimétrica

Atualmente d(pred, correct) = d(correct, pred) — a matriz é simétrica. Mas nem todos os erros são igualmente custosos nas duas direções:

- `ɛ→e` em posição tônica: perceptível ao ouvido → deveria custar mais
- `ɛ→e` em posição átona: neutralização natural em PT-BR → poderia custar menos

Uma matriz de distância assimétrica com pesos contextuais (posição silábica, stress) poderia capturar isso. A implementação exigiria expandir `_build_distance_matrix()` para receber contexto fonológico.

---

## 7. Resumo: Vantagens e Desvantagens Reais

| Aspecto | DA Loss Atual (λ=0.20) |
|---------|----------------------|
| **Eficácia na zona de aprendizado** | Alta — orienta o modelo quando incerto entre candidatos |
| **Eficácia em erros graves** | Baixa — CE domina, DA representa < 5% do sinal |
| **Eficácia em acertos** | Nula — d=0 quando pred=correct |
| **Impacto na atenção** | Indireto — propaga gradiente diferenciado para pesos W_s, W_h |
| **Custo computacional** | Baixo — lookup em matriz pré-computada + multiplicação elemento a elemento |
| **Interpretabilidade de λ** | Alta — escala absoluta [0, 0.20] independe da CE |
| **Robustez a modelos grandes** | Baixa — Exp10 (17.2M) mostrou interferência negativa |
| **Limitação de escala** | DA bounded em 0.20; CE unbounded em [0, ~16] |

---

## 8. Conexão com o Artigo (ARTICLE.md)

- **§4.2**: Fórmula e intuição da DA Loss — base desta análise
- **§4.3**: Distâncias customizadas para `.` e `ˈ` — caso especial de DA
- **§8.2**: Limites identificados (escala e estrutural)
- **§9.1**: Trabalhos futuros — sugestões C (λ adaptativo) e D (assimetria) têm potencial

As sugestões A–D desta análise são candidatas a experimentos futuros. As mais promissoras dado o setup atual são **B (threshold)** e **C (λ por época)** — ambas implementáveis sobre a infraestrutura existente sem mudanças arquiteturais.

---

## Glossário

Termos técnicos usados neste documento, em ordem do mais básico ao mais específico.

---

### Termos de Redes Neurais

**Logit**
Valor numérico bruto produzido pela última camada linear de uma rede neural, antes de qualquer normalização. O nome vem da função logística inversa, mas no uso moderno em deep learning significa simplesmente "score não-normalizado". Para um vocabulário de 50 fonemas, o decoder produz 50 logits a cada passo — um por fonema candidato. Logits podem ser qualquer número real (−∞, +∞).

```
logits = [-1.2,  3.4,  0.8,  -0.3, ...]   # 50 valores, um por fonema
           /k/   /s/   /ʃ/    /e/
```

**Softmax**
Função que transforma logits em probabilidades: aplica exponencial a cada logit e divide pela soma, garantindo que todos os valores fiquem em (0, 1) e somem 1. Nunca produz exatamente 0 ou 1 — isso é importante para que `log(p)` não seja infinito.

```
softmax([−1.2, 3.4, 0.8]) = [0.02, 0.88, 0.10]
                              ↑     ↑     ↑
                              soma = 1.00
```

**Argmax**
A operação de pegar o índice do maior valor em um vetor. Aplicado sobre as probabilidades (ou logits, o resultado é o mesmo), retorna o fonema que o modelo escolheu como predição. É uma operação não-diferenciável — não tem gradiente, por isso o gradiente é calculado sobre os logits/probabilidades antes do argmax.

```
argmax([0.02, 0.88, 0.10]) = 1   → fonema de índice 1 = /s/
```

**p_correct** (`p(y)`, probabilidade do fonema correto) e **p_pred** (`p(ŷ)`, probabilidade do fonema predito)

Esses dois valores vêm do **mesmo vetor softmax** — a saída do decoder para um passo de predição. O que os diferencia é a *posição do vetor* que cada loss function consulta.

#### O vetor softmax

O decoder produz um vetor de logits com uma entrada por fonema do vocabulário (~60 fonemas no FG2P). O softmax converte esses logits em probabilidades que somam 1.0:

```
Vocab:    /a/   /s/   /k/   /ɾ/   /ɛ/  ...  (simplificado: 5 fonemas)
Índice:    0     1     2     3     4

Logits:  [0.1,  1.2,  2.8,  0.5,  0.3]
         ─────────── softmax ───────────
Probs:   [0.05, 0.14, 0.69, 0.07, 0.06]   ← soma = 1.0
          /a/   /s/   /k/   /ɾ/   /ɛ/
               ──┘     └────────
          target correto      predito (argmax)
               (/s/)                  (/k/)
```

Nesse exemplo, o modelo deveria ter dito `/s/` (índice 1), mas escolheu `/k/` (índice 2, maior logit):

```
p_correct = probs[índice do target]  = probs[1] = 0.14   ← usado pela CE
p_pred    = probs[argmax(logits)]    = probs[2] = 0.69   ← usado pela DA
```

#### Por que cada loss usa um valor diferente?

**Cross-Entropy usa `p_correct`** porque sua pergunta é:
*"Qual a probabilidade que o modelo atribuiu à resposta certa?"*
A CE quer forçar o modelo a concentrar confiança no fonema correto, independente do que ele escolheu. Se `p_correct = 0.14`, a loss é `-log(0.14) = 1.97` — alta, sinalizando que o modelo está inseguro na resposta certa.

```
L_CE = -log(p_correct) = -log(0.14) = 1.97
```

**DA Loss usa `p_pred`** porque sua pergunta é:
*"Quão confiante o modelo estava no fonema errado que escolheu?"*
A DA quer escalar a penalidade de distância pela confiança na predição errada. Se o modelo escolheu `/k/` com 69% de confiança (não por acaso, mas com convicção), a penalidade de distância deve ser maior do que se tivesse escolhido com 15%.

```
L_DA = λ × d(/k/, /s/) × p_pred = 0.20 × d × 0.69   (penalidade proporcional à convicção)
```

#### Caso de acerto — quando p_correct = p_pred

Quando o modelo acerta, argmax = target. Ambas as variáveis apontam para o mesmo índice:

```
Logits:  [0.1,  3.5,  0.8,  0.5,  0.3]
Probs:   [0.03, 0.85, 0.06, 0.04, 0.03]
                 ↑
          argmax = target = /s/

p_correct = p_pred = 0.85   ← mesma posição do vetor
```

Nesse caso:
- CE = `-log(0.85) = 0.16` — baixa (modelo está confiante na resposta certa)
- DA = `λ × d(/s/, /s/) × 0.85 = 0.20 × 0 × 0.85 = 0` — zero (distância = 0 quando acerta)

Ou seja: **quando o modelo acerta, DA não contribui nada**. Ela só entra em cena no erro.

#### Resumo visual

```
                   Vetor softmax
           ┌──────────────────────────────────┐
           │  /a/  /s/  /k/  /ɾ/  /ɛ/  ...  │
           │ 0.05 0.14 0.69 0.07 0.06  ...  │
           └──────────────────────────────────┘
                   ↑           ↑
              p_correct     p_pred
           (índice target) (índice argmax)

           Usado por CE    Usado por DA
           "confiança na   "confiança na
            resposta certa" resposta errada"
```

> Nota: p_correct e p_pred são a *mesma variável* apenas quando o modelo acerta (argmax = target). Em erros, são valores de posições **diferentes** do mesmo vetor softmax — e é exatamente essa diferença que torna as duas losses complementares.

**Gradiente**
Vetor de derivadas parciais da loss em relação a cada parâmetro do modelo. Durante o treinamento, os pesos são ajustados na direção oposta ao gradiente (descida do gradiente) para minimizar a loss. O gradiente "flui para trás" pela rede (backpropagation), do último layer até o primeiro.

**Backpropagation**
Algoritmo que calcula os gradientes de todos os parâmetros da rede aplicando a regra da cadeia do cálculo diferencial, da última camada (loss) até a primeira (embedding). Permite que o sinal da DA Loss, calculado na saída do decoder, influencie os pesos do BiLSTM encoder e os pesos de atenção.

**Embedding**
Vetor denso de números reais que representa um símbolo discreto (letra ou fonema) num espaço contínuo. Em vez de tratar "a" como o número 1 e "b" como 2 (o que implica ordem arbitrária), embeddings aprendem representações onde símbolos similares ficam próximos no espaço vetorial. No FG2P, grafemas de entrada são convertidos em vetores de 192 dimensões antes de entrar no BiLSTM.

**Epoch (Época)**
Uma passagem completa por todo o conjunto de treinamento. Em cada epoch, o modelo vê todos os pares (palavra, fonemas), calcula a loss, e atualiza os pesos. O FG2P treina por até 120 epochs com early stopping.

**Batch**
Subconjunto de exemplos de treinamento processados juntos antes de atualizar os pesos. O FG2P usa batch_size=64 — a cada passo, 64 palavras são processadas, as losses são somadas, e o gradiente é calculado sobre essa média.

**Token**
Unidade mínima do vocabulário. Na saída do FG2P, cada fonema IPA é um token (`k`, `ɔ`, `ˈ`, `.`, etc.), incluindo tokens especiais como `<PAD>` (preenchimento) e `<EOS>` (fim de sequência).

**Padding** (`<PAD>`)
Token especial usado para igualar o comprimento de todas as sequências num batch. Palavras mais curtas recebem `<PAD>` no final. A loss é calculada com `ignore_index=0` para ignorar esses tokens e não penalizar predições sobre posições de padding.

---

### Termos de Funções de Custo

**Loss / Função de Custo**
Número que mede o "erro" do modelo numa predição. O objetivo do treinamento é minimizar esse número. A loss deve ser diferenciável para que o gradiente possa ser calculado.

**Cross-Entropy (CE)**
A loss padrão para classificação multi-classe: `L_CE = -log(p_correct)`. Penaliza o modelo por não ter atribuído probabilidade alta ao fonema correto. Intervalo matemático: [0, +∞). Intervalo computacional com log-sum-exp: [0, ~16].

**NLL (Negative Log-Likelihood)**
Mesmo conceito que Cross-Entropy no contexto de classificação. PyTorch separa em duas funções: `LogSoftmax` + `NLLLoss`, ou diretamente `CrossEntropyLoss` (equivalente, mais estável numericamente).

**log-sum-exp trick**
Técnica de estabilidade numérica usada pelo PyTorch internamente: em vez de calcular `log(sum(exp(logits)))` diretamente (que causa overflow para logits grandes), subtrai o máximo antes: `log(sum(exp(logits - max))) + max`. Resultado idêntico matematicamente, mas sem risco de overflow/underflow em float32.

**Overflow / Underflow**
Overflow: número muito grande para ser representado em float32 (>~3.4×10³⁸) → vira `inf`. Underflow: número muito pequeno (< ~1.2×10⁻³⁸) → vira `0`. Ambos corrompem o gradiente. O log-sum-exp trick previne overflow; softmax em logits normais previne underflow.

**FP32 (float32)**
Representação de número em ponto flutuante de 32 bits, o padrão para treinamento de redes neurais em GPU. Tem ~7 dígitos de precisão decimal. Valores além de ±3.4×10³⁸ causam overflow.

**Hiperparâmetro (λ, learning rate, etc.)**
Parâmetro que controla o processo de treinamento mas **não é aprendido** pela rede — deve ser definido manualmente ou por busca (ablation study). λ=0.20 é um hiperparâmetro da DA Loss; learning_rate=0.001 é outro.

**Label Smoothing**
Técnica de regularização que substitui o alvo hard (one-hot: correto=1, todos outros=0) por um alvo suave (correto=1−ε, outros distribuem ε/(V−1)). Evita que o modelo fique excessivamente confiante. DA Loss é análoga: ela distribui "custo" para fonemas vizinhos proporcionalmente à distância fonológica.

**Annealing (de λ)**
Estratégia de reduzir gradualmente um hiperparâmetro ao longo do treinamento, inspirada no recozimento (annealing) em metalurgia. λ annealing começaria com λ alto (sinal fonológico forte) nas primeiras épocas, reduzindo para λ=0.20 ou menos no final.

**Threshold (limiar)**
Valor mínimo abaixo do qual uma operação é ignorada ou tratada como zero. Na Sugestão B (§6.3), `threshold δ=0.25` faz com que distâncias menores que 0.25 sejam ignoradas pela DA Loss — focando o sinal apenas em erros fonologicamente significativos.

---

### Termos de Arquitetura

**BiLSTM (Bidirectional Long Short-Term Memory)**
Rede recorrente que processa a sequência de grafemas em **duas direções** simultaneamente: esquerda→direita e direita→esquerda. O estado oculto `h_i` de cada posição combina contexto do passado e do futuro. Importante para G2P porque a pronúncia de "c" em "cena" depende do "e" que vem depois.

**Encoder**
Parte da rede que lê a sequência de entrada (grafemas) e produz uma representação interna. No FG2P, é o BiLSTM: lê "c-a-s-a" e produz `h_1, h_2, h_3, h_4` — um vetor de 384 dimensões por grafema.

**Decoder**
Parte da rede que gera a sequência de saída (fonemas) passo a passo, usando a representação do encoder. Produz um fonema por vez, condicionado no fonema anterior e no contexto de atenção.

**Atenção de Bahdanau**
Mecanismo que permite ao decoder focar em partes específicas da entrada a cada passo de decodificação. Calcula um vetor de pesos `α` (soma=1) sobre os estados `h_i` do encoder e produz um vetor de contexto `c_t = Σ αᵢ · hᵢ`. Permite, por exemplo, que ao gerar o fonema para "sc" em "biscoito", o decoder foque simultaneamente em "s" e "c".

**PanPhon**
Biblioteca de features articulatórias fonéticas: representa cada fonema IPA como um vetor de 24 valores binários/ternários codificando propriedades físicas da produção do som (vozeamento, nasalidade, ponto de articulação, modo de articulação, altura da língua, etc.). Usada para calcular distâncias fonológicas na DA Loss.

**vocab_size**
Tamanho do vocabulário de saída — o número de fonemas IPA distintos + tokens especiais que o modelo pode produzir. No FG2P: ~52–54 tokens. Define a dimensão do vetor de logits produzido pelo decoder a cada passo.
