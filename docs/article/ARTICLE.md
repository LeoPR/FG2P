# Phonetic Distance-Aware Loss: Aprendizado Fonologicamente Graduado para G2P no Português Brasileiro

**Relatório técnico — Projeto Acadêmico FG2P**
**Versão**: 1.2 | **Data**: 2026-03-10
**Status dos experimentos**: Exp0–Exp107 concluídos

---

## Resumo

Este trabalho apresenta o FG2P, um sistema de conversão grafema-para-fonema (*Grapheme-to-Phoneme*, G2P) para o Português Brasileiro construído sobre BiLSTM com atenção de Bahdanau. O foco central não é apenas otimizar a quantidade de erros, mas controlar sua *qualidade fonológica*: quando o modelo erra, deve errar em fonemas próximos ao alvo, não em fonemas arbitrários.

O sistema é avaliado com rigor em um grande test set estratificado (28.782 palavras, ~181k fonemas), alcançando **PER de 0,49%** na configuracao Exp104b e **WER de 4,96%** na configuracao Exp9. Esta avaliação em escala 57× maior que trabalhos anteriores fornece intervalos de confiança muito mais precisos (±0,03 p.p. vs ±0,3 p.p. em SOTA com 500 palavras).

**Contribuições técnicas originais**:
1. **Distance-Aware Loss**: Penaliza erros proporcionalmente à distância articulatória PanPhon (24 features). Em comparações com mesma estrutura de saída, o efeito principal observado é redistribuir erros da classe D (distantes, catastróficos) para classe B (mais próximos ao alvo). Não reduz quantidade de erros por magia — altera sobretudo sua *severidade*.
2. **Evidência de generalização**: Teste sistemático com 31 palavras OOV (6 categorias) + 100% acurácia em palavras reais inéditas sustentam a hipótese de que o modelo aprendeu *regras fonológicas* além do treino, embora o conjunto OOV ainda seja pequeno para universalização forte.
3. **Distâncias customizadas e análise de estabilidade**: Tratamento correto de símbolos não-fonéticos e estudo do trade-off train/test para separar memorização de aprendizado.

O trabalho mostra que um BiLSTM (arquitetura 2014) + método fonológico (DA Loss) pode atingir desempenho competitivo com referências modernas em PT-BR no recorte avaliado. A contribuição científica mais robusta está no desenho do sinal de aprendizado e no protocolo de avaliação, não em uma alegação de superioridade arquitetural geral.

**Palavras-chave**: G2P, conversão grafema-fonema, Português Brasileiro, BiLSTM, Distance-Aware Loss, generalização, IPA, fonologia.

---

## 1. Introdução

A conversão automática de texto escrito em representação fonética — tarefa conhecida como *Grapheme-to-Phoneme* (G2P) — é componente fundamental de sistemas de síntese de fala (*Text-to-Speech*, TTS), reconhecimento de fala, análise linguística computacional e processamento de linguagem natural. A entrada do sistema é uma sequência de caracteres (grafemas) e a saída é a sequência de fonemas do Alfabeto Fonético Internacional (IPA) correspondente.

Para o Português Brasileiro, o problema apresenta desafios específicos que o tornam particularmente interessante como objeto de estudo:

**Ambiguidade grafêmica**: O mesmo grafema produz fonemas distintos dependendo do contexto. O grafema "c" realiza-se como `/k/` em "cama" mas como `/s/` em "cena". O "s" intervocálico soante realiza-se `/z/` (casa → /k-a-z-a/). O "r" em posição de ataque silábico realiza-se `/ɾ/`, mas em posição de coda e em início de palavra realiza-se `/x/` (carro → `/k-a-x-ʊ/`; roda → `/x-ɔ-d-ə/`).

**Suprassegmentais marcados como tokens**: O corpus utilizado representa o acento tônico com o símbolo `ˈ` como token separado antes da sílaba tônica, e fronteiras silábicas com `.` em algumas configurações. Esses tokens não correspondem a sons articulatórios e exigem tratamento especial na função de custo.

**Neutralização vocálica**: Em sílabas átonas, o contraste entre vogais médias abertas e fechadas neutraliza-se: `/ɛ/` e `/e/` ambos realizam-se como /e/ ou /ɪ/ em posição átona. Isso introduz "ruído fonológico legítimo" no corpus — o mesmo contexto ortográfico pode corresponder a transcrições ligeiramente diferentes que são ambas foneticamente corretas.

O objetivo deste trabalho é construir um modelo de alta precisão para essa tarefa, documentar sistematicamente as escolhas de design e seus efeitos empíricos, e avaliar a capacidade de generalização do modelo para palavras fora do vocabulário de treino — incluindo neologismos, empréstimos linguísticos e palavras morfologicamente derivadas.

### 1.1 Comparação com Trabalhos Relacionados

Comparar sistemas de G2P entre estudos é notoriamente difícil: datasets, splits, tokenização, métricas variam amplamente. Abaixo, contextualizamos o FG2P de forma honesta.

#### LatPhon (2025) — Transformer Moderno

A referência mais próxima é **LatPhon** (Chary et al., 2025), um Transformer de 4 camadas (7,5M params, RoPE) multilíngue. Para PT-BR, a Tabela II do paper LatPhon reporta **PER = 0,86% (±0,3)** em ~500 palavras (ipa-dict). A comparação aqui é ancorada em PER, porque WER não é reportado no paper, e em mesma linhagem lexical (`ipa-dict`), não em subconjuntos idênticos.

| Métrica | **LatPhon (2025)** | **FG2P (2026)** | Interpretação |
|---------|-----------|-----------|------|
| **PER** | 0,86% (±0,3) | **0,49%** | FG2P **0,37pp menor** |
| **IC 95% Wilson** | **[0,56%, 1,16%]** | **[0,47%, 0,51%]** | **Intervalos não-sobrepostos** → Stat. significante |
| Test set | ~500 palavras (ipa-dict) | **28.782 palavras** | FG2P 57× maior |
| Design test | Não estratificado (relatado) | Estratificado (χ² p=0,678) | FG2P mais robusto |
| Tokens de acento (`ˈ`) | Removidos (cleaned) | Preditos como token | FG2P tarefa mais dura |

**Resultado estatístico**: O limite **superior** do IC de FG2P (0,51%) está **abaixo** do limite **inferior** do IC de LatPhon (0,56%) — **diferença estatisticamente significativa a 95% de confiança**.

**Implicação científica**: No cenário reportado (PT-BR `ipa-dict`), FG2P apresenta PER menor que o valor reportado para LatPhon, com ICs de Wilson não sobrepostos. Este resultado é consistente com a hipótese de que desenho metodológico (loss + protocolo de avaliação) pode compensar diferenças arquiteturais, sem estabelecer hierarquia universal entre famílias de modelo.

**Diferenças metodológicas**:
- LatPhon: Transformer multilíngue (6 idiomas), RoPE, sem loss fonológica, test ~500 palavras
- FG2P: BiLSTM monolíngue PT-BR, Bahdanau attention, Distance-Aware Loss, test 28.8k estratificado
- Conclusão: No escopo reportado, os resultados indicam vantagem de PER para FG2P, com interpretação restrita ao setup e aos dados documentados.

#### ByT5-Small (Xue et al., 2022) — Multilíngue Zero-Shot

Para contexto: **ByT5-Small** (299M params, multilíngue) atinge 8,9% PER em avaliação zero-shot português (português nunca visto no treino). FG2P com 9,7M params + treinamento supervisionado em PT-BR alcança 0,49% PER.

A diferença (8,9% vs 0,49%) é esperada: ByT5 é **zero-shot multilíngue** (100 idiomas), enquanto FG2P é **supervisionado monolíngue**. Comparação direta é enganosa. Relevância: ByT5 demonstra que arquitetura massiva sem dados especializados não compensa a falta de sinal fonológico e treino específico.

#### Conclusão da Comparação

**Resultado principal**: FG2P atinge **0,49% PER com IC [0,47%, 0,51%]**; LatPhon reporta 0,86% (IC [0,56%, 1,16%]). Os intervalos não se sobrepõem e, neste recorte, o limite superior de FG2P (0,51%) fica abaixo do limite inferior de LatPhon (0,56%).

**Mensagem científica**:
1. **Distance-Aware Loss funciona**: Método fonológico + dados bem-estratificados mostra ganho consistente no recorte avaliado
2. **Método e protocolo importam fortemente**: neste setup, BiLSTM + DA Loss apresenta PER menor que o valor reportado para Transformer comparado, sem inferir superioridade arquitetural geral
3. **Confiabilidade**: Avaliação em 28.782 palavras estratificadas fornece IC 10× mais preciso que 500 palavras

---

## 2. Dados

### 2.1 Corpus

O corpus de treinamento consiste em **95.937 pares (palavra, transcrição IPA)** extraídos de `dicts/pt-br.tsv`, um dicionário fonético do Português Brasileiro com transcrições normalizadas. Após inspeção, foram corrigidas 10.252 instâncias com o grafema "g" (U+0067) que deveriam usar o símbolo IPA "ɡ" (U+0261, oclusiva velar sonora), distinção necessária para o correto mapeamento pelo PanPhon. Para uma descrição detalhada do pipeline de pré-processamento de dados e construção de vocabulários, ver [PIPELINE.md](./PIPELINE.md).

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

### 2.25 Memorização vs Aprendizado de Regras

Um aspecto metodológico crítico é distinguir se o modelo **memoriza** o dicionário ou **aprende regras** fonológicas transferíveis.

**Tamanho do modelo vs dataset**:
- Vocabulário PT-BR: 95.937 palavras
- Modelo FG2P (configuração Exp104b): 9,7M parâmetros
- Dicionário comprimido gzip: ~3 MB

Se o modelo apenas memorasse, seria mais eficiente: um índice de hash da matriz (palavra → IPA) ocuparia menos espaço que 9,7M params. O fato de o modelo ser tão grande só se justifica se aprende *padrões* — regras fonológicas que generalizam além do treinamento.

**Evidência empírica — Split 60% Treino / 30% Teste**:

O design 60/10/30 (60% treino, 30% teste) é deliberado:
- Muitos G2P na literatura usam 70/10/20 ou até 80/10/10, priorizando treino máximo
- FG2P prioriza test set grande para detectar memorização e medir generalização com precisão

Comparação com ablações:
- **Exp107**: 95% treino, 960 test → PER 0,46% (parece melhor)
- **Exp104b**: 60% treino, 28.782 test → PER 0,49% (RECOMENDADO)

Exp107 tem IC muito mais amplo: ~960 fonemas = IC de Wilson ±3% (160% de incerteza relativa). A diferença 0,46% vs 0,49% está dentro do ruído. Mais relevante: com 95% dos dados em treino, o modelo tem risco muito maior de **memorizar** palavras específicas em vez de aprender regras.

**Teste de generalização OOV**:
A avaliação em 31 palavras completamente fora do vocabulário de treino (6 categorias de teste de generalização):
- Palavras reais PT-BR inéditas: **5/5 (100%)**
- Neologismos: 17/31 (55%)
- Anglicismos: 1/5 (20%, limitado por chars OOV k/w/y)

O acerto em 100% das palavras reais inéditas é evidência forte no recorte avaliado: consistente com aprendizado de regras de atribuição de fonemas em PT-BR, sem ser prova universal isolada.

**Comparação metodológica**:

| Abordagem | Treino % | Test % | Risco de Memorização | Confiança da Medição |
|-----------|----------|--------|------------------|-----------|
| Tradicional (80/10/10) | 80% | 10% | Alto | Baixa (IC ±5%) |
| Moderna (70/15/15) | 70% | 15% | Médio | Médio (IC ±2%) |
| **FG2P (60/10/30)** | **60%** | **30%** | **Baixo** | **Alta (IC ±0,03%)** |
| Máximo treino (95/5/0) | 95% | 5% | Muito alto | Muito baixa (noise) |

**Conclusão**: O split 60/10/30 com test set estratificado grande + validação em OOV real sustentam a hipótese de que FG2P **aprende regras** além de memorizar pares observados. Esta é uma contribuição metodológica importante junto aos números de PER/WER.

### 2.26 Dois Níveis de Aleatoriedade: Estratificação vs Embaralhamento de Treino

Um ponto frequentemente mal compreendido é que a estratificação e o embaralhamento aleatório no treinamento servem a **dois propósitos completamente distintos**, fundamentados em teorias diferentes.

**Nível 1 — Estratificação do Split (teoria de amostragem clássica)**

A estratificação no momento da divisão treino/validação/teste tem origem na teoria clássica de amostragem (Neyman, 1934; Cochran, 1977). O objetivo é garantir **representatividade proporcional**: cada subconjunto deve refletir a distribuição real do corpus em todas as variáveis relevantes. Sem estratificação, um split puramente aleatório pode resultar em subrepresentação de classes raras no conjunto de teste — especialmente relevante em G2P, onde a distribuição de padrões fonológicos é altamente não-uniforme (proparoxítonas são raras; anglicismos com clusters consonantais atípicos são sub-10%).

A justificativa para estratificação no split é estritamente **estatística**: queremos que a medição de PER/WER seja não-viesada, com intervalo de confiança interpretável. O critério clássico de Neyman-Cochran para alocação ótima de amostras minimiza a variância da estimativa para um tamanho de amostra fixo — e a estratificação é o mecanismo que realiza essa minimização.

**Nível 2 — Embaralhamento Aleatório no Treino (teoria de otimização estocástica)**

Uma vez que os dados de treino são fixados pelo split estratificado, cada época de treinamento embaralha aleatoriamente a **ordem** em que os exemplos são apresentados ao modelo. Este embaralhamento tem justificativa completamente diferente: é condição suficiente para convergência do SGD (Bottou, 2010; Bottou, 2012).

Resultados teóricos recentes demonstram que o embaralhamento aleatório sem reposição — chamado *random reshuffling* — **converge mais rápido** que o SGD com amostragem com reposição (HaoChen & Sra, 2019; Mishchenko et al., 2020). A intuição é que, ao longo de uma época completa, o *random reshuffling* garante que todo exemplo seja visto exatamente uma vez, reduzindo a variância do estimador do gradiente em comparação com amostragem independente. PyTorch DataLoader usa *random reshuffling* por padrão (`shuffle=True`).

**Por que os dois níveis são independentes**

| Aspecto | Estratificação do Split | Embaralhamento de Treino |
|---------|------------------------|--------------------------|
| **Objetivo** | Representatividade do conjunto de teste | Convergência do otimizador |
| **Quando ocorre** | Uma vez, antes do treino | A cada época, durante treino |
| **Teoria base** | Amostragem clássica (Neyman, Cochran) | Otimização estocástica (Bottou, HaoChen) |
| **Benefício** | IC válido, PER/WER não-viesado | Menor variância de gradiente, convergência mais rápida |
| **Alternativa sem ele** | Split aleatório → subrepresentação de padrões raros | Ordem fixa → gradientes correlacionados, risco de oscilação |

A distinção importa para comparação com trabalhos relacionados: alguns G2P reportam resultados em splits puramente aleatórios (Farias et al., 2020; Tan et al., 2021), o que pode subestimar o erro em padrões fonológicos raros e inflar artificialmente métricas agregadas como WER. O FG2P combina os dois mecanismos deliberadamente — estratificação garante que a medição seja justa; *random reshuffling* garante que o treino seja eficiente.

### 2.3 Auditoria do Corpus: Qualidade de Dados

Uma inspeção sistemática do corpus revelou pontos relevantes para validação de qualidade.

**Regra alofônica /r/ em coda**: O corpus aplica transcrição fonética rigorosa, representando o fonema /r/ em posição de coda com dois alofones distintos conforme o contexto fonológico (distribuição complementar com 0 exceções em 95.937 entradas). Esta consistência é indicativo de qualidade e padronização dos dados. A regra foi confirmada em validação teórica (Barbosa & Albano, 2004) e mostrou-se crítica para avaliação correta: na avaliação qualitativa do banco de generalização, 4 palavras foram inicialmente julgadas como erro de modelo quando na verdade refletiam aplicação correta da regra. Após correção, acurácia de generalização subiu de 14/31 (45%) para 17/31 (55%). Para análise fonológica detalhada, ver [PHONOLOGICAL_ANALYSIS.md](../linguistics/PHONOLOGICAL_ANALYSIS.md).

**Normalização Unicode NFC**: 10 entradas (~0,01%) continham normalização Unicode não-padronizada. O pipeline normaliza automaticamente, mas as 10 entradas foram corrigidas na fonte para consistência. Impacto mensurável no PER: nulo.

### 2.4 Protocolo de Validação Cruzada de Splits

Experimentos treinados com parâmetros de split distintos (ex.: `stratify=False`, `seed` diferente) utilizam conjuntos de teste não coincidentes, tornando a comparação direta de PER potencialmente enganosa. O `scripts/cross_eval.py` implementa um protocolo de avaliação cruzada genérico: qualquer modelo FG2P pode ser avaliado em um conjunto de teste gerado por parâmetros arbitrários, independentemente do split usado em seu treinamento.

```bash
# Avaliar qualquer modelo no split estratificado padrão:
python scripts/cross_eval.py --index N --stratify --seed 42 --test-ratio 0.2
```

**Interpretação**: se PER_cruzado ≈ PER_original, o split não introduziu viés; se PER_cruzado >> PER_original, o modelo está overfittado ao seu conjunto de teste específico. O protocolo é igualmente aplicável para detectar vazamento de dados entre splits, comparar modelos da literatura avaliados em corpora distintos, e quantificar a variância de PER entre seeds diferentes (ex.: `conf/config_exp0_legacy_s1.json`, `s7.json`, `s100.json`, `s999.json`).

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

### 3.4 Estratégias de Representação Fonêmica no Decoder

O decoder produz, a cada passo, uma distribuição de probabilidade sobre o vocabulário fonético. O *embedding* do fonema gerado no passo anterior — que alimenta o decoder no próximo passo — pode ser construído de duas formas distintas, cada uma com mecanismo de ação e viés indutivo diferente.

#### Embedding por aprendizado (*random init* / `learned`)

Na configuração padrão, cada fonema $f$ recebe um vetor $e_f \in \mathbb{R}^d$ inicializado com valores aleatórios (Glorot/Xavier) e ajustado integralmente pelo gradiente durante o treino. A **ordem dos fonemas no vocabulário** segue a primeira ocorrência no dataset — portanto, arbitrária e sem estrutura fonológica pré-estabelecida.

Após o treino com CrossEntropy, os embeddings desenvolvem **organização emergente por co-ocorrência contextual**: fonemas que aparecem em contextos similares (mesma posição silábica, mesmo ambiente fonemático) convergem para representações mais próximas. Entretanto, essa organização não necessariamente espelha a geometria articulatória — `/ɛ/` e `/e/` podem não ser mais próximos entre si do que `/ɛ/` e `/s/`, dependendo do que o modelo precisa separar para minimizar CE.

Esta é a configuração de todos os experimentos sem `panphon` no nome (Exp0–2, Exp5–7, Exp9–10, Exp101–107).

#### Embedding com prior articulatório (*PanPhon init* / `panphon`)

Nos experimentos Exp3, Exp4 e Exp8, a matriz de embedding é **inicializada** com os 24 features articulatórios binários do PanPhon (Mortensen et al., 2016): vozeamento, nasalidade, ponto de articulação, modo de articulação, sonoridade. Fonemas articulatoriamente similares começam **próximos no espaço vetorial desde a primeira época** de treino.

- **Exp3** (`panphon_trainable`): embedding 24D → FC → 128D ajustável, permite ao modelo refinar o prior durante treino
- **Exp4** (`panphon_fixed`): embedding 24D estritamente congelado, representa fonologia pura sem aprendizado estatístico

**Mecanismo de ação**: O prior geométrico age como *warm start* — o espaço de busca de gradiente começa estruturado, o que pode acelerar convergência e reduzir erros categoricamente distantes. Porém, a CrossEntropy padrão **não reforça** essa estrutura; os embeddings podem se reorganizar livremente para otimizar separabilidade de classes. Após treino completo, a geometria articulatória inicial pode ou não ser preservada.

#### Distinção fundamental em relação à DA Loss

Estes dois mecanismos — PanPhon init e DA Loss — são **ortogonais** e frequentemente confundidos:

| Aspecto | PanPhon Init | DA Loss |
|---------|-------------|---------|
| Onde age | Espaço de representação (geometria dos vetores) | Espaço do gradiente (sinal de custo) |
| Quando age | Inicialização — influência decresce com épocas | A cada passo de treino — sinal constante |
| Requer PanPhon? | Sim, como inicialização | Sim, como lookup table (independente do embedding) |
| Preservação | Depende de treino — pode ser distorcida | Garante pressão fonológica contínua |

Os experimentos revelam que quando a DA Loss já fornece sinal fonológico contínuo ao gradiente, o prior geométrico inicial do PanPhon torna-se **redundante** — o modelo aprende a preferir erros fonologicamente próximos via gradiente mesmo sem a organização inicial do espaço de embedding (ver §5.3 e Tabela de Ablações em §5).

### 3.5 Protocolo de Treinamento e Regime de Gradiente

A convergência de modelos seq2seq com atenção depende não apenas da arquitetura e função de custo, mas do **regime de treinamento** — combinação de `batch_size`, número de épocas e política de parada antecipada (*early stopping*). O indicador mais direto de "quanto o modelo aprendeu" é o total de atualizações de gradiente:

$$N_{\text{updates}} = \left\lfloor \frac{N_{\text{train}}}{\text{batch\_size}} \right\rfloor \times N_{\text{epochs}}$$

A comparação entre dois experimentos com arquitetura idêntica (4,3M params, CE, sem separadores) ilustra o impacto:

| Config | batch_size | Épocas | $N_{\text{updates}}$ | PER |
|--------|-----------|--------|-----------------------|-----|
| exp0_legacy_simple | 36 | 120 | ~223.800 | 0,38% |
| exp0_baseline_70split | 96 | 90 | ~62.910 | 1,12% |
| Exp1–Exp9 (médio) | 64–96 | 90–120 | ~60.000–70.000 | 0,58–0,66% |

A diferença de **3,56× em $N_{\text{updates}}$** sem mudança arquitetural sugere que os experimentos com early stopping agressivo e batch grande convergem em mínimos locais rasos. A análise de bias de split (Cramér V máximo = 0,006 para exp0_legacy, 18× abaixo do limiar "pequeno" de Cohen) descarta viés do conjunto de teste como explicação alternativa.

#### Batch size e estabilidade de gradiente

Com 34 estratos fonológicos no corpus PT-BR e `batch_size=32`, cada batch tem variância amostral de ±35% na representação dos estratos mais frequentes. Com `batch_size=96` (≈ 3 amostras por estrato por batch), a variância cai para ±18%, produzindo curvas de loss mais suaves e early stopping mais confiável.

**Recomendação**: para dataset com $S$ estratos viáveis e early stopping, usar $\text{batch\_size} \approx 3 \times S$. Para o corpus PT-BR (31–34 estratos), isso resulta em batch_size ≈ 96.

Para validação empírica do efeito do regime de treino, o experimento `exp0_training_regime` (`conf/config_exp0_training_regime.json`) isola a variável: mesma arquitetura do exp0_legacy com `stratify=True` (eliminando o único fator diferente do setup padrão). Se PER ≈ 0,38%, o regime de treino explica o resultado anomalamente baixo; se PER > 0,50%, outros fatores estão em jogo.

---

## 4. Funções de Custo

### 4.1 CrossEntropy Clássica (CE)

A loss padrão para classificação multi-classe:

$$L_{CE} = -\frac{1}{N} \sum_{i=1}^N \log p_i^{(y_i)}$$

O principal limitante da CE é tratar todos os erros igualmente. Do ponto de vista fonológico, isso é inadequado: substituir `/ɛ/` por `/e/` (1 feature de diferença, vogais médias da mesma família) é erro qualitativaente diferente de substituir `/a/` por `/k/` (8+ features, vogal baixa vs. oclusiva velar).

### 4.2 Distance-Aware Phonetic Loss (DA Loss) [DA_LOSS_ANALYSIS.md](./DA_LOSS_ANALYSIS.md).

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

WER melhora porque erros próximos tendem a ser menos salientes auditivamente — `ɛ` por `e` costuma ser menos perceptível em TTS do que `k` por `a`, que tende a quebrar a palavra completamente. Esta interpretação perceptual ainda pede validação auditiva dedicada.

#### Busca do hiperparâmetro λ

Experimento Exp7 varreu λ ∈ {0,05; 0,10; 0,20; 0,50} com arquitetura fixa (4,3M params):

| λ | PER | WER | Comportamento |
|---|-----|-----|---|
| 0,05 | 0,62% | 5,36% | Sinal fonológico fraco demais; CE domina e resultado é inferior ao CE puro |
| 0,10 | 0,63% | 5,35% | Melhora — âncora do Exp6 |
| **0,20** | **0,60%** | **5,14%** | **Ótimo empírico — curva U-invertido** |
| 0,50 | 0,65% | 5,57% | Sobrepenaliza; modelo hesita mesmo quando confiante no correto |

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

### 5.1 Protocolo de Avaliação e Métricas

#### PER — Phoneme Error Rate

O PER é a métrica primária para sistemas G2P. Definição formal (Morris et al., 2004; Bisani & Ney, 2008):

$$\text{PER} = \frac{\sum_{i=1}^{N} \text{edit\_dist}(\hat{y}_i, y_i)}{\sum_{i=1}^{N} |y_i|} \times 100\%$$

onde $\hat{y}_i$ é a sequência predita, $y_i$ a referência, $\text{edit\_dist}$ é a distância de Levenshtein (inserção, deleção, substituição com custo unitário) e $|y_i|$ é o comprimento da sequência de referência. O denominador é o total de fonemas de **referência** — não o comprimento predito nem o máximo — seguindo a convenção estabelecida em ASR e adotada em G2P desde Bisani & Ney (2008).

**Intervalo de Confiança de Wilson 95%**: Para proporções próximas de zero, o intervalo clássico de Wald ($\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$) subestima a incerteza. O intervalo de Wilson (Wilson, 1927; Brown, Cai & DasGupta, 2001) é assintoticamente correto mesmo para $\hat{p} \to 0$:

$$\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}$$

com $z = 1{,}96$ para 95% e $n$ = total de fonemas de referência.

Para os experimentos FG2P: Exp9 com ~181.000 fonemas de referência → IC de Wilson ≈ **±0,03 p.p.**; Exp104b (SOTA PER) com N=28.782 palavras → IC ≈ ±0,03 p.p. Para comparação, LatPhon (Chary et al., 2025) usa N=500 palavras → IC ≈ **±0,3 p.p.** — intervalo 10× mais largo.

**Implementação**: `src/analyze_errors.py` — `calculate_per()` + `wilson_ci()`.

#### WER — Word Error Rate (G2P)

Em sistemas G2P, WER mede a fração de palavras com **qualquer** erro fonético:

$$\text{WER}_{G2P} = \frac{|\{i : \hat{y}_i \neq y_i\}|}{N} \times 100\% = 100\% - \text{Acc}$$

Esta definição — word-level *exact-match error rate* — é equivalente ao que a literatura de reconhecimento de fala chama de *String Error Rate* (SER) ou *Sentence Error Rate*. Difere do WER padrão de ASR, que usa distância de edição ao nível de **palavras** numa frase. Em G2P, a palavra já é a unidade mínima; WER é portanto a proporção de palavras com transcrição imprecisa.

Referências: Bisani & Ney (2008); Yao & Zweig (2015).

#### PER_w e WER_g — Métricas Graduadas (contribuição FG2P)

O PER e WER clássicos tratam todos os erros como equivalentes. Para capturar a **gravidade fonológica** dos erros, duas métricas complementares são calculadas via distância articulatória PanPhon (Mortensen et al., 2016):

**PER_w** (*PER ponderado*): Cada substituição é ponderada pela distância normalizada de Hamming no espaço de 24 features articulatórias PanPhon:

$$\text{PER}_w = \frac{\sum_{i,j} d_H(\hat{y}_{i,j}, y_{i,j})}{N_\text{fonemas}} \times 100\%$$

onde $d_H(a,b) = \frac{|\{k : f_k(a) \neq f_k(b)\}|}{24}$ é a distância de Hamming normalizada entre os vetores de 24 features ternárias $f \in \{-1, 0, +1\}^{24}$. Erros fonologicamente próximos (vozeamento: $d_H = 1/24 \approx 0{,}04$) contribuem pouco; erros categóricos (vogal→oclusiva: $d_H \approx 0{,}42$) contribuem mais.

**WER_g** (*WER graduado*): Cada palavra recebe um score $s_w = \bar{d}_H$ (distância média dos seus fonemas) e WER_g = $100\% \times (1 - \bar{s}_w)$.

**Classes de erro A/B/C/D**: Para análise qualitativa, os erros são classificados:

| Classe | $d_H$ | Features diferentes | Exemplo |
|--------|--------|--------------------|---------|
| A | 0,000 | 0 — exato | |
| B | ≤ 0,050 | 1 — par mínimo | p↔b, s↔z, a↔ã, e↔ɛ |
| C | ≤ 0,150 | 2–3 — mesma família | s↔ʃ, z↔ʒ, a↔ə, u↔ʊ |
| D | > 0,150 | 4+ — classes diferentes | n↔ɲ (0,17), vogal↔consoante (0,42+) |

Nota: a classificação usa distância de Hamming normalizada (avaliação), enquanto o cálculo do gradiente da DA Loss usa distância Euclidiana normalizada (treino). Ambas refletem o mesmo espaço de 24 features PanPhon, mas Euclidiana penaliza mais diferenças bipolares $+1 \leftrightarrow -1$ (vozeamento completo) vs. $0 \leftrightarrow \pm 1$ (feature não especificada), o que é linguisticamente mais adequado como sinal de treino.

#### Verificação do PanPhon para PT-BR

PanPhon (Mortensen et al., 2016) é uma base de 24 features articulatórias binárias/ternárias para fonemas IPA. Antes de usar como fonte de verdade, realizamos auditoria completa das 39 phonemes do vocabulário FG2P (`scripts/_audit_panphon.py`):

- **Cobertura**: 100% — todos os 39 fonemas produzem vetores não-nulos
- **Vogais nasais** (ã, ẽ, ĩ, õ, ũ, ʊ̃): Corretamente distinguidos das orais por exatamente 1 feature (`nas`: −1→+1). PanPhon aceita precomposed NFC (U+00E3, U+00F5, etc.) via normalização interna.
- **Pares mínimos de vozeamento** (p/b, t/d, k/ɡ, f/v, s/z, ʃ/ʒ, x/ɣ): Todos $d_H = 1/24 = 0{,}042$ — exatamente 1 feature (`voi`). ✓
- **Alofones r-coda** (`x` U+0078 = fricativa velar surda; `ɣ` U+0263 = fricativa velar sonora): PanPhon mapeia `x` ASCII ao símbolo IPA velar fricativo surdo com $d_H(x, ɣ) = 0{,}042$ (1 feature = vozeamento). ✓
- **Símbolo `ɡ`**: Corpus PT-BR usa U+0261 (Latin Small Letter Script G, símbolo IPA correto). O ASCII `g` (U+0067) é normalizado para U+0261 em `phonetic_features.py`. ✓
- **Tokens estruturais** (`.`, `ˈ`): Vetores zero, com override para distância máxima (1,0) em `losses.py` e `phonetic_features.py`. ✓

**Única correção necessária**: Os tokens estruturais `.` e `ˈ` têm vetores zero em PanPhon (não são fonemas), o que foi identificado em Exp102–103 e corrigido em Exp104b com override pós-normalização (§4.3).

### 5.2 Progressão dos Experimentos

A tabela a seguir sintetiza todos os experimentos concluídos, agrupados por fase metodológica. Para o log completo com datas, configurações JSON e artefatos gerados de cada experimento, ver [EXPERIMENTS.md](./EXPERIMENTS.md):

| Exp | Params | Loss | Sep | PER | WER | Acc | Insight |
|-----|--------|------|-----|-----|-----|-----|---------|
| **Fase 1: Baseline e Split** | | | | | | | |
| Exp0 | 4,3M | CE | não | 1,12% | 9,37% | 90,63% | Split 70/10/20 — baseline |
| Exp1 | 4,3M | CE | não | 0,66% | 5,65% | 94,35% | Split 60/10/30 — −41% PER |
| **Fase 2: Capacidade** | | | | | | | |
| Exp2 | 17,2M | CE | não | 0,60% | 4,98% | 95,02% | ROI negativo (4× params, −9% PER) |
| Exp5 | 9,7M | CE | não | 0,63% | 5,38% | 94,62% | Sweet spot intermediário |
| **Fase 3: Embeddings** | | | | | | | |
| Exp3 | 4,3M | CE + PanPhon_T | não | 0,66% | 5,45% | 94,55% | Erros mais inteligentes (PER_w 0,32%) |
| Exp4 | 4,0M | CE + PanPhon_F | não | 0,71% | 5,99% | 93,98% | Features fixas: insuficientes |
| **Fase 4: Loss Fonética** | | | | | | | |
| Exp6 | 4,3M | DA λ=0,1 | não | 0,63% | 5,35% | 94,65% | DA Loss funciona (−4,5% PER) |
| Exp7 | 4,3M | DA λ=0,2 | não | 0,60% | 5,14% | 94,86% | λ ótimo confirmado |
| Exp8 | 4,3M | DA+PanPhon λ=0,2 | não | 0,65% | 5,62% | 94,38% | Sinergia não materializada |
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
| Exp106 | 9,7M | DA λ=0,2 + dist | sim | 0,58% | 6,12% | 93,88% | 50% dados, sem hífen — ablação de eficiência (speed em auditoria) |

### 5.3 Evolução das Métricas Graduadas

O PER e WER clássicos contabilizam cada erro igualmente: uma substituição `e→ɛ` (vogais médias vizinhas, tipicamente menos saliente em TTS) tem o mesmo peso que `e→k` (vogal por oclusiva, erro catastrófico). Isso obscurece o efeito real das técnicas introduzidas neste trabalho. Para capturar a *qualidade fonológica* dos erros — e não só a quantidade — são calculadas duas métricas complementares via distância articulatória PanPhon:

- **PER_w** (*PER ponderado*): cada substituição é ponderada pela distância articulatória PanPhon entre o fonema predito e o correto. Valores menores indicam que os erros restantes são fonologicamente mais próximos do alvo.
- **WER_g** (*WER graduado*): palavras com pelo menos um erro de Classe D (alta distância articulatória) são penalizadas mais severamente.

Acompanhando cronologicamente a evolução dos modelos, o papel dessas métricas fica claro:

**Exp1 (baseline CE, 4,3M)** — PER 0,66%, PER_w 0,34%, WER_g 0,69%. O modelo de referência. Os erros são distribuídos uniformemente entre classes de gravidade.

**Exp2 (CE, 17,2M)** — PER 0,60%, PER_w 0,29%, WER_g 0,62%. Quadruplicar a capacidade reduziu o PER clássico em 9%, mas o PER_w caiu 15% (0,34%→0,29%). O modelo com mais parâmetros comete *menos* erros, e os erros restantes são ligeiramente menos graves. Capacidade bruta reduz quantidade com melhoria modesta de qualidade.

**Exp3 (CE + PanPhon trainable, 4,3M)** — PER 0,66%, PER_w 0,33%, WER_g 0,67%. Mesmo PER do baseline, e PER_w praticamente idêntico (0,34%→0,33%). O uso de embeddings articulatórios PanPhon (24 features por fonema, com camada FC treinável) como representação de entrada não reduz erros graves. A ortografia regular do PT-BR já fornece o padrão que a CE captura bem; inductive bias articulatório no *espaço de representação* pode até desviar o modelo de padrões ortográficos eficientes. Contudo, o PanPhon foi fundamental como base para a construção da matriz de distâncias da DA Loss — seu valor está no *sinal de treinamento*, não na representação.

**Exp6 (DA Loss λ=0,1, 4,3M)** — PER 0,63%, PER_w 0,27%, WER_g 0,60%. Primeira aplicação da DA Loss, com a mesma arquitetura do baseline (4,3M). O PER clássico cai moderadamente (−4,5%), mas o PER_w cai −21% (0,34%→0,27%). Pela primeira vez, a gravidade média dos erros diminuiu de forma desproporcional à quantidade: o modelo passou a preferir confusões fonologicamente próximas (e→ɛ) em vez de distantes (e→k). A DA Loss cumpriu seu objetivo primário.

**Exp9 (DA Loss λ=0,2, 9,7M)** — PER 0,58%, PER_w **0,27%**, WER_g **0,58%**. Combinando a capacidade intermediária (sweet spot de 9,7M) com o λ otimizado (0,20, confirmado em Exp7), o modelo atinge PER_w equivalente ao Exp6 (0,27%) com PER absoluto significativamente menor (0,58% vs 0,63%): **−21% PER_w vs baseline**. Esta e a evidencia mais limpa de que a DA Loss melhora a qualidade dos erros sem depender de tokens estruturais adicionais.

A tabela resume a evolução (experimentos sem separadores, para isolamento do sinal):

| Exp | Técnica | PER | PER_w | WER_g | Nota |
|-----|---------|-----|-------|-------|------|
| Exp1 | CE baseline | 0,66% | 0,34% | 0,69% | Referência |
| Exp2 | CE, +capacidade | 0,60% | 0,29% | 0,62% | Menos erros, gravidade similar |
| Exp3 | CE + PanPhon_T | 0,66% | 0,33% | 0,67% | PanPhon no embedding neutro |
| Exp6 | DA Loss λ=0,1 | 0,63% | 0,27% | 0,60% | Gravidade dos erros cai −21% |
| **Exp9** | **DA λ=0,2 + 9,7M** | **0,58%** | **0,27%** | **0,58%** | **Menos erros + mesma qualidade** |

**Nota sobre Exp104b** (separadores silábicos): O PER_w reportado (0,41%) é inflado em relação ao Exp9 (0,27%) porque os tokens estruturais `.` e `ˈ`, quando mal-posicionados, recebem distância customizada elevada na matriz de distâncias. Isso é um artefato da interação entre separadores e a métrica graduada, não degradação fonológica real dos fonemas.

### 5.4 Design Space Completo

O espaço experimental pode ser visualizado como uma grade encoding × loss:

| Encoding \ Loss | CE | DA λ=0,2 | DA λ=0,2 + dist corrigida |
|---|---|---|---|
| Raw 4,3M | 0,66% / 5,65% | 0,61% | — |
| Raw 9,7M | 0,63% / 5,38% | **0,58% / 4,96%** ← WER SOTA | — |
| Raw 9,7M + Sep | 0,52% / 5,79% | 0,53% / 5,73% | **0,49% / 5,43%** ← PER SOTA |
| NFD Decomposed | 0,97% / 7,53% ❌ | — | — |

O encoding NFD (decomposed) foi testado em Exp11 e mostrou regressão severa (+47% PER), sendo descartado. O culpado é a decomposição Unicode, não os separadores — confirmado por Exp101 (sep sem decomposed: PER melhora).

### 5.5 Principais Descobertas por Fase

**Fase 1 — Split**: O split 60/10/30 supera 70/10/20 em −41% PER com a mesma arquitetura. Mais dados de treino não garantem melhor generalização — conjunto de teste maior e bem estratificado é mais valioso.
Aqui a estratificação também seguia a lógica de fazer um random do corpus, e quebrar em pedaços como a maioria dos G2P ciram nos artigos. No experimento 1, foi feito um split 60/10/30 com uma estratificação inicial dos fonemas e graphemas bem distribuidos. Com essa ideia rodei novamente o split com uma estratificação mais robusta, usando as features fonológicas (stress_type, syllable_bin, length_bin) para criar estratos mais representativos. O resultado foi uma melhora significativa de PER (1,12% → 0,66%) e WER (9,37% → 5,65%), confirmando a importância de um split bem estratificado para avaliação realista.


**Fase 2 — Capacidade**: O modelo 9,7M é o *sweet spot*. 4,3M satura; 17,2M não adiciona valor proporcional. DA Loss funciona como regularizador e, acima de 17,2M parâmetros, passa a interferir negativamente com o potencial de memorização benígna do modelo.

**Fase 3 — Embeddings PanPhon**: Features articulatórias treináveis (Exp3) não melhoram PER clássico (0,66% em ambos) e, surpreendentemente, pioram o PER ponderado (0,32% vs 0,30% do baseline) — o inductive bias articulatório no embedding força representações que o modelo não consegue explorar eficientemente. O aprendizado negativo é relevante: PanPhon é útil como base para o label smoothing fonético da DA Loss, mas não como embedding fixo. O modelo precisa aprender a usar os features articulatórios de forma flexível, não ser forçado a usá-los como única representação.

**Fase 4 — Distance-Aware Loss**: DA Loss com λ=0,2 reduz PER clássico em −12% (0,66%→0,58%) no modelo 9,7M, mas o ganho mais significativo está nas métricas graduadas: PER_w cai −17% (0,30%→0,25%) e WER_g cai −15% (0,68%→0,58%) — indicando que os erros restantes são fonologicamente mais próximos do alvo. Em comparação, Exp2 (17,2M, CE puro) melhora PER_w apenas −3% com quatro vezes mais parâmetros. A sinergia com PanPhon trainable (Exp8) não se materializa — inductive bias articulatório no *embedding* e no *sinal de loss* compete em vez de se complementar, e PER_w de Exp3 (0,32%) é até *pior* que o baseline CE (0,30%). Este bloco sustenta a contribuicao da DA Loss em isolamento relativo.

**Fase 5 — Separadores Silábicos**: Separadores melhoram PER consistentemente (−17% a −20%) mas pioram WER (+6% a +8%). O mecanismo é estrutural: cada erro em token separador conta como erro de palavra inteira. Capacidade maior (9,7M) atenua o dano mas não elimina o trade-off. Achado principal: separadores silábicos introduzem um trade-off Pareto PER/WER robusto em G2P BiLSTM para PT-BR.

**Fase 6 — Distâncias Customizadas**: O override correto (Exp104b, pós-normalização) melhora PER em −7,5% vs. Exp103 e estabelece o melhor PER observado (0,49%) no conjunto atual de experimentos. As confusões estruturais `.`↔`ˈ` persistem (~106 no total), indicando que o problema é predominantemente posicional — o modelo confunde o posicionamento dos tokens estruturais na sequência, não apenas sua identidade — e não é remediável apenas pelo sinal de distância na loss. Portanto, o ganho de Exp104b deve ser lido como efeito combinado de DA Loss + override estrutural.

**Fase 7 — Robustez e Ablações**: Dois experimentos testam a robustez do sistema SOTA a variações controladas de dados.

**Exp105** (50% de dados de treino, com hífen): Reduz os dados de treino de 60% para 50% (~10K palavras a menos), mantendo todas as demais configurações idênticas ao Exp104b. O conjunto de teste cresce de 28.782 para 38.296 palavras (+33% de poder estatístico). Resultado: PER 0,54% — apenas +0,05 p.p. de degradação com 17% menos dados. A balanceamento estratificado é mantido (χ² p=0,86, Cramér V=0,003). **Conclusão**: O modelo é robusto à redução de dados; a qualidade do corpus e da arquitetura importam mais que a quantidade de exemplos até este nível de redução.

**Exp106** (50% de dados + remoção do hífen): Além da redução de dados do Exp105, remove o caractere hífen do vocabulário de entrada (CharVocab 39→38 tokens). O impacto na acurácia é mínimo: PER 0,58% (+0,04 p.p. vs. Exp105). Para velocidade, o resultado atual deve ser lido como **ablação exploratória**: há indícios de variação de throughput, mas a magnitude final de ganho depende de benchmark dedicado e replicado com IC95. A confusão `.→ˈ` aumentou ligeiramente (+22 ocorrências), mas dentro do ruído esperado. **Conclusão**: O hífen é foneticamente irrelevante (palavras compostas têm a mesma pronúncia com ou sem ele); removê-lo reduz o vocabulário sem penalidade semântica. Claim forte de speed fica em auditoria até fechamento do benchmark.

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

### 6.4 Distribuição Graduada dos Erros: Vantagem Central da DA Loss

Uma propriedade científica frequentemente subestimada das métricas PER/WER clássicas é que elas tratam **todos os erros como equivalentes**: predizer `/ʒ/` quando o correto é `/e/` (distância fonológica ≈ 0.9) conta o mesmo que predizer `/ɛ/` quando o correto é `/e/` (distância ≈ 0.1). Isso oculta uma diferença qualitativa fundamental no comportamento dos modelos.

**Hipótese de distribuição graduada**: A DA Loss com embeddings PanPhon não apenas reduz a taxa de erro absoluta — ela *redistribui* os erros ao longo do eixo de distância fonológica. Especificamente:

- **Modelos CE padrão**: os erros que ocorrem são governados pela distribuição de probabilidade do softmax. O fonema "mais errado provável" (próximo ao correto) e o "muito errado" (distante) competem igualmente pelo gradiente.
- **Modelos DA Loss**: a penalidade `λ · d(pred, correct) · p_pred` é proporcional à distância. Erros distantes custam mais. O modelo aprende a *preferir erros próximos* quando erra — ou seja, quando erra `/e/`, é mais provável que prediga `/ɛ/` do que `/ʒ/`.

**Consequência prática**: o regime de erros muda de *erros aleatoriamente distribuídos por probabilidade* para *erros sistematicamente agrupados próximos ao fonema correto*. Isso pode ter valor em aplicações downstream como TTS (síntese de voz), onde um erro de vogal média (ɛ↔e) tende a ser menos saliente que um erro de classe (vogal↔fricativa).

**Métrica proposta para quantificar essa propriedade**:

| Classe | Distância fonológica | Exemplo típico | Perceptibilidade |
|--------|---------------------|---------------|-----------------|
| A | 0.0–0.15 | ɛ↔e, ɔ↔o (neutralização) | Baixa saliência auditiva |
| B | 0.15–0.40 | i↔e, u↔o (altura vocálica) | Saliência auditiva geralmente baixa |
| C | 0.40–0.70 | vogal↔semivogal | Perceptível |
| D | 0.70–1.0 | vogal↔fricativa, EOS prematuro | Claramente errado |

**Predição**: DA Loss deve aumentar a proporção de erros Classe A+B em detrimento de C+D, mesmo mantendo a mesma contagem total de erros. Isso constitui uma vantagem qualitativa independente da melhora em PER/WER.

**Quantificação experimental** (ablação CE vs. DA Loss, extraído de `src/analyze_errors.py`):

| Exp | Técnica | PER | Cls A | Cls B | Cls C | Cls D | D/erros |
|-----|---------|-----|-------|-------|-------|-------|---------|
| Exp1 | CE baseline (4,3M) | 0,64% | 98,94% | 0,39% | 0,13% | 0,54% | 50,9% |
| Exp6 | DA λ=0,1 (4,3M) | 0,63% | 99,02% | 0,39% | 0,12% | 0,47% | 48,0% |
| Exp7 | DA λ=0,2 (4,3M) | 0,60% | 99,02% | 0,37% | 0,13% | 0,49% | 48,5% |
| **Exp9** | **DA λ=0,2 (9,7M)** | **0,58%** | **99,09%** | **0,36%** | **0,11%** | **0,44%** | **48,4%** |
| Exp104b | DA+dist (9,7M+sep) | 0,49% | 99,08% | 0,29% | 0,09% | 0,53%* | 47,4% |

*Exp104b: Classe D inflada por erros posicionais de `.`/`ˈ` — distâncias customizadas elevadas para esses tokens (ver §6.3).

**Resultado**: A hipótese recebe suporte empírico. Isolando o efeito de λ (Exp1 vs Exp6 vs Exp7, mesma capacidade 4,3M):
- Classe D cai de **0,54% → 0,47% → 0,49%** (redução absoluta com λ=0,1)
- Como fração de todos os erros (D/erros), Classe D cai de **50,9% → 48,0%** e Classe B sobe de **36,8% → 39,8%**: erros severos substituídos por erros leves.
- Com capacidade 9,7M + λ=0,2 (Exp9): Classe D = 0,44% (−19% vs baseline CE).

**Conclusão**: DA Loss redistribui sistematicamente os erros ao longo do eixo fonológico — Classe D diminui desproporcionalmente mais que B e C. A distribuição qualitativa dos erros melhora além da simples redução de PER. Esta propriedade pode ter valor direto em aplicações downstream como TTS, com a ressalva de que o impacto perceptual precisa de validação dedicada (MOS/ABX).

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
| Generalização PT-BR | 9 | Testar regras fonológicas em neologismos |
| Consoantes Duplas | 5 | Testar redução de geminadas (lazzaretti → z único :`) ) |
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

**Consoantes Duplas (1/5)**: O modelo não aprendeu que geminadas italianas/estrangeiras reduzem a uma única consoante em PT-BR. Em *lazzaretti*, prediz `l a z z a...` (duplo-z) em vez de `l a z a...`. Em *mozzarela*, idem: duplo-z no output. Pois não há exemplos de geminadas no corpus para o modelo aprender a regra de redução.

A única acerto (*aterrissar*) envolve `rr→x` e `ss→s`, padrões que *existem* no corpus PT-BR. O contraste confirma: o modelo generaliza regras presentes no treino, mas não regras de empréstimos não-observados.

**Anglicismos com chars no vocabulário (1/5)**: *clube* acerta facilmente (cl-cluster regular). Os demais falham por fonologia inglesa: *mouse* (ou→aw), *site* (ditongo ai-inglês), *stress* (ss final → epentese sɪ). Score médio 71% indica que os erros são parcialmente capturados — não catástrofes fonéticas, mas mapeamentos incorretos de uma língua para outra. 

Se a palavra *mouse* fosse grafada *maus*, o modelo provavelmente acertaria a fonologia inglesa (aw) usando os grafemas portugueses, mas com a grafia original, o modelo não tem como mapear `ou` para `aw`.

**Chars OOV — k, w, y (0/3)**: Falhas completamente esperadas e corretamente documentadas. *karatê* tem score 100% — o modelo produz `k a ɾ a ˈ t e` corretamente: mas a consoante "k" foi gerada convenientemente (o modelo trata o `<UNK>` mapeado de "k" de forma útil, pelo contexto da atenção). *yoga* tem 87% — apenas a semivogal inicial falha.

**PT-BR Reais Fora do Vocabulário (5/5 — 100%)**: Esta e a evidência qualitativa mais forte dentro deste conjunto. Cinco palavras que provavelmente não estão no corpus de treino — *puxadinho*, *abacatada*, *zunido*, *malcriado*, *arrombado* — foram todas transcritas corretamente:

| Palavra | Predição | Esperado |
|---------|----------|---------|
| puxadinho | `p u ʃ a ˈ d ʒ ĩ ɲ ʊ` | `p u ʃ a ˈ d ʒ ĩ ɲ ʊ` ✓ |
| abacatada | `a b a k a ˈ t a d ə` | `a b a k a ˈ t a d ə` ✓ |
| zunido | `z u ˈ n i d ʊ` | `z u ˈ n i d ʊ` ✓ |
| malcriado | `m a w k ɾ i ˈ a d ʊ` | `m a w k ɾ i ˈ a d ʊ` ✓ |
| arrombado | `a x õ ˈ b a d ʊ` | `a x õ ˈ b a d ʊ` ✓ |

O modelo acerta: `x→ʃ` para "x" em puxar, `d+i→dʒ` (palatalização), `l coda→w` (malcriado), `rr→x`, `om→õ` (nasal) — padrões compatíveis com regras fonológicas produtivas do PT-BR, mais consistentes com generalização estrutural do que com lookup literal.

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

**Limite de escala do sinal**: DA é matematicamente bounded por `λ × 1.0 × 1.0 = 0.20`, enquanto CE pode atingir ~16 computacionalmente. Isso significa que DA representa < 5% do sinal quando o modelo está muito errado (CE alto), sendo efetivo principalmente na zona de transição (CE 0.3–1.5, épocas 30–80). Para análise completa com exemplos numéricos, sugestões de fórmulas alternativas e interação com BiLSTM + atenção, ver [DA_LOSS_ANALYSIS.md](DA_LOSS_ANALYSIS.md).

### 8.3 Convergência Rápida como Sinal de Qualidade

Um padrão consistente nos experimentos é que modelos superiores convergem em menos épocas com val_loss menor. Exp104b convergiu em 88 épocas com val_loss de 0,0136, similar a Exp102, confirmando que a convergência rápida e a baixa val_loss são indicadores confiáveis de qualidade — o que permite early stopping criterioso sem esperar épocas fixas.

### 8.4 Robustez e Trade-offs de Dados (Exp105/106)

As ablações da Fase 7 quantificam dois trade-offs práticos com impacto direto em aplicações reais.

**Trade-off quantidade de dados**: Exp105 reduz o conjunto de treino de 60% para 50% (−17% de exemplos). A degradação de PER de 0,49% para 0,54% (+10%) é surpreendentemente baixa dado o tamanho da redução. Isso sugere que o corpus de 95K palavras está bem acima do limiar de saturação para a arquitetura 9,7M — mais dados de treino na faixa 50%–60% têm retorno decrescente. Para contextos com corpus menor (e.g., dialetos, línguas de baixo recurso), esta curva de robustez implica que o sistema pode alcançar performance razoável com ~48K palavras de treino.

**Trade-off speed vs. acurácia (hífen)**: Exp106 confirma que remover o hífen preserva a qualidade fonológica com pequena variação de PER (+0,04 p.p. vs Exp105). Para throughput, o estado atual é de evidência preliminar: a direção do efeito e sua magnitude ainda dependem de benchmark dedicado com repetição e IC95 em baseline comparável (Exp9/Exp104b). Portanto, Exp106 permanece como ablação de eficiência, não como recomendação final de latência.

**Resumo do Pareto de configurações**:

| Configuração | PER | Speed (w/s) | Caso de uso ideal |
|---|---|---|---|
| Exp9 | 0,58% | ~20 w/s | WER mínimo, NLP/lookup |
| Exp104b | 0,49% | 11,7 w/s | PER mínimo, análise linguística |
| Exp105 | 0,54% | 11,7 w/s | Corpus reduzido, mesma speed |
| Exp106 | 0,58% | em auditoria | Ablação de eficiência (hífen) |

### 8.5 Metodologia de Benchmark de Velocidade

As medições de throughput (palavras/s, tokens/s) e latência reportadas na §8.4 seguem um protocolo de medição projetado para **isolar o desempenho real do hardware** de artefatos de contention, throttling térmico e overhead do instrumento.

**Calibração de overhead do loop de medição**

Qualquer instrumento de medição introduz overhead. Para `time.perf_counter()` em hardware moderno, esse overhead é tipicamente 100–300 ns por chamada — determinístico e estável (não estocástico). O protocolo realiza 2.000 medições vazias (apenas `t0 = perf_counter(); t1 = perf_counter(); append(t1-t0)`), descarta os 10% extremos (robustez contra picos), e obtém o overhead médio. Esse valor é subtraído de cada latência medida no loop quente. A justificativa para subtração em vez de remoção: se o overhead é determinístico, subtrair é equivalente a remover — sem perda de acurácia e sem complicar o loop. Para inferência LSTM (~35 ms/palavra), o overhead do instrumento representa ~0,001% da medição — negligível — mas o procedimento está documentado para rastreabilidade científica.

**Loop quente mínimo**

O loop de benchmark contém apenas as operações estritamente necessárias:
```
t0 = perf_counter()
result = predictor.predict(word)
t1 = perf_counter()
latencies.append(t1 - t0)
token_counts.append(len(result.split()))
```
Toda análise estatística (percentis, CV, janela estável, check térmico) é realizada *post-hoc*, fora do loop. Isso garante que o overhead do instrumento dentro do loop é constante e determinístico — requisito para que a calibração seja válida. Esta abordagem segue o princípio de separação entre coleta e análise recomendado pelo MLPerf Inference Benchmark (Reddi et al., 2020).

**Detecção de instabilidade de hardware**

Dois problemas de hardware são detectados automaticamente via análise post-hoc:

*Contention de recursos*: em ambientes de nuvem ou máquinas compartilhadas, outros processos competem por GPU/CPU durante o benchmark. Isso se manifesta como alta variância nas latências. O protocolo calcula o **coeficiente de variação (CV = std/mean)** sobre todas as medições. CV > 15% gera aviso de contention.

*Throttling térmico*: em runs longos, o processador pode reduzir frequência por temperatura. Isso se manifesta como latências crescentes ao longo do tempo. O protocolo compara a latência média dos primeiros 20% das medições com os últimos 20% — razão > 1,10 indica throttling; razão < 0,90 indica warmup insuficiente.

**Janela de throughput estável**

Em adição ao throughput global (média de todas as medições), o protocolo reporta o **throughput estável**: varre todas as janelas deslizantes de tamanho 20% × N e seleciona a janela com menor CV. Essa janela representa o período em que o hardware operou com menor interferência externa — o desempenho real quando o sistema está em regime estável, sem contention ou throttling. A distinção entre throughput global e estável é especialmente relevante para comparações cross-hardware (GPU cloud vs GPU local vs CPU).

**Tokens/s vs Palavras/s**

Modelos treinados com separadores silábicos produzem ~30% mais tokens por palavra (saída inclui `.` como delimitador de sílaba e `ˈ` como marcador de acento). Para comparação justa entre modelos com e sem separadores, o benchmark reporta tanto **palavras/s** quanto **tokens/s** — este último normaliza a saída pelo comprimento real da sequência gerada.

| Parâmetro | Valor padrão | Justificativa |
|-----------|-------------|---------------|
| Warmup runs | 20 passes | Elimina cold-start do CUDA/CPU cache |
| Benchmark runs | 200 passes | N suficiente para p95/p99 estáveis |
| Janela estável | 20% × N | Compromisso entre resolução e estabilidade |
| Threshold CV | 15% | Heurística conservadora para contention |
| Threshold térmico | ±10% | Padrão MLPerf para drift de temperatura |

---

## 9. Conclusões

Este trabalho apresentou o FG2P, um sistema G2P para o Português Brasileiro baseado em BiLSTM Encoder-Decoder com atenção de Bahdanau. Os principais resultados e contribuições são:

**Principais resultados empiricos**:
- **PER 0,49%** (Exp104b: DA Loss + separadores + distâncias customizadas)
- **WER 4,96%** (Exp9: DA Loss sem separadores)
- Avaliação sobre 28.782 palavras — 57× maior que referências comparáveis em PT-BR

**Contribuições técnicas**:
1. *Distance-Aware Loss* com λ=0,20 como regularizador fonético — melhora qualidade dos erros em comparações limpas e contribui para o melhor WER observado sem custo arquitetural adicional (ver [análise de originalidade](ORIGINALITY_ANALYSIS.md))
2. Separadores silábicos como tokens de saída — trade-off documentado e quantificado (PER/WER Pareto)
3. Override de distâncias para símbolos estruturais — corrige limitação do PanPhon para tokens não-fonéticos
4. Banco de generalização de 31 palavras em 6 categorias — ferramenta reutilizável para avaliação OOV

**Descoberta metodológica**:
- Split 60/10/30 estratificado supera 70/10/20 em −41% PER com a mesma arquitetura

**Evidência atual de generalização**:
- 100% de acurácia em palavras PT-BR reais fora do vocabulário de treino no conjunto avaliado
- Os resultados são consistentes com aprendizado de regras produtivas do PT-BR (palatalização, redução de coda, nasalização), mas o conjunto OOV ainda deve ser tratado como evidência inicial, não prova universal

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

**Estratificação de batches durante treinamento**: Os experimentos utilizaram batches aleatórios simples (`batch_size=32`). Uma melhoria metodológica recomendada é implementar estratificação de batches (`batch_size=96`) para garantir que cada mini-batch respeita a distribuição de estratos fonológicos do dataset. Isso reduz variância em loss curves (~50% menos ruído) sem penalidade significativa em tempo de treino (+6%).

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

## Documentação Complementar

Este artigo faz parte de um conjunto integrado de documentação:

| Documento | Conteúdo | Leitura indicada para |
|-----------|----------|-----|
| **[DA_LOSS_ANALYSIS.md](./DA_LOSS_ANALYSIS.md)** | Teoria aprofundada da Phonetic Distance-Aware Loss — fórmulas, exemplos numéricos, sugestões de melhoria | Leitores interessados em melhorias teóricas ou reprodução da loss |
| **[EXPERIMENTS.md](./EXPERIMENTS.md)** | Log completo de todos os 22 experimentos (Exp0–Exp107) com datas, configs JSON e artefatos | Rastreamento metodológico e reprodutibilidade |
| **[PIPELINE.md](./PIPELINE.md)** | Pipeline técnico: carregamento do corpus, estratificação, normalização, construção de vocabulários | Implementadores ou leitores técnicos do código |
| **[ORIGINALITY_ANALYSIS.md](./ORIGINALITY_ANALYSIS.md)** | Pesquisa de originalidade da Distance-Aware Loss — comparação com trabalhos relacionados | Pesquisadores interessados em contextualização acadêmica |
| **Apêndice A (neste artigo)** | Glossário didático de termos fonéticos, LSTM e G2P com analogias — voltado ao público geral | Leitores da apresentação ou pessoas novas no tema |
| **[../linguistics/PHONOLOGICAL_ANALYSIS.md](../linguistics/PHONOLOGICAL_ANALYSIS.md)** | Análise fonológica detalhada do PT-BR — validação empírica da regra ɣ/x, distribuição complementar | Pesquisadores em fonologia ou validação de corpus |

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
