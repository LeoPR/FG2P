# Distance-Aware Loss: Penalidade Fonologicamente Graduada para Conversao Grafema-Fonema no Portugues Brasileiro

**[STIL 2026 — Submissao Anonima]**
**Autores**: [ANONIMO]
**Afiliacao**: [ANONIMO]

---

## Resumo

Este trabalho apresenta uma funcao de custo fonologicamente informada — a *Distance-Aware (DA) Loss* — para sistemas de conversao grafema-fonema (G2P) no Portugues Brasileiro. Diferente da CrossEntropy padrao, que trata todos os erros de substituicao igualmente, a DA Loss penaliza erros proporcionalmente a distancia articulatoria entre o fonema predito e o alvo, ponderada pela confianca do modelo na predicao incorreta. A distancia e calculada sobre 24 features articulatorias do PanPhon. Aplicada a um BiLSTM encoder-decoder com atencao de Bahdanau, a DA Loss redistribui sistematicamente os erros das classes catastroficas (Classe D, fonemas articulatoriamente distantes) para classes mais proximas (Classe B, pares minimos). O sistema e avaliado sobre um conjunto de teste estratificado de 28.782 palavras (~181k fonemas) — escala 57x maior que avaliacoes comparaveis em PT-BR — alcancando PER de 0,48% (IC Wilson 95% [0,46%; 0,51%]) e WER de 5,33% na configuracao de referencia; uma configuracao complementar sem separadores silabicos atinge WER de 4,96%. Uma analise fatorial de capacidade x separadores silabicos x DA Loss revela que a interacao entre esses fatores, nao a capacidade isolada, determina o desempenho. Testes em 31 palavras fora do vocabulario (OOV) em 6 categorias mostram 100% de acuracia em palavras reais ineditas do PT-BR, evidencia consistente com generalizacao de regras fonologicas alem da memorizacao.

**Palavras-chave**: G2P, conversao grafema-fonema, Portugues Brasileiro, BiLSTM, Distance-Aware Loss, fonologia, PanPhon.

---

## 1. Introducao

A conversao automatica de texto escrito em representacao fonetica — tarefa *Grapheme-to-Phoneme* (G2P) — e componente fundamental de sistemas de sintese de fala (TTS), reconhecimento de fala e processamento de linguagem natural. Para o Portugues Brasileiro, a tarefa apresenta desafios especificos: (i) **ambiguidade grafemica** — o grafema "c" realiza-se como /k/ em "cama" mas /s/ em "cena"; o "r" em coda realiza-se /x/ em final de palavra mas /ɣ/ antes de consoante vozeada; (ii) **neutralizacao vocalica** — em silabas atonas, /ɛ/↔/e/ e /ɔ/↔/o/ neutralizam-se, introduzindo ambiguidade legitima no corpus; (iii) **suprassegmentais** — o acento tonico (ˈ) e fronteiras silabicas (.) sao tokens nao-articulatorios que exigem tratamento especial na funcao de custo.

Modelos seq2seq treinados com CrossEntropy (CE) tratam todos os erros igualmente: predizer /ɛ/ quando o correto e /e/ — par minimo com 1 feature de diferenca — incorre na mesma penalidade que predizer /k/ para /a/ — erro de 8+ features. Essa cegueira fonologica distorce o sinal de treinamento: o modelo aprende *que* errou, mas nao *o quanto* errou.

Este trabalho propoe a **Distance-Aware (DA) Loss**, que adiciona ao sinal de treinamento um termo proporcional a distancia articulatoria entre o fonema predito e o alvo, ponderado pela confianca do modelo. Aplicamos essa funcao a um BiLSTM encoder-decoder com atencao de Bahdanau — arquitetura deliberadamente escolhida para isolar a contribuicao da funcao de custo de novidades arquiteturais.

**Comparacao com LatPhon.** A referencia mais proxima e o LatPhon [Chary et al. 2025], um Transformer multilingue de 4 camadas (7,5M params, RoPE) que reporta PER de 0,86% (IC Wilson 95% [0,56%; 1,16%]) em ~500 palavras PT-BR do mesmo dicionario fonetico. Nosso sistema alcanca PER de 0,48% (IC [0,46%; 0,51%]) em 28.782 palavras. Os intervalos de confianca nao se sobrepoem — o limite superior do nosso sistema (0,51%) fica abaixo do limite inferior do LatPhon (0,56%), diferenca estatisticamente significativa a 95% de confianca.

**Contribuicoes**:
1. DA Loss: objetivo de treinamento fonologicamente graduado combinando distancia articulatoria PanPhon com confianca de predicao
2. Avaliacao em larga escala com split estratificado e evidencia empirica do impacto de vies de split
3. Analise fatorial de capacidade x separadores x DA Loss
4. Taxonomia de qualidade de erros (Classes A-D) revelando redistribuicao sistematica de erros catastroficos
5. Avaliacao de generalizacao OOV em 6 categorias diagnosticas

---

## 2. Dados e Protocolo de Avaliacao

### 2.1 Corpus

O corpus de treinamento consiste em **95.937 pares (palavra, transcricao IPA)** de um dicionario fonetico do Portugues Brasileiro. O charset de entrada cobre a-z (exceto k, w, y, ausentes do dicionario) mais diacriticos portugueses. Palavras contendo k, w ou y sao tratadas como OOV de caractere. Previamente ao treinamento, 10.252 entradas foram corrigidas para a distincao ASCII-g (U+0067) vs. IPA-ɡ (U+0261), necessaria para lookup correto de features PanPhon.

### 2.2 Split Estratificado

O corpus e dividido **60/10/30** (treino/validacao/teste) com estratificacao por tres variaveis fonologicas: tipo de acento (oxitona/paroxitona/proparoxitona), faixa de contagem de silabas (1, 2, 3, 4, 5+) e faixa de comprimento em grafemas (<=4, 5-7, 8-10, 11+). A combinacao gera ~48 estratos.

Qualidade do split: chi-quadrado=0,95 (p=0,678), Cramer V=0,0007 — ausencia de diferenca distribucional significativa entre subconjuntos.

**Evidencia empirica de vies de split.** Um split nao-estratificado 70/10/20 (Exp0, mesma arquitetura 4,3M, CE) atingiu PER de 1,12%, enquanto o split estratificado 60/10/30 (Exp1) atingiu 0,66% — reducao de 41% no PER atribuivel inteiramente ao protocolo de avaliacao, nao a melhoria do modelo. Sem estratificacao, a particao aleatoria pode concentrar palavras dificeis (proparoxitonas, palavras longas) no treino e faceis no teste, inflando metricas artificialmente.

### 2.3 Metricas

**PER** (Phoneme Error Rate): distancia de Levenshtein sobre sequencias de fonemas, normalizada pelo comprimento da referencia [Morris et al. 2004; Bisani e Ney 2008]. **WER** (Word Error Rate): fracao de palavras com qualquer erro (match exato). **IC Wilson 95%** [Wilson 1927; Brown et al. 2001]: usado no lugar do intervalo de Wald, que subestima incerteza proximo a p->0. Para nosso test set (~181k fonemas de referencia), o IC Wilson sobre o PER e de +/-0,03 p.p.

**Metricas graduadas (contribuicao deste trabalho)**: **PER_w** (PER ponderado), onde cada substituicao e ponderada pela distancia articulatoria PanPhon normalizada. Classes de erro: A (exato), B (<=0,050, par minimo), C (<=0,150, mesma familia), D (>0,150, classes diferentes).

---

## 3. Arquitetura

Utilizamos um **BiLSTM encoder-decoder com atencao de Bahdanau** [Bahdanau et al. 2014], arquitetura estabelecida para G2P supervisionado [Rao et al. 2015].

**Encoder**: BiLSTM de 2 camadas. Para cada posicao t, h_t = [h->_t; h<-_t] fornece contexto bidirecional completo — critico para resolver ambiguidades grafemicas onde a identidade do fonema depende dos caracteres circundantes.

**Atencao de Bahdanau**: a cada passo de decodificacao t, o vetor de contexto c_t e calculado como soma ponderada dos estados do encoder: e_{t,j} = v^T tanh(W_h h_j + W_s s_{t-1}), alfa_{t,j} = softmax(e_{t,j}), c_t = sum_j alfa_{t,j} h_j.

**Decoder**: LSTM de 2 camadas, teacher forcing no treino, autoregressivo na inferencia.

**Configuracoes testadas**:

| Config | Embedding | Hidden | Parametros |
|--------|-----------|--------|------------|
| Pequena | 128D | 256D | 4,3M |
| Intermediaria | 192D | 384D | 9,7M |
| Grande | 256D | 512D | 17,2M |

---

## 4. Distance-Aware Loss

### 4.1 O Problema

A CE trata todos os erros igualmente. Fonologicamente, isso e inadequado: substituir /ɛ/ por /e/ (1 feature de diferenca) e qualitativamente diferente de substituir /a/ por /k/ (8+ features).

| Situacao | Erro | d_PanPhon | CE |
|----------|------|-----------|----|
| A: prediz ɛ, correto e | near-miss (1 feature) | 0,10 | 1,0 |
| B: prediz k, correto a | catastrofico (8+ features) | 0,90 | 1,0 |

### 4.2 A Formula

$$L = L_{CE} + \lambda \cdot d_{PanPhon}(\hat{y}_i, y_i) \cdot p_i^{(\hat{y}_i)}$$

Onde:
- **d_PanPhon(y-hat, y)**: distancia calculada sobre 24 features articulatorias binarias do PanPhon [Mortensen et al. 2016], normalizada para [0, 1]
- **p_i^(y-hat_i)**: probabilidade atribuida pelo modelo ao fonema predito (argmax do softmax) — fator de confianca
- **lambda**: peso do sinal fonologico (otimo empirico: lambda=0,20)

A DA Loss adiciona uma penalidade extra proporcional a: (i) o quanto o modelo errou (distancia) e (ii) o quao confiante estava no erro (probabilidade). Quando o modelo esta incerto entre dois candidatos, aprende a "desempatar para o lado correto" — preferir o fonema articulatoriamente mais proximo do alvo.

### 4.3 Busca de lambda

| lambda | PER | WER | Comportamento |
|--------|-----|-----|---------------|
| 0,05 | 0,62% | 5,36% | Sinal fraco |
| 0,10 | 0,63% | 5,35% | Melhora moderada |
| **0,20** | **0,60%** | **5,14%** | **Otimo** |
| 0,50 | 0,65% | 5,57% | Sobrepenalizacao |

Curva em U-invertido: muito pouco sinal = inocuo; muito = atrapalha a CE.

### 4.4 Distancias Customizadas para Simbolos Estruturais

O PanPhon atribui vetor zero a tokens nao-foneticos (separador silabico ".", marcador de acento "ˈ"), resultando em d(., ˈ)=0,0. Override pos-normalizacao seta d=1,0 para pares envolvendo simbolos estruturais, corrigindo o problema com 3 linhas de codigo.

---

## 5. Experimentos e Resultados

### 5.1 Progressao dos Experimentos

| Exp | Params | Loss | Sep | PER | WER | Insight |
|-----|--------|------|-----|-----|-----|---------|
| Exp0 | 4,3M | CE | nao | 1,12% | 9,37% | Split 70/10/20 — baseline |
| Exp1 | 4,3M | CE | nao | 0,66% | 5,65% | Split 60/10/30 — -41% PER |
| Exp5 | 9,7M | CE | nao | 0,63% | 5,38% | Sweet spot |
| Exp7 | 4,3M | DA 0,2 | nao | 0,60% | 5,14% | lambda otimo |
| **Exp9** | **9,7M** | **DA 0,2** | **nao** | **0,58%** | **4,96%** | **Melhor WER** |
| Exp102 | 9,7M | CE | sim | 0,52% | 5,79% | Sep melhora PER, piora WER |
| Exp103 | 9,7M | DA 0,2 | sim | 0,53% | 5,73% | Efeitos nao-aditivos |
| **Exp104d** | **17,2M** | **DA+dist** | **sim** | **0,48%** | **5,33%** | **Referencia PER** |

### 5.2 Analise Fatorial: Separadores x DA Loss

Fatorial 2x2 limpo (9,7M params, split 60/10/30):

|  | CE | DA lambda=0,2 |
|--|----|----|
| Sem sep | 0,63% / 5,38% | **0,58% / 4,96%** |
| Com sep | 0,52% / 5,79% | 0,53% / 5,73% |

**Descobertas**:
- Separadores melhoram PER consistentemente (-17% a -20%) mas pioram WER (+6% a +8%)
- DA Loss melhora tanto PER quanto WER sem separadores
- Com separadores, o efeito da DA Loss e atenuado — o sinal da CE ja e dominante para posicionamento de tokens estruturais

### 5.3 Redistribuicao Graduada dos Erros

| Exp | Tecnica | PER | Cls B | Cls D | D/erros |
|-----|---------|-----|-------|-------|---------|
| Exp1 | CE baseline | 0,66% | 0,39% | 0,54% | 50,9% |
| Exp6 | DA 0,1 | 0,63% | 0,39% | 0,47% | 48,0% |
| Exp7 | DA 0,2 | 0,60% | 0,37% | 0,49% | 48,5% |
| **Exp9** | **DA 0,2 (9,7M)** | **0,58%** | **0,36%** | **0,44%** | **48,4%** |

Classe D (catastrofica) cai de 0,54% para 0,44% (-19%) enquanto Classe B (proxima) se mantem — os erros severos sao substituidos por erros leves. A DA Loss redistribui sistematicamente os erros ao longo do eixo fonologico.

---

## 6. Analise de Erros

### 6.1 Padroes de Confusao Dominantes

Mais de 60% dos erros sao neutralizacoes vocalicas — substituicoes entre vogais medias abertas e fechadas (/ɛ/<->/e/, /ɔ/<->/o/). Estas refletem ambiguidade fonologica genuina do PT-BR: em posicao atona, vogais medias neutralizam-se. A distribuicao no corpus revela a causa: /e/ tem razao 24,9:1 vs. /ɛ/ em posicao pre-tonica, mas /ɛ/ e dominante em posicao tonica (razao 0,33:1). O modelo aprende o vies pre-tonico e o generaliza para silabas tonicas.

### 6.2 Regra Alofonica r-coda

Durante a avaliacao de generalizacao, o modelo produziu /ɣ/ em coda antes de consoante vozeada (e.g., *borboleta* -> b o ɣ . b o...). Auditoria do corpus revelou que o modelo estava **correto**: a regra de assimilacao de vozeamento do PT-BR [Barbosa e Albano 2004] exige /ɣ/ antes de vozeada e /x/ em coda final — distribuicao complementar com 0 excecoes no corpus de 95.937 entradas.

### 6.3 Avaliacao de Generalizacao OOV

Banco diagnostico de 31 palavras em 6 categorias:

| Categoria | Corretas | Score Fonol. | Insight |
|-----------|----------|-------------|---------|
| Generalizacao PT-BR | 6/9 (67%) | 97% | Near-misses: ĩ->i |
| Consoantes Duplas | 1/5 (20%) | 81% | Geminadas nao vistas |
| Anglicismos (invocab) | 1/5 (20%) | 71% | Fonologia inglesa OOV |
| Chars OOV (k/w/y) | 0/3 (0%) | 68% | Falha esperada |
| **PT-BR Reais (OOV)** | **5/5 (100%)** | **100%** | **Generalizacao perfeita** |
| Controles (em treino) | 4/4 (100%) | 100% | Baseline de sanidade |

**Total**: 17/31 (55%). O resultado 5/5 em palavras reais ineditas — *puxadinho*, *abacatada*, *zunido*, *malcriado*, *arrombado* — todas transcritas corretamente, e evidencia de que o modelo aprendeu regras fonologicas produtivas do PT-BR (palatalizacao, reducao de coda, nasalizacao), nao memorizou o corpus.

---

## 7. Discussao

### 7.1 Trade-off PER/WER com Separadores

Separadores silabicos melhoram PER (-17% a -20%) mas pioram WER (+6% a +8%), independentemente da capacidade ou funcao de custo. O mecanismo e estrutural: cada token separador mal-posicionado conta como erro de palavra inteira. A escolha entre regimes depende da aplicacao: TTS prioriza PER (Exp104d); NLP/lookup prioriza WER (Exp9).

### 7.2 Limites da DA Loss

**Limite estrutural**: Simbolos nao-foneticos recebem vetor zero no PanPhon. O override corrige parcialmente, mas confusoes posicionais persistem.

**Limite de escala do sinal**: DA e bounded por lambda x 1,0 x 1,0 = 0,20, enquanto CE pode atingir ~16. DA representa <5% do sinal quando o modelo esta muito errado, sendo efetiva principalmente na zona de transicao (CE 0,3-1,5).

### 7.3 Memorizacao vs. Aprendizado

O split 60/10/30 e deliberado: prioriza test set grande sobre treino maximo. Exp107 (95% treino, 960 teste) atinge PER 0,46%, mas com IC 160% mais amplo. A diferenca 0,46% vs. 0,48% esta no ruido. Mais relevante: com 95% em treino, o risco de memorizacao e muito maior. O acerto de 100% em palavras reais ineditas (Secao 6.3) sustenta a hipotese de aprendizado de regras.

---

## 8. Limitacoes

1. **Conjunto OOV pequeno**: O banco de 31 palavras e uma sonda diagnostica, nao avaliacao exaustiva. Universalizacao das conclusoes de generalizacao requer banco OOV maior e mais diverso.
2. **Monolinguismo**: O sistema foi treinado e avaliado apenas em PT-BR. Transferencia da DA Loss para outros idiomas requer validacao especifica.
3. **Validacao perceptual ausente**: A hipotese de que erros proximos sao menos saliveis auditivamente em TTS nao foi validada com testes de percepcao (MOS/ABX).
4. **Corpus unico**: Todos os experimentos usam o mesmo dicionario fonetico. Generalizacao para outros corpora PT-BR (FalaBrasil, CommonVoice) nao foi testada.
5. **Arquitetura unica**: DA Loss foi avaliada apenas com BiLSTM. Avaliacao com Transformers e CTC permanece como trabalho futuro.

---

## 9. Declaracao de Etica

Este trabalho utiliza um dicionario fonetico publicamente disponivel sem informacoes pessoais identificaveis. Nao foram realizados experimentos com participantes humanos. Os modelos treinados convertem texto em representacao fonetica e nao geram conteudo potencialmente prejudicial. Nao identificamos riscos eticos significativos associados a esta pesquisa.

---

## 10. Uso de IA Generativa

Ferramentas de IA generativa (Claude, Anthropic) foram utilizadas exclusivamente como assistencia editorial: revisao gramatical e ortografica, pesquisa de referencias bibliograficas, gestao de citacoes, verificacao de consistencia terminologica entre secoes e formatacao LaTeX. A ferramenta nao gerou texto novo, hipoteses, figuras, codigo nem interpretacoes de dados. Todo o conteudo cientifico — design experimental, implementacao, execucao dos experimentos, analise dos resultados e conclusoes — e de autoria exclusiva dos autores. Este uso enquadra-se no Nivel 2 (assistencia editorial) conforme taxonomia de Resnik e Hosseini (2025).

---

## Referencias

[Bahdanau et al. 2014] Bahdanau, D., Cho, K. e Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv:1409.0473.

[Barbosa e Albano 2004] Barbosa, P. A. e Albano, E. C. (2004). Brazilian Portuguese. Journal of the International Phonetic Association, 34(2):227-232.

[Bisani e Ney 2008] Bisani, M. e Ney, H. (2008). Joint-Sequence Models for Grapheme-to-Phoneme Conversion. Speech Communication, 50(5):434-451.

[Brown et al. 2001] Brown, L. D., Cai, T. T. e DasGupta, A. (2001). Interval Estimation for a Binomial Proportion. Statistical Science, 16(2):101-133.

[Chary et al. 2025] Chary, K. et al. (2025). LatPhon: Multilingual Grapheme-to-Phoneme Conversion with Language-Aware Encoders. arXiv:2509.03300.

[Kohavi 1995] Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. In Proc. IJCAI, volume 2, pages 1137-1143.

[Morris et al. 2004] Morris, A. C., Maier, V. e Green, P. (2004). From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition. In Interspeech, pages 2765-2768.

[Mortensen et al. 2016] Mortensen, D. R. et al. (2016). PanPhon: A Resource for Mapping IPA Segments to Articulatory Feature Vectors. In Proc. COLING, pages 3264-3273.

[Rao et al. 2015] Rao, K., Sak, H. e Prabhavalkar, R. (2015). Grapheme-to-Phoneme Conversion Using Long Short-Term Memory Recurrent Neural Networks. In Proc. ICASSP.

[Wilson 1927] Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. Journal of the American Statistical Association, 22(158):209-212.
