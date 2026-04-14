# Camadas de Tokenizacao no FG2P — Pesquisa em Andamento

**Ultima atualizacao**: 2026-04-12
**Status**: SKELETON — documento em construcao.
**Proposito**: Mapear as camadas entre o formato de arquivo dos dicionarios
e o token real que alimenta o gradiente da DA Loss. Nao contem decisoes
finais — contem perguntas abertas que precisam de discussao dedicada.

> **AVISO IMPORTANTE**: Este documento esta **deliberadamente incompleto**.
> A tokenizacao afeta diretamente o gradiente da DA Loss e a distancia
> PanPhon — decisoes aqui nao podem ser feitas a esmo. Cada camada precisa
> de discussao propria, com pesquisa de formatos oficiais amplos e atualizados
> (nao formatos arbitrarios do projeto). Ver secao "Perguntas abertas".

---

## 1. Por que este documento existe

O FG2P tem um pipeline conceitual:

```
Arquivo fisico no disco (.tsv)
  ↓
Parser do dataset (Python)
  ↓
Tokenizacao (split de IPA em unidades discretas)
  ↓
Vocabulario do modelo (PhonemeVocab)
  ↓
Indices inteiros para o modelo
  ↓
Embedding layer
  ↓
Forward pass (LSTM encoder-decoder)
  ↓
Logits -> softmax -> argmax
  ↓
Perda CE (cross-entropy) sobre indices
  ↓
Perda DA (distance-aware) sobre distancia PanPhon
  ↓
Gradiente
  ↓
Atualizacao de pesos
```

Cada seta representa **uma transformacao** que pode alterar a
representacao da informacao fonetica. Erros ou escolhas mal-documentadas
em qualquer camada propagam-se para o modelo e para a interpretacao
cientifica dos resultados.

Este documento mapeia essas camadas e lista as perguntas abertas em
cada uma.

---

## 2. Camadas identificadas (listagem inicial, a completar)

### Camada 0 — Representacao fisica no disco

**O que e**: bytes no arquivo `.tsv`.

**Exemplo atual** (canonical v1.x, `dicts/pt-br.tsv`):
```
abadia<TAB>a . b a . ˈ d ʒ i . ə<LF>
```

**Caracteristicas observaveis**:
- UTF-8 sem BOM
- Normalizacao Unicode: NFC (aparentemente — a confirmar)
- Line endings: a verificar (CRLF ou LF)
- Separator: tab entre palavra e IPA; espaco entre tokens IPA
- Um par palavra/IPA por linha

**Perguntas abertas**:
1. NFC ou NFD? Qual foi a decisao do v1.x e por que?
2. Line ending consistente?
3. O formato TSV e adequado para representar metadata futura (POS, freq,
   origem)? Ou migraremos para JSONL, Parquet, YAML?
4. Como lidar com palavras com tab interno (caso exista)?
5. Como representar palavras homografas com pronuncias diferentes
   (`forma` verbo vs substantivo)?

---

### Camada 1 — Parse do arquivo

**O que e**: leitura do arquivo e separacao em estruturas de dados.

**Implementacao atual** (aproximada, a verificar em `src/g2p.py`):
```python
for line in file:
    word, ipa_string = line.rstrip('\n').split('\t', 1)
```

**Saida**: pares `(word, ipa_string)` onde `ipa_string` e uma string
com espacos delimitando tokens.

**Perguntas abertas**:
1. O parser faz NFC/NFD explicitamente ou assume que o arquivo ja esta
   normalizado?
2. Como trata CRLF vs LF?
3. Como trata linhas vazias, comentarios `#`, linhas malformadas?
4. O parser valida que o conteudo de `ipa_string` contem so simbolos
   IPA validos, ou aceita qualquer coisa?

---

### Camada 2 — Tokenizacao do IPA string

**O que e**: transformar a string de IPA em uma lista de tokens.

**Implementacao atual** (provavelmente):
```python
tokens = ipa_string.split()  # split por qualquer whitespace
# resultado: ['a', '.', 'b', 'a', '.', 'ˈ', 'd', 'ʒ', 'i', '.', 'ə']
```

**Observacao critica**: aqui o **espaco no arquivo** e consumido como
delimitador. O espaco **nao vira token**. O que era separador visual
no arquivo vira fronteira logica entre tokens.

**Perguntas abertas** (as mais importantes de todas):
1. **Qual e a unidade de token?** 1 caractere Unicode? 1 segmento IPA
   atomico (podendo ser multi-char como `tʃ`)? 1 gesto articulatorio?
2. **Como o arquivo comunica isso?** Hoje o espaco e a unica pista.
   Isso e suficiente ou precisa de metadata explicita?
3. Se um fonema **composto** (como africadas `tʃ`, `dʒ`) aparece num
   arquivo novo sem espaco, o modelo interpreta como 1 ou 2 tokens?
4. Como lidar com NFD de outros idiomas (ex: alemao `ü` vs `u + U+0308`)
   de forma consistente com o canonical PT-BR?
5. O que acontece com simbolos estruturais (`.`, `ˈ`)? Sao tokens reais
   que o modelo preve, ou sao consumidos como metadata silabica?
6. Diferentes idiomas (PT-BR, EN-US, Tupinamba) podem ter convencoes
   diferentes no mesmo super-modelo multilingue? Ou precisamos forcar
   uma convencao unica?

---

### Camada 3 — Construcao do vocabulario (PhonemeVocab)

**O que e**: coletar todos os tokens distintos do corpus e atribuir
indices inteiros.

**Implementacao atual** (provavelmente em `src/g2p.py::PhonemeVocab`):
```python
vocab = set()
for word, tokens in corpus:
    vocab.update(tokens)
token_to_id = {tok: i for i, tok in enumerate(sorted(vocab))}
```

**Caracteristicas observaveis**:
- Vocab do v1.x tem ~39 tokens (a confirmar)
- Inclui: fonemas IPA, `.`, `ˈ`, tokens especiais `<PAD>`, `<UNK>`, `<EOS>`
- Ordem dos indices: "primeira ocorrencia no dataset" (ja documentado em
  MEMORY.md)

**Perguntas abertas**:
1. Token desconhecido no v2 multilingue: cai em `<UNK>` global ou tem
   `<UNK>` por idioma?
2. Ordem dos indices importa? Hoje e arbitraria. Seria melhor ordenar por
   frequencia? Por similaridade PanPhon?
3. Tamanho do vocab escala com multilingue. Ha um limite pratico?

---

### Camada 4 — Conversao para indices

**O que e**: mapear lista de tokens para lista de inteiros.

**Implementacao atual**:
```python
indices = [token_to_id.get(t, unk_id) for t in tokens]
```

**Perguntas abertas**:
1. OOV em tempo de treino (palavra nova com fonema que nao estava no
   vocab de treino): como lidar? Hoje e `<UNK>`. Perde informacao.
2. Ha alguma normalizacao de ultimo minuto aqui (ex: NFC final)?

---

### Camada 5 — Embedding layer

**O que e**: transformar indice inteiro em vetor denso.

**Implementacao atual**: `nn.Embedding(vocab_size, embedding_dim)`.

**Dois modos no v1.x**:
- `learned` — embedding treinavel, inicializado com Glorot
- `panphon_T` / `panphon_F` — inicializado com vetores PanPhon 24D
  projetados para dim maior (trainable ou frozen)

**Perguntas abertas**:
1. A decisao token = segmento atomico e **necessaria** para PanPhon init
   funcionar. Trocar essa convencao quebra o embedding geometrico.
   Confirmar.
2. Em multilingue, o embedding e compartilhado entre idiomas? Ou cada
   idioma tem seu proprio?
3. Tokens estruturais (`.`, `ˈ`) recebem vetor zero no PanPhon init —
   isso ja e tratado em `phonetic_features.py` com early-return. Mas
   tokens compostos novos (se existissem) cairiam onde?

---

### Camada 6 — DA Loss (onde a tokenizacao importa MAIS)

**O que e**: calculo da perda que combina CE com distancia articulatoria.

**Formula atual**:
```
L = L_CE + λ * d_PanPhon(y_hat, y) * p(y_hat)
```

Onde `d_PanPhon(a, b)` e uma distancia pre-calculada entre **tokens**
a e b. A tabela de distancias tem dimensao `vocab_size × vocab_size`.

**Por que a tokenizacao impacta DA Loss**:
- Se `tʃ` vira **1 token** com seu proprio vetor PanPhon, a distancia
  entre `tʃ` e `ʃ` e uma unica medida direta
- Se `tʃ` vira **2 tokens** (`t` e `ʃ`), o modelo preve cada um
  separadamente, e as distancias sao calculadas entre pares individuais
- A distribuicao de erros nas classes A/B/C/D muda drasticamente entre
  as duas convencoes
- O PER muda (mais tokens no denominador = PER menor aparente)

**Perguntas abertas** (muito importantes):
1. A distribuicao de erros Classe A/B/C/D reportada nos papers v1.x
   assume qual convencao? Char-por-token separado?
2. Se mudarmos a convencao, os numeros dos papers ficam invalidados?
3. Override de distancia para tokens estruturais (`.`, `ˈ`) — como se
   integra com DA Loss? Ja esta implementado em `losses.py`, mas vale
   revisar.
4. Em multilingue, a distancia PanPhon e universal ou tem ajuste por
   idioma?

---

### Camada 7 — Gradiente e atualizacao de pesos

**O que e**: backprop e atualizacao do embedding + pesos do LSTM.

**Perguntas abertas**:
1. O gradiente "reforca" a geometria PanPhon no embedding treinavel, ou
   destroi ela ao longo do treino? (hipotese ja investigada no v1.x)
2. Em multilingue, um fonema comum entre idiomas recebe gradiente somado
   de todos? Ha algum balanceamento?

---

## 3. Hipoteses nao decididas (precisam estudo dedicado)

### H1. Token atomico vs gesto articulatorio
Ver analise em conversa anterior (2026-04-12). Consenso tentativo: token
atomico para G2P, gesto para TTS/ASR. Mas precisa fechar via **analise
formal com referencias**:
- Browman & Goldstein (1992) — Articulatory Phonology
- Fant (1960) — teoria fonte/filtro
- Mortensen et al. (2016) — PanPhon (ja no REFERENCES.bib)
- Xue et al. (2022) — ByT5 (character-level)
- Literatura recente de G2P multilingue

### H2. Formato do arquivo de dicionario
TSV e adequado? JSONL? Parquet? Precisa pesquisar padroes **oficiais**
usados pela comunidade:
- CMUdict (usa formato proprio)
- Wiktionary (XML + JSON)
- CLDR (XML)
- Common Voice (TSV)
- LREC datasets (varios)

Pergunta central: qual formato permite metadata extensivel, multi-idioma,
versionamento, auditoria, e interoperabilidade com ferramentas padrao de
NLP?

### H3. Normalizacao Unicode — PARCIALMENTE RESOLVIDA (2026-04-13)

**Decisao operacional (v2.0)**: NFC como formato padrao no pipeline e
nos arquivos em disco. Implementado via action `nfc true` no `.rules.tsv`.

**Achados empiricos com PanPhon (2026-04-13)**:

1. PanPhon internamente converte para **NFD** (usa `unicodedata.normalize('NFD')`
   na sua funcao `normalize()`). Mas aceita **tanto NFC quanto NFD** como input
   e produz vetores 24D **identicos** em ambos os casos.

2. Teste empírico com 6 segmentos (nasais PT-BR):

```
Segmento              NFC?  NFD?  PanPhon segs  nas  Vetores iguais?
NFC a-tilde (U+00E3)  sim   nao   1             +1   referencia
NFD a-tilde (a+0303)  nao   sim   1             +1   SIM (24/24 features)
NFC e-tilde (U+1EBD)  sim   nao   1             +1   —
NFD e-tilde (e+0303)  nao   sim   1             +1   —
epsilon puro (U+025B) —     —     1             -1   —
epsilon-nasal (ɛ+0303)nao   nao*  1             +1   diff: apenas nas (-1→+1)
```

*ɛ̃ NAO tem forma precomposed NFC — `NFC('ɛ'+U+0303)` continua como
2 chars. PanPhon reconhece como 1 segmento mesmo assim.

3. A nasalizacao **ja e uma dimensao do PanPhon 24D** (feature `nas`).
   Nao e necessario criar dimensao 25D para nasais. A diferenca entre
   vogal oral e nasal e **1 feature de 24** (distancia PanPhon minima).

4. Total de segmentos nasais reconhecidos pelo PanPhon: **725** (incluindo
   combinacoes com diacriticos combinantes).

**Fluxo de normalizacao confirmado no codigo**:

```
Arquivo disco     g2p.py L410     PhonemeVocab     PanPhon interno
NFC ou NFD   -->  NFC (forcado) -->  NFC tokens  -->  NFD (transparente)
```

O `g2p.py` L410 forca NFC na carga. PanPhon converte internamente para
NFD. A action `nfc` no pipeline alinha o arquivo em disco com o que o
modelo espera. Resultado: menos uma transformacao implicita.

**Perguntas que permanecem abertas**:
- NFKC e NFKD: nao avaliados. Poderiam ser relevantes para scripts
  exoticos (ex: meia-largura katakana). Nao e prioridade para PT-BR.
- Idiomas com segmentos que nao tem forma NFC precomposed: o pipeline
  aceita mixed (NFC onde possivel, NFD onde nao). PanPhon e agnostico.
- Parametrizacao do encode: hoje o `g2p.py` L410 e hardcoded NFC.
  Idealmente seria parametro do manifest (v2.x).
- Pos-filtro de saida: na inferencia, se o consumer espera NFD,
  precisa de filtro configuravel (v2.x).

### H4. Simbolos estruturais como tokens
Hoje `.` e `ˈ` sao tokens no vocab. Isso e pragmatico (o modelo aprende
silabificacao e stress como parte do output) mas **conceitualmente**
mistura segmentos fonologicos com metadata prosodica.

Alternativas:
- Tokens separados (atual) — simples, mas tem overhead no PER
- Metadata paralela — mais limpo, mas precisa arquitetura multi-head
- Caracteres no proprio fonema (ex: `ˈa` como 1 token) — mais tokens,
  menos overhead estrutural

### H5. Multilingue: vocab compartilhado vs por-idioma
Em multilingue, dois modelos possiveis:
- **Vocab universal**: um conjunto unico de fonemas cobrindo todos os
  idiomas. Transfer entre idiomas e gratis. Mas vocab grande.
- **Vocab por idioma**: cada idioma tem seu sub-vocab. Menor, mas perde
  transfer implicito.

A decisao afeta a arquitetura do embedding e da tag BCP 47 (que ja foi
discutida).

---

## 4. Dependencias externas a pesquisar

Antes de fechar qualquer camada, precisamos pesquisar:

- [ ] **Unicode**: qual forma normalizada usar e por que
  - UAX #15 (Unicode Normalization Forms) — documento oficial
- [ ] **IPA**: existe um "padrao oficial" para escrita de sequencias IPA
  em arquivos de texto?
  - IPA Handbook (Cambridge University Press)
  - X-SAMPA como alternativa ASCII-safe
- [ ] **Metadata fonologica**: existe ontologia padrao?
  - PHOIBLE (https://phoible.org/) — ontologia de inventarios fonologicos
  - CLTS (Cross-Linguistic Transcription Systems) — padronizacao
- [ ] **Formato de arquivo**: quais formatos a literatura G2P multilingue
  usa?
  - ipa-dict (nosso submodulo): TSV simples
  - CMUdict: formato proprio
  - Epitran: TSV + mappings
  - LanguageNet, Wiktionary, Wikipron

---

## 5. Ordem sugerida de pesquisa

Antes de implementar qualquer coisa na Fase 2.5+, fechar:

1. **Unicode normalization form** — decisao binaria, com base em UAX #15
   e testes empiricos com PanPhon e dados multilingues.
2. **Convencao de token atomico** — fechar via analise de literatura +
   conferencia com papers de G2P recentes.
3. **Formato do arquivo** — pesquisa comparativa de TSV/JSONL/Parquet
   com requisitos (metadata, versionamento, diff amigavel).
4. **Tokens estruturais** — decisao arquitetural dependente de 2 e 3.
5. **Vocab multilingue** — decisao dependente de todos os acima.

Cada item acima merece seu **proprio ticket de pesquisa** quando chegar
a hora.

---

## 6. O que este documento NAO e

- Nao e uma especificacao de formato — nao ha decisoes fechadas aqui.
- Nao e um guia de implementacao — nao diz como codar nada.
- Nao e substituto para leitura de fontes primarias — ele **aponta**
  para elas.
- Nao e estavel — vai evoluir conforme a pesquisa progride.

## 7. O que este documento E

- Um **mapa de dependencias** entre camadas de tokenizacao
- Um **registro de perguntas abertas** que precisam de estudo
- Um **ponto de partida** para tickets de pesquisa especificos
- Um **alerta** de que decisoes aqui afetam o gradiente da DA Loss
  diretamente e nao podem ser feitas a esmo

---

## Historico

- **2026-04-13** — Fase 3: achados empiricos do PanPhon e NFC/NFD
  adicionados a H3. Confirmado que PanPhon e agnostico a NFC/NFD
  (produz vetores identicos). NFC adotado como padrao operacional via
  action `nfc true` no pipeline. 42.293 diferencas Unicode eliminadas.
- **2026-04-12** — criacao inicial (skeleton) apos conversa sobre
  tokenizacao, espacos no arquivo, e relacao com DA Loss. Status:
  em construcao.

## Referencias iniciais

- Browman, C. P. & Goldstein, L. (1992). "Articulatory Phonology: An
  Overview". Phonetica 49:155-180.
  (ja em REFERENCES.bib)
- Fant, G. (1960). "Acoustic Theory of Speech Production". Mouton, The Hague.
  (a adicionar ao REFERENCES.bib)
- Mortensen, D. R. et al. (2016). "PanPhon: A Resource for Mapping IPA
  Segments to Articulatory Feature Vectors". COLING 2016.
  (ja em REFERENCES.bib)
- Unicode Standard Annex #15 — Unicode Normalization Forms.
  https://unicode.org/reports/tr15/
- IPA Handbook (1999). Cambridge University Press.
- PHOIBLE — https://phoible.org/
- CLTS — https://clts.clld.org/
