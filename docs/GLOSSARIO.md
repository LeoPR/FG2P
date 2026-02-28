# Glossário Unificado — Fonética, Algoritmos, e Projeto FG2P

**Propósito**: Centralizar todas as definições pedagógicas para a apresentação FG2P em um único arquivo.

---

## 🎯 PROJETO FG2P

### **G2P (Grapheme-to-Phoneme)**
Sistema que converte **grafemas** (letras escritas) em **fonemas** (sons).

```
Entrada:  "gato"  (grafemas: g-a-t-o)
↓
[G2P Model]
↓
Saída:    /ɡ a t u/  (fonemas: oclusiva-velar-vozeada, vogal-baixa, ...)
```

**Aplicações**:
- **TTS (Text-to-Speech)**: Ler documentos em voz
- **Busca fonética**: Encontrar "gato" mesmo escrito "gatt"
- **Análise linguística**: Entender padrões sonoros

---

### **LSTM (Long Short-Term Memory)**
Tipo avançado de rede neural para **processar sequências**.

**Por quê LSTM?**
- Problema: RNNs simples "esquecem" informações antigas
- Solução: LSTM tem "memória" de longo prazo + curto prazo
- Resultado: Acorda o stress/contexto mesmo em palavras longas

```
Palavra: "c-o-m-p-u-t-a-d-o-r"
         ↓
    [LSTM Encoder] ← lembra de tudo
         ↓
    [LSTM Decoder] ← gera fonemas sabendo o contexto todo
```

---

### **PT-BR (Português Brasileiro)**
Variante do português com características fonológicas distintas.

**Diferenças vs. Português Europeu**:
- `/tʃ/` em "tia" (português BR) vs `/ti/` (português EU)
- Redução vocálica diferente em átona
- Rhotacism: /r/ em coda tem dois alofones

---

## 📍 FONÉTICA PT-BR — Articulações e Dimensões Vocálicas

### **Dimensões Articulatórias — Como Funciona na Boca**

#### **Ponto de Articulação** (Onde na boca soa)

Imagine a cavidade oral como um mapa de zonas. Cada consoante é produzida tocando/aproximando elementos da boca em posições específicas:

```
         palato duro (céu da boca)
              ↓
labial ─ alveolar ─ palatal ─ velar
  ↑        ↑         ↑        ↑
lábios   dentes   céu da boca  garganta
```

| Termo | Onde | Exemplos PT-BR | Como Soa |
|-------|------|-----------------|----------|
| **Labial** | Lábios (superior/inferior) | /p/, /b/, /m/ | "**p**ão", "**b**ola", "**m**ão" |
| **Alveolar** | Atrás dos dentes superiores (crista alveolar) | /t/, /d/, /n/, /s/, /z/, /ɾ/ | "**t**ato", "**d**ado", "**n**ão", "**s**ala", "**z**ero" |
| **Palatal** | Céu da boca (parte dura anterior) | /ʃ/, /ʒ/, /ɲ/, /j/ | "**ch**ave" (/ʃ/), "**j**ar" (/ʒ/), "nho" (/ɲ/) |
| **Velar** | Véu (parte mole do palato, garganta) | /k/, /ɡ/, /ŋ/ | "**c**asa" (/k/), "**g**ato", "co**ng**o" (/ŋ/) |
| **Labiodental** | Lábio inferior + dentes superiores | /f/, /v/ | "**f**ogo", "**v**inho" |

#### **Modo de Articulação** (Como o ar passa)

Diferentes formas de bloquear/deixar o ar passar:

| Termo | Mecanismo | Exemplos | Como Soa |
|-------|-----------|----------|----------|
| **Oclusiva** (ou "stop") | Bloqueia COMPLETAMENTE o fluxo de ar | /p/, /b/, /t/, /d/, /k/, /ɡ/ | "**p**ão" — há explosão de ar |
| **Fricativa** | Ar passa com FRICÇÃO (barulho) | /f/, /v/, /s/, /z/, /ʃ/, /ʒ/, /x/ | "**s**ala" — ar passando entre língua e dentes |
| **Nasal** | Ar passa pela NARIZ (véu abaixado) | /m/, /n/, /ɲ/ | "**m**ão", "**n**ão" — ressonância nasal |
| **Africada** | Começa OCLUSIVA, termina FRICATIVA | /tʃ/, /dʒ/ | "**t**chia" (português padrão "tia"), "**j**ar" |
| **Lateral** | Ar passa pelos LADOS da língua | /l/ | "**l**ado" — ar passa lateralmente |
| **Vibrante** | Língua VIBRA rapidamente | /r/ (múltipla), /ɾ/ (simples) | "ca**rr**o" (/r/ vibrante), "ca**r**a" (/ɾ/ simples) |

#### **Vozeamento** (Cordas Vocais Vibram?)

| Termo | O Que Faz | Pares PT-BR |
|-------|-----------|------------|
| **Vozeado** | Cordas vocais **vibram** | /b/, /d/, /ɡ/, /v/, /z/, /ʒ/ — "**b**ola", "**d**ado" |
| **Desvozeado** | Cordas vocais **não vibram** | /p/, /t/, /k/, /f/, /s/, /ʃ/ — "**p**ão", "**t**ato" |

**Teste de vozeamento**: Coloque a mão na garganta ao falar `/b/` (sente vibração) vs `/p/` (não sente)

---

### **Dimensões Vocálicas — Vogais PT-BR**

Vogais são **sons abertos** — o ar flui livremente. A posição da língua determina a qualidade:

```
        FRENTE          CENTRAL         TRÁS
ALTO      /i/                             /u/
          (como "si")                     (como "tu")

MÉDIO     /e/              /ə/            /o/
          (como "pé")    (neutra)        (como "pó")

BAIXO               /a/
              (como "pá")
```

#### **Altura da Língua** (Vertical)

| Nível | Termo | Exemplos | Sensação |
|-------|-------|----------|----------|
| **Alto** | Língua perto do palato | /i/, /u/ | "si", "tu" — língua levantada |
| **Médio** | Língua no meio | /e/, /o/, /ə/ | "pé", "pó" — língua meia-altura |
| **Baixo** | Língua baixa, boca aberta | /a/ | "pá" — boca bem aberta |

#### **Posição Anterior-Posterior** (Horizontal)

| Posição | Exemplos | Sensação |
|---------|----------|----------|
| **Anterior (Frente)** | /i/, /e/, /a/ | Língua para frente ("si", "pé", "pá") |
| **Posterior (Trás)** | /u/, /o/ | Língua para trás ("tu", "pó") |
| **Central** | /ə/ | Língua neutra (posição de repouso) |

#### **Arredondamento dos Lábios**

| Tipo | Exemplos | Sensação |
|------|----------|----------|
| **Não-arredondado** | /i/, /e/, /a/ | Lábios abertos/espalhados |
| **Arredondado** | /u/, /o/ | Lábios em "O" |

---

### **Prosódia — Acentuação e Timing**

#### **Stress (Acento Tônico)**

Em português, uma sílaba é **tônica** (acentuada) ou **átona** (desacentuada):

| Termo | O Quê | Exemplo |
|-------|-------|---------|
| **Tônica** | Sílaba **pronunciada com mais força** | com**PU**-ta-dor (2ª sílaba enfatizada) |
| **Átona** | Sílaba **pronunciada com menos força** | com-pu-ta-**DOR** (outras são fracas) |
| **Redução vocálica** | Vogal átona **muda de timbre** | ca**sa** → /ə/ (em vez de /a/) |

**Representação IPA**: `/ˈ/` marca stress. Exemplo: com·**ˈ**pu·ta·dor

#### **Silabificação**

Uma sílaba contém um **núcleo vocálico** cercado opcionalmente por consoantes:

```
  Onset    Núcleo    Coda
    ↓        ↓        ↓
   (C)      (V)      (C)

   [con] [som] [ante] — estrutura possível
```

| Padrão | Exemplos | Notas |
|--------|----------|-------|
| **V** (aberta) | a, e, o | Vogal pura |
| **CV** (aberta) | ba, te, do | Consoante + Vogal |
| **CVC** (fechada) | bal, ter, dor | Consoante + Vogal + Consoante |
| **CCV** | pra, tre, gra | 2 consoantes + vogal (clusters) |
| **CCVC** | prat, tren | 2 consoantes + vogal + consoante |

#### **Contexto Fonológico — Influências**

**Coda (Posição Final de Sílaba)**: Consoantes em coda sofrem mudanças:

| Contexto | Fenômeno | Exemplo |
|----------|----------|---------|
| **Coda final de palavra** | /r/ final → /x/ (fricativa velar) | "**ar**" → /ax/ (pronuncia-se como "arr" suave) |
| **Coda antes de C vozeada** | /r/ → /ɣ/ (fricativa velar vozeada) | "borbo**le**ta" → /boɾ**ɡ**o/**le/**tə/ (r assimilado) |
| **Coda antes de C desvozeada** | /z/ antes /p/: /s/ | "despe**dir**" → /des**pe**dir/ (z desvozeado → s) |

---

## 🤖 APRENDIZADO DE MÁQUINA — Conceitos Básicos

### **Modelo**
Um **modelo** é uma função matemática que aprende padrões a partir de dados.

**Analogia**: Como uma criança aprende a reconhecer uma árvore vendo vários exemplos, um modelo aprende a reconhecer padrões vendo dados de treino.

```
Dados de treino → [Modelo aprende padrões] → Modelo treinado → Predições
```

**No FG2P**: O modelo recebe uma palavra (ex: "computador") e prediz os sons (/k õ p u t a ˈ d o x/).

---

### **Treino (Training)**
Processo de **ajustar os parâmetros do modelo** para minimizar erro.

| Termo | O Quê |
|-------|-------|
| **Época (Epoch)** | Uma passada completa pelos dados de treino |
| **Batch** | Um pequeno grupo de exemplos processados por vez |
| **Learning Rate** | "Tamanho do passo" ao ajustar parâmetros (muito rápido = instável; muito lento = demora) |
| **Early Stopping** | Parar o treino quando o modelo para de melhorar (evita overfitting) |

---

### **Validação e Teste**
- **Validação**: Dados usados para monitorar progresso durante treino (não treina, apenas observa)
- **Teste**: Dados **nunca vistos** antes, usados para avaliar o modelo final

**Por quê separar?** Se testar com dados que treinou, o modelo parece melhor do que realmente é (memorização).

---

## 📊 MÉTRICAS E VALIDAÇÃO

### **Cross Entropy (CE) — Função de Perda**

A **Cross Entropy** mede "quanto erro o modelo está cometendo".

**Ideia simples**:
- Se o modelo **acerta** completamente → perda = 0
- Se o modelo **erra** completamente → perda = alta

**Problema do CE no G2P**:
- Tratar `/b/` vs `/p/` como igualmente ruins (distância de 0)
- Tratar `/b/` vs `/ə/` como igualmente ruins (distância de 0)
- Na realidade, `/b/` e `/p/` são **muito parecidos** (só vozeamento varia)

**Solução**: Usar **Distance-Aware Loss** (ver abaixo).

---

### **Distance-Aware Loss (DA Loss)**

Uma **loss customizada** que penaliza erros **proporcionalmente à distância articulatória**.

```
L = L_CE + λ · d(ŷ, y) · p(ŷ)

Componentes:
  L_CE       = perda base (CrossEntropy)
  d(ŷ, y)   = distância articulatória entre predito e correto (0-1)
  p(ŷ)      = confiança do modelo no token predito (0-1)
  λ         = peso do sinal fonológico (0.2 empiricamente)
```

**Interpretação**:
- Se erra `/b/` quando era `/p/` (distância ≈ 0.05) → penalidade pequena
- Se erra `/b/` quando era `/a/` (distância ≈ 0.90) → penalidade grande
- Se está CONFIANTE no erro → penalidade maior

**Resultado**: Modelo aprende a "preferir erros inteligentes".

---

### **PER (Phoneme Error Rate)**
Porcentagem de **fonemas individuais** errados.

```
PER = (número de fonemas errados) / (total de fonemas) × 100%

Exemplo:
  Correto:  /k õ p u t a ˈ d o x/       (10 fonemas)
  Predito:  /k õ p u t ə ˈ d o x/       (substitui /a/→/ə/)
  Erros:    1 (um erro em 10)
  PER:      10%
```

**Foco**: Acerto individual de fonemas (importante para TTS, análise linguística).

---

### **WER (Word Error Rate)**
Porcentagem de **palavras inteiras** com qualquer erro.

```
WER = (número de palavras com ≥1 erro) / (total de palavras) × 100%

Exemplo:
  Palavra:  "computador"
  Correto:  /k õ p u t a ˈ d o x/
  Predito:  /k õ p u t ə ˈ d o x/       (1 fonema errado)
  Resultado: FALHA (a palavra inteira conta como erro)
  WER:      100% para essa palavra
```

**Foco**: Qualidade geral (importante para busca, indexação, NLP).

**Trade-off**: PER e WER geralmente são **inversamente correlacionados** (trade-off Pareto).

---

### **Accuracy (Acurácia)**
Simples: porcentagem de acertos.

```
Accuracy = (acertos) / (total) × 100%
```

---

### **Overfitting (Sobreajuste)**
O modelo **memoriza os dados de treino** em vez de aprender padrões gerais.

```
Treino:  99% acurácia  ✓
Teste:   60% acurácia  ✗

Conclusão: O modelo memorizou treino, não aprendeu regras.
```

**Prevenção**: Early stopping, validação, regularização.

---

### **Underfitting (Subajuste)**
O modelo é **muito simples** para capturar os padrões.

```
Treino:  70% acurácia
Teste:   68% acurácia

Conclusão: O modelo não é bom em nada (nem em treino).
```

**Solução**: Modelo mais complexo, mais dados, treino mais longo.

---

### **Generalização**
Capacidade do modelo de **fazer boas predições em dados novos**.

**No FG2P**:
- **Overfitting** = modelo só acerta palavras que viu (30K treino)
- **Boa generalização** = modelo acerta palavras novas (5/5 palavras OOV)

---

## 🧠 ARQUITETURA NEURAL

### **RNN (Recurrent Neural Network)**
Uma rede que **processa sequências** lembrando do que viu antes.

**Analogia**: Imagine uma pessoa lendo uma palavra letra por letra. Para cada letra, ela **lembra das letras anteriores** para prever o som:

```
Palavra: c-o-m-p-u-t-a-d-o-r
         ↓
    [RNN remembers]
    "vi 'c' + 'o' + 'm' + 'p' + 'u' + 't' + 'a'..."
         ↓
    Prediz: /a/ (e não /ə/, porque lembra que é tônica)
```

**LSTM (Long Short-Term Memory)**: Versão melhorada de RNN que "lembra por mais tempo".

---

### **Embedding (Embedding Layer)**
Converte **símbolos discretos** (letras, fonemas) em **vetores numéricos** que a rede entende.

```
Letra 'a' → Vetor numérico [0.2, -0.5, 0.8, ...]
Letra 'b' → Vetor numérico [0.3, -0.4, 0.7, ...]
```

**Mágica**: Letras **parecidas** recebem vetores **parecidos**.

---

### **Attention Mechanism (Mecanismo de Atenção)**
Permite o modelo **focar em partes importantes** da entrada.

**Analogia**: Ao ler "computador", o modelo concentra atenção em:
- "co**m**p" para decidir a palatalização
- "compu**t**a" para decidir o stress
- "computa**dor**" para decidir a vogal final

Sem atenção, o modelo trata todas as letras igualmente (ineficiente).

---

## 📈 TÉCNICAS DE TREINAMENTO

### **Data Augmentation (Aumento de Dados)**
Criar **exemplos artificiais** a partir dos reais para treinar melhor.

**No FG2P**: Poderia remover hífens, aplicar filtros grafêmicos, etc.

---

### **Label Smoothing**
Em vez de:
```
Correto: 1.0
Errado:  0.0
```

Usar:
```
Correto:  0.9
Errado:   0.1 / 4 (distribuído)
```

**Efeito**: Modelo fica menos confiante (evita overfitting).

**No FG2P**: Distance-Aware Loss é versão **não-uniforme** disso — penaliza proporcionalmente à distância.

---

### **Regularização**
Técnicas para evitar overfitting:

| Técnica | Função |
|---------|--------|
| **Dropout** | Desativa neurônios aleatoriamente (força o modelo a redundância) |
| **L1/L2 Regularization** | Penaliza parâmetros grandes (simplifica modelo) |
| **Early Stopping** | Para treino quando validação para de melhorar |

---

## 🎓 TERMOS DE AVALIAÇÃO

### **SOTA (State-of-the-Art)**
"**Estado da arte**" — o melhor resultado conhecido até o momento.

```
Exp104b: PER 0.49% ← SOTA PER (nosso melhor)
Exp9:    WER 4.96% ← SOTA WER (nosso melhor)
```

**Competidores**:
- LatPhon 2025 (PT-BR especializado, mas corpus pequeno)
- ByT5-Small (multilíngue, mas 30× maior)

---

### **Baseline**
Resultado de **referência simples** para comparação.

```
Exp1 (Baseline): 0.66% PER
Exp104b (SOTA):  0.49% PER
Melhoria:        ~25% relativa
```

---

### **Trade-off (Compromisso)**
Situação onde melhorar uma métrica **piora outra**.

**No FG2P**:
```
Com separadores silábicos:
  ✓ PER melhora: 0.58% → 0.52% (menos erros fonema)
  ✗ WER piora:   4.96% → 5.79% (mais erros palavra)

Razão: Um separador mal-posicionado = palavra inteira errada
```

---

### **Distribuição Estratificada**
Dividir dados mantendo **proporções representativas**.

```
Dataset completo:
  - 60% treino (57.561 palavras)
  - 10% validação (9.594 palavras)
  - 30% teste (28.782 palavras) ← maior para medição estatística confiável
```

**Teste de balanceamento**: χ² = 0.95 (p = 0.678) — distribuição fonológica balanceada ✓

---

## 🔬 TÉCNICAS E INOVAÇÕES DO FG2P

### **Separadores Silábicos**
Adicionar token **`.`** entre sílabas.

**Entrada com separadores**:
```
Sem:  c-o-m-p-u-t-a-d-o-r
Com:  c-o-.-m-p-u-.-t-a-.-d-o-r
```

**Efeito**:
- ✓ Modelo aprende limites silábicos
- ✓ PER melhora (0.58% → 0.52%)
- ✗ WER piora (4.96% → 5.79%)

**Trade-off permanece**: Melhoria em fonemas, piora em palavras inteiras.

---

### **Distâncias Customizadas**
PanPhon (ferramenta padrão) tem problema: **marcas diacríticas** (stress `.`, silabificação `ˈ`) têm distância = 0.

```
Problema:
  d(., ˈ) = 0.0 ← Loss não diferencia confusão!

Solução (Exp104b):
  d(., anything) = 1.0  ← força máxima penalidade
  d(ˈ, anything) = 1.0 ← força máxima penalidade
```

**Resultado**: Exp104b reduz confusão `.↔ˈ` de ~119 para ~106.

---

## 🧪 CATEGORIAS DE TESTE

### **Neologismos**
Palavras **novas** criadas no português contemporâneo.

**Exemplos**:
- "printar" (do inglês "print")
- "tchau" (abrasileiramento)
- "computadorzinho" (diminutivo)

**Teste**: Modelo acerta generalizações PT-BR (portmanteaux, diminutivos)?

---

### **Palavras OOV (Out-of-Vocabulary)**
Palavras **não vistas no treino**, usadas para testar **generalização genuína**.

```
Treino: 57.561 palavras (60% do corpus)
Teste OOV: 5 palavras PT-BR reais nunca vistas

Resultado: 5/5 corretas (100% de sucesso) ← prova de generalização
```

---

### **Geminadas (Consoantes Duplas)**
Consoantes repetidas: "pp", "zz", "tt".

**Desafio**: Treino tem poucas geminadas (maioria são empréstimos).

```
Palavra:  "cappuccino" (itáliano)
Treino:   < 0.01% geminadas
Resultado: Modelo falha (gap conhecido)
```

---

### **Anglicismos**
Empréstimos do inglês com **fonologia estrangeira**.

```
Palavra:  "mouse" (inglês)
Esperado: /m a w s ə/ (passaporte português)
Modelo:   Acerta se fonologia PT-BR, erra se tentar inglês

Status: Parcial (fonologia em parte PT-BR)
```

---

### **OOV Caractere**
Caracteres **nunca vistos** (fora do charset de treino).

```
Charset treino: a-z (exceto k,w,y) + ç + acentos = 39 chars
Palavras fora:  "yoga", "wifi" (têm k, w, y)

Resultado: Falha esperada (sem esses sons no treino)
```

---

### **Controles**
Palavras que o modelo **deve acertar** (verificação de sanidade).

```
Controle 1: "biscoito" ← simples, comum
Controle 2: "computador" ← complexo, no artigo

Esperado: 100% acurácia (senão há bug)
Resultado: ✓ (4/4 controles acertos)
```

---

## 📈 MÉTRICAS DE QUALIDADE

### **Phonological Score (Score Fonológico)**
Medida **não-binária** de qualidade mesmo quando erra.

```
Exemplo:
  Correto:  /a/ (vogal baixa, central)
  Predito:  /ə/ (vogal neutra, central)

Score: 95% (mesmo erro, mesma região articulatória)
```

**Escala**:
- **100%**: Exato
- **95-99%**: Muito próximo (um traço diferente)
- **80-94%**: Próximo (2 traços diferentes)
- **50-79%**: Parcial (3 traços diferentes)
- **< 50%**: Distante (> 3 traços diferentes)

---

### **Character Coverage**
Porcentagem de caracteres de uma palavra que estão no vocabulário.

```
Palavra: "yoga"
Vocab:   a-z (sem y,w,k)

Chars:   y-o-g-a
          ✗ ✓ ✓ ✓

Coverage: 75% (3 de 4 caracteres conhecidos)
OOV:      {y}
```

---

## 📚 ESTRUTURA ACADÊMICA

### **Artigo (Paper)**
Documento técnico completo (~700 linhas) com:
- Motivação e problema
- Revisão de literatura
- Metodologia e dataset
- Resultados e análise
- Conclusões e trabalho futuro

**Arquivo**: `docs/16_SCIENTIFIC_ARTICLE.md`

---

### **Apresentação (Presentation)**
Resumo visual (26 slides + glossários) para comunicação em conferência.

**Formato**: Markdown Marp → PPTX gerado automaticamente

**Estrutura**:
1. Motivação (slides 1-3)
2. Metodologia (slides 4-13)
3. Resultados (slides 14-21)
4. Conclusões (slides 22-26)
5. Glossários (referência final)

---

### **Relatório (Report)**
Documento HTML dinâmico com todos os experimentos disponíveis.

**Gerado automaticamente** de `models/*/metadata.json`.

---

## 📌 ABREVIAÇÕES COMUNS

| Abreviação | Significado |
|------------|------------|
| **PT-BR** | Português Brasileiro |
| **G2P** | Grapheme-to-Phoneme |
| **LSTM** | Long Short-Term Memory |
| **RNN** | Recurrent Neural Network |
| **IPA** | International Phonetic Alphabet |
| **CE** | Cross Entropy |
| **DA** | Distance-Aware |
| **PER** | Phoneme Error Rate |
| **WER** | Word Error Rate |
| **SOTA** | State-of-the-Art |
| **OOV** | Out-of-Vocabulary |
| **TTS** | Text-to-Speech |
| **NLP** | Natural Language Processing |
| **Coda** | Posição final de sílaba |
| **Onset** | Posição inicial de sílaba |

---

**Uso na Apresentação**: Esse glossário unificado é consultado quando um termo específico é introduzido, ou visto como referência rápida ao final da apresentação.
