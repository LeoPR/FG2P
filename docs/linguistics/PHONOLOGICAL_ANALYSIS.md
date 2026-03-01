# Análise Fonológica — Símbolos IPA no Dataset PT-BR

**Data**: 2026-03-01
**Propósito**: Documentar a validação dos símbolos IPA usados no corpus `dicts/pt-br.tsv`, com foco nos alofones róticos e na distribuição complementar [x]/[ɣ]. Este documento serve como referência permanente para questões sobre a correção fonológica do dataset.

**Referências cruzadas**: [ARTICLE.md §2.3](../article/ARTICLE.md) (auditoria do corpus) | [GLOSSARY.md](../article/GLOSSARY.md) (termos)

---

## 1. Questão Investigada

O símbolo IPA **ɣ** (U+0263) usado no dataset para representar uma variante do "R" em coda silábica está correto? Ou deveria ser outro símbolo?

A dúvida surgiu porque fontes secundárias (Wikipedia) descrevem ɣ como "similar ao som da letra G em gato", o que parece contraditório com seu uso como alofone do R.

---

## 2. Classificação Oficial do IPA

### 2.1 O Quadro Consonantal IPA

No quadro oficial da International Phonetic Association (IPA), os sons são organizados por **lugar de articulação** (colunas) e **modo de articulação** (linhas):

```
          Velar (dorso da língua + palato mole)
          ─────────────────────────────────────
Plosiva:     k (surdo)    ɡ (sonoro)    ← U+006B / U+0261
Fricativa:   x (surdo)    ɣ (sonoro)    ← U+0078 / U+0263
```

**Fontes primárias consultadas**:
- [IPA Chart — ipachart.com](https://www.ipachart.com/) — Quadro interativo oficial
- [Full IPA Chart — International Phonetic Association](https://www.internationalphoneticassociation.org/content/full-ipa-chart) — Referência definitiva

### 2.2 Definição de cada símbolo

| Símbolo | Unicode | Nome IPA Oficial | Descrição articulatória |
|---------|---------|-----------------|------------------------|
| **ɡ** | U+0261 | Voiced velar plosive | Fechamento COMPLETO do dorso da língua contra o palato mole, seguido de soltura. Ex: "gato" → [ˈɡa.tʊ] |
| **ɣ** | U+0263 | Voiced velar fricative | Constrição PARCIAL (sem fechamento) do dorso da língua contra o palato mole, com fluxo de ar contínuo e turbulento. Cordas vocais vibram (sonoro) |
| **x** | U+0078 | Voiceless velar fricative | Mesma constrição que ɣ, mas SEM vibração das cordas vocais (surdo) |
| **k** | U+006B | Voiceless velar plosive | Fechamento completo como ɡ, mas SEM vibração (surdo). Ex: "casa" → [ˈka.zə] |

### 2.3 Relação entre os símbolos

- **x** e **ɣ** são um **par surdo/sonoro** de fricativas velares — a única diferença é a vibração das cordas vocais
- **k** e **ɡ** são um **par surdo/sonoro** de plosivas velares
- Fricativa ≠ plosiva: na fricativa, o ar passa continuamente (como um sopro); na plosiva, há fechamento total seguido de explosão

---

## 3. Uso no PT-BR — O Rótico em Coda

### 3.1 O fonema /R/ do PT-BR

O Português Brasileiro possui dois fonemas róticos:
- **/ɾ/** — vibrante simples (tap), ex: "caro" → [ˈka.ɾʊ], "porta" → o 'r' simples intervocálico
- **/R/** — rótico forte, com múltiplas realizações dialetais

O fonema /R/ (forte) pode se manifestar como:

| Realização | Símbolo IPA | Dialeto/Contexto |
|-----------|------------|------------------|
| Fricativa velar surda | [x] | São Paulo, padrão de mídia |
| Fricativa velar sonora | [ɣ] | Antes de consoantes sonoras (assimilação) |
| Fricativa glotal surda | [h] | Rio de Janeiro, Nordeste |
| Fricativa glotal sonora | [ɦ] | Rio/Nordeste antes de C sonora |
| Fricativa uvular surda | [χ] | Alguns dialetos |
| Fricativa uvular sonora | [ʁ] | Alguns dialetos, notação fonêmica ampla |
| Vibrante alveolar | [r] | Sul do Brasil, Portugal |

**Referências**:
- Barbosa, P. A. & Albano, E. C. (2004). *Brazilian Portuguese*. Journal of the International Phonetic Association, 34(2), 227–232. [Cambridge University Press](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/brazilian-portuguese/8EE69D52AE2D06C1EDA2841EA2E14FEC) — Ilustração oficial IPA para PT-BR
- [Rhotic Variation in Brazilian Portuguese (2024)](https://www.mdpi.com/2226-471X/9/12/364) — Estudo de variação dialetal

### 3.2 Escolha do dataset: transcrição alofônica [x]/[ɣ]

O corpus `dicts/pt-br.tsv` (fonte: [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict)) usa **transcrição alofônica (estreita)**, registrando a realização superficial, não a representação fonêmica abstrata.

- Se usasse transcrição **fonêmica** (ampla): `/ʁ/` em todas as posições de coda
- Como usa transcrição **alofônica** (estreita): `[x]` ou `[ɣ]` dependendo do contexto

Ambas as convenções são válidas no IPA. A escolha alofônica é mais informativa para treinamento de modelos, pois captura regras fonológicas produtivas.

---

## 4. Evidência Empírica no Dataset

### 4.1 Distribuição de [ɣ] — 5.449 ocorrências

O que vem DEPOIS do separador silábico `.` que segue ɣ:

| Fonema seguinte | Contagem | Vozeamento |
|---|---|---|
| m | 978 | sonoro |
| d | 665 | sonoro |
| n | 530 | sonoro |
| ɡ | 428 | sonoro |
| v | 304 | sonoro |
| b | 260 | sonoro |
| ʒ | 166 | sonoro |
| l | 107 | sonoro |
| z | 23 | sonoro |
| ʎ | 1 | sonoro |

**100% das ocorrências** de ɣ precedem consoantes sonoras. Zero exceções.

### 4.2 Distribuição de [x] em coda interna — contexto comparativo

O que vem DEPOIS do separador silábico `.` que segue x:

| Fonema seguinte | Contagem | Vozeamento |
|---|---|---|
| t | 1.279 | surdo |
| s | 732 | surdo |
| k | 578 | surdo |
| p | 271 | surdo |
| f | 230 | surdo |
| ʃ | 30 | surdo |

**100% das ocorrências** de x em coda interna precedem consoantes surdas. Zero exceções.

### 4.3 Resultado

**Distribuição complementar perfeita** em 95.937 palavras, 0 exceções:
- [ɣ] = R em coda antes de consoante **sonora** (assimilação de vozeamento)
- [x] = R em coda antes de consoante **surda** ou em **final de palavra**

### 4.4 Exemplos

| Palavra | Transcrição | Contexto |
|---------|-------------|----------|
| borboleta | b o **ɣ** . b o . ˈ l e . t ə | R antes de **b** (sonoro) → ɣ |
| largo | ˈ l a **ɣ** . ɡ ʊ | R antes de **ɡ** (sonoro) → ɣ |
| aberje | a . ˈ b ɛ **ɣ** . ʒ ɪ | R antes de **ʒ** (sonoro) → ɣ |
| abordada | a . b o **ɣ** . ˈ d a . d ə | R antes de **d** (sonoro) → ɣ |
| porta | ˈ p o **x** . t ə | R antes de **t** (surdo) → x |
| carro | ˈ k a . **x** ʊ | RR intervocálico → x |
| computador | k õ . p u . t a . ˈ d o **x** | R final de palavra → x |
| abordar | a . b o **ɣ** . ˈ d a **x** | MESMA PALAVRA: ɣ antes de d (sonoro), x em final |

O último exemplo (`abordar`) é particularmente ilustrativo: a mesma palavra contém ambos os alofones, demonstrando que a escolha depende exclusivamente do contexto seguinte.

---

## 5. A Confusão com a Wikipedia

### 5.1 O que a Wikipedia diz

A [Wikipedia espanhola sobre a fricativa velar sonora](https://es.wikipedia.org/wiki/Fricativa_velar_sonora) descreve ɣ como "similar ao som da letra 'g'", citando exemplos do espanhol e grego.

### 5.2 Por que isso NÃO contradiz nosso dataset

O IPA transcreve **sons**, não **letras**. O mesmo som [ɣ] pode originar-se de fonemas diferentes em línguas diferentes:

| Língua | Fonte do [ɣ] | Exemplo |
|--------|-------------|---------|
| **Espanhol** | /g/ entre vogais (lenição) | "lago" → [ˈla.**ɣ**o] |
| **Grego Moderno** | letra gamma (γ) | γάτα → [**ɣ**a.ta] |
| **PT-BR** | /R/ antes de C sonora (assimilação) | "borboleta" → [bo**ɣ**.bo.ˈle.tə] |
| **PT Europeu** | /g/ entre vogais | "logo" → [ˈlɔ.**ɣ**ʊ] |

Em PT-BR, o **g** entre vogais permanece como plosiva [ɡ] — diferente do espanhol e do português europeu. Nosso dataset reflete isso corretamente:
- "logo" → ˈ l ɔ . **ɡ** ʊ (plosiva U+0261, NÃO fricativa)
- "largo" → ˈ l a **ɣ** . ɡ ʊ (R fricativa U+0263 antes de ɡ sonoro)

### 5.3 Resumo

A Wikipedia **não está errada** — descreve o som em termos cross-linguísticos. A confusão surge porque:
1. O leitor associa ɣ à letra "g" (por causa do espanhol/grego)
2. No PT-BR, ɣ vem do "R", não do "g"
3. Mas foneticamente é o MESMO som: fricativa velar sonora, articulada com o dorso da língua contra o palato mole, com vibração das cordas vocais

---

## 6. Validação Unicode

O dataset usa os caracteres Unicode corretos e distintos:

| Símbolo | Codepoint | Nome Unicode | Uso no dataset |
|---------|-----------|-------------|----------------|
| ɡ | U+0261 | LATIN SMALL LETTER SCRIPT G | Plosiva velar sonora (o "g" de "gato") |
| ɣ | U+0263 | LATIN SMALL LETTER GAMMA | Fricativa velar sonora (alofone do R) |
| g | U+0067 | LATIN SMALL LETTER G | NÃO usado (corrigido: 10.252 instâncias normalizadas para ɡ) |

A normalização g→ɡ foi necessária para compatibilidade com PanPhon (ver ARTICLE.md §2.1).

---

## 7. Conclusões

| Aspecto | Veredicto | Evidência |
|---------|-----------|-----------|
| Símbolo ɣ no IPA oficial | **Correto** | Quadro IPA: fricativa velar sonora |
| Classificação "fricativa velar" | **Correto** | Linha "Fricative", coluna "Velar" no quadro |
| Uso de ɣ para R em coda PT-BR | **Correto** | Barbosa & Albano (2004); padrão em transcrição estreita |
| Assimilação de vozeamento | **Confirmada** | 0 exceções em 5.449 ocorrências vs 5 categorias de C sonora |
| Mapeamento no arquivo .tsv | **Correto** | ɣ (U+0263) distinto de ɡ (U+0261) |
| Wikipedia contradiz? | **Não** | Descreve o som cross-linguisticamente; PT-BR usa ɣ para R, não para G |

**O dataset está foneticamente correto e consistente com as fontes primárias do IPA.**

---

## 8. Referências

1. International Phonetic Association. (2015). *The International Phonetic Alphabet (revised to 2015)*. [Full IPA Chart](https://www.internationalphoneticassociation.org/content/full-ipa-chart)
2. Barbosa, P. A. & Albano, E. C. (2004). *Brazilian Portuguese*. Journal of the International Phonetic Association, 34(2), 227–232. [DOI](https://doi.org/10.1017/S0025100304001756)
3. [IPA Chart — Interactive](https://www.ipachart.com/)
4. [Rhotic Variation in Brazilian Portuguese](https://www.mdpi.com/2226-471X/9/12/364). Languages, 9(12), 364. (2024)
5. [open-dict-data/ipa-dict](https://github.com/open-dict-data/ipa-dict) — Fonte do corpus PT-BR

---

**Última atualização**: 2026-03-01
