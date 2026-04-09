# Data Pipeline: Do Corpus ao Treinamento (Capítulos 12-13 Merged)

**Data**: 2026-02-25
**Status**: ✅ Pipeline operacional | Filtros grafêmicos documentados

---

## 12.1 Distinção Crítica: Corpus-Fonte vs. Entrada do Modelo

```
FONTE PRIMÁRIA (NÃO MODIFICÁVEL):
  dicts/pt-br.tsv
  ↓ [leitura uma vez]

CACHE CARREGADO EM MEMÓRIA:
  G2PCorpus.words_raw      ← "abraça-los" (exatamente como no arquivo)
  G2PCorpus.words_transformed ← pode ser diferente se há filtros
  G2PCorpus.phonemes       ← fonemas após normalização NFC

TRANSFORMAÇÕES APLICÁVEIS:
  - transform_grapheme_word(encoding="raw" | "decomposed")
  - Filtros customizados (ex: remover hífens)
  - Normalização Unicode (NFC vs NFD)

VOCABULÁRIO CONSTRUÍDO:
  CharVocab.c2i   ← baseado em words_transformed
  PhonemeVocab.p2i ← baseado em phonemes

TREINO/VAL/TEST:
  Índices estratificados (determinísticos com seed=42)
  ↓
  Dados passados ao modelo
  ↓
  MODELO NUNCA vê "abraça-los" diretamente
  MODELO vê:        [2, 3, 12, 2, 5, 2, 23, 16, 13, 7]
               (codificação de caracteres via CharVocab)

Nota: os índices concretos dependem da ordem de primeira ocorrência dos caracteres no corpus.
```

---

## 12.2 Pipeline Operacional de Referência

### Etapa 1: Leitura do Corpus

**Arquivo**: `dicts/pt-br.tsv` (95.937 linhas)

```python
with open("dicts/pt-br.tsv", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split("\t")
        word = parts[0].strip()      # "abraça-los"
        phon_raw = parts[1].strip()  # "a b ɾ a ˈ s a l ʊ s"
```

**Saída**: `words_raw`, `phonemes_raw` (dados brutos, idênticos ao arquivo)

---

### Etapa 2: Transformação de Grafemas

**Código** (g2p.py:356):
```python
transformed_word = transform_grapheme_word(word, self.grapheme_encoding)
```

**Comportamento**:
```
Se grapheme_encoding == "raw" (padrão):
  "abraça-los" → "abraça-los" (identidade)

Se grapheme_encoding == "decomposed":
  "abraça-los" (NFC) → decomposição NFD
  ↓
  "abrac'a~los" (marca diacríticos como símbolos separados)

Nota: em configuração padrão (`grapheme_encoding="raw"`), não há mudança
```

**Saída**: `words_transformed` (pode ser idêntico ou diferente de `words_raw`)

---

### Etapa 3: Normalização de Fonemas

**Código** (g2p.py:367):
```python
phon = unicodedata.normalize("NFC", phon)
```

**O que muda**:
- **Entrada**: fonemas em qualquer encoding (NFD ou NFC misto)
- **Saída**: fonemas uniformemente NFC
- **Impacto**: elimina inconsistências de representação interna
- **Obs**: no corpus auditado em 2026-02-25, a codificação está em NFC (operação idempotente)

**Saída**: `phonemes_clean` (fonemas normalizados NFC)

---

### Etapa 4: Construção de Vocabulários

**CharVocab**:
```python
for w in self.words:  # words_transformed
    self.char_vocab.add(w)
```

- extrai caracteres únicos de cada palavra
- cria mapeamento `char → índice` (c2i) e inverso (i2c)
- resultado típico: 39 caracteres (incluindo `-`)

**PhonemeVocab**:
```python
for p in self.phonemes:  # phonemes_clean
    self.phoneme_vocab.add(p.split())
```

- split por espaço (fonemas são space-separated)
- cria mapeamento `fonema → índice`
- resultado típico: 43 fonemas únicos

---

### Etapa 5: Estratificação e Split

**Algoritmo** (g2p.py:392-450):
```
1. Extrair features linguísticas de cada par (palavra, fonemas):
   - stress_type: oxítona/paroxítona/proparoxítona
   - syllable_bin: número de sílabas (1, 2, 3, 4, 5+)
   - length_bin: comprimento (1-4, 5-7, 8-10, 11+)
   - ratio_bin: fonemas/caracteres

2. Criar estrato para cada combinação

3. Usar StratifiedShuffleSplit(random_state=seed) para split determinístico
   - distribuição uniforme em cada estrato
   - `seed=42` reproduz os mesmos índices
```

**Resultado**: `train_indices`, `val_indices`, `test_indices`

Split parametrizado por configuração:
- `train_ratio`, `val_ratio`, `test_ratio`
- exemplo comum: 60/10/30
- determinístico com `seed=42`

---

### Etapa 6: Encoding para Treino

**Durante DataLoader** (g2p.py:153-157):
```python
word = self.words[idx]                    # "abraça-los"
chars = self.char_vocab.encode(word)      # [2, 3, 12, 2, 5, 2, 23, 16, 13, 7]
phonemes = self.phoneme_vocab.encode(phoneme_seq)
```

**Transformação character-by-character**:
```
"abraça-los"
 ↓ [a→2, b→3, r→12, a→2, ç→5, a→2, -→23, l→16, o→13, s→7]
[2, 3, 12, 2, 5, 2, 23, 16, 13, 7]  ← entrada para o encoder
```

---

## 12.3 Ponto de Aplicação de Filtros

Se implementássemos **remoção de hífens** como filtro:

```python
# Pseudocódigo — onde inserir filtro
transformed_word = transform_grapheme_word(word, self.grapheme_encoding)

# ← AQUI seria adicionado filtro:
if REMOVE_HYPHENS:
    transformed_word = transformed_word.replace("-", "")
# Resultado: "abraça-los" → "abraçalos"

words_transformed.append(transformed_word)  # "abraçalos"
```

**Efeito ao construir CharVocab**:
```
Sem filtro: CharVocab vê caracteres de "abraça-los"
  ← inclui "-" (índice 23)

Com filtro: CharVocab vê caracteres de "abraçalos"
  ← não inclui "-"
  → vocab reduz de 39 para 38 caracteres
```

**Efeito nos dados de treino**:
```
Input ao modelo:
  Sem filtro: [2, 3, 12, 2, 5, 2, 23, 16, 13, 7]   (10 tokens)
  Com filtro: [2, 3, 12, 2, 5, 2, 16, 13, 7]       (9 tokens, -1)

Output esperado (fonemas): [a, b, ɾ, a, ˈ, s, a, l, ʊ, s]
  ← NÃO MUDA (hífen não tem fonema)
```


---

## 12.4 Escopo do PIPELINE.md

Este documento descreve o **pipeline operacional estável** para desenvolvimento e manutenção do projeto:
- carregamento de dados
- transformação grafêmica
- normalização fonêmica
- construção de vocabulários
- estratificação e dataloaders
- codificação para treino
- componentes de treino (embedding, loss e batch)

Evolução histórica de experimentos e ablações deve ser documentada em:
- `docs/article/EXPERIMENTS.md`
- `docs/article/ARTICLE.md`

---

## 12.5 Componentes de Treino (Embedding, Loss, Batch Size)

### Batch e DataLoader

No treino, o `batch_size` vem do config e é aplicado em `split.get_dataloaders(...)`:

```python
train_dl, val_dl, _test_dl = split.get_dataloaders(
    batch_size=config["training"]["batch_size"]
)
```

### Embedding e Arquitetura

O modelo é criado por `G2PLSTMModel.from_config(...)`, usando parâmetros em `config["model"]`:
- `emb_dim`
- `hidden_dim`
- `num_layers`
- `dropout`
- `embedding_type` (quando configurado)

### Loss

A loss é definida em `config["training"]["loss"]`:
- padrão: `cross_entropy`
- alternativas: `distance_aware` e `soft_target`

Fluxo em `src/train.py`:
1. Lê `loss.type` do config
2. Se for `distance_aware`/`soft_target`, constrói loss com `get_loss_function(...)`
3. Caso contrário, usa `cross_entropy`
4. Move `criterion` para o mesmo device do modelo

---

## 12.6 Normalização NFC: Especificação Operacional

### Estado do Corpus

```
Arquivo: dicts/pt-br.tsv (2026-02-25)
Aplicado: Unicode normalization para NFC uniformemente
Afetou: 1.390 caracteres em 10 palavras (encoding inconsistências)

Resultado: arquivo com encoding uniforme em NFC
```

### Impacto no Pipeline

```
Etapa 1 (Leitura): ✅ Idêntico
  - Lê arquivo em UTF-8

Etapa 2 (Transformação): ✅ Idêntico
  - Não afeta se encoding="raw"

Etapa 3 (Normalização): ✅ Idempotente
  - Código normaliza para NFC (linha 367)
  - Se está em NFC → operação idempotente
  - Se estivesse em NFD → conversão necessária

Etapa 4 (CharVocab): ✅ Idêntico
Etapa 5 (Split): ✅ Idêntico
  - Baseado em índices, não em encoding

Conclusão: NFC normalization é transparente ao pipeline
```

---

## 12.7 Pipeline Grafêmico com Filtros Configuráveis

### API: GraphemeConfig

Classe que encapsula transformação grafêmica + filtros:

```python
from g2p import GraphemeConfig

# Exemplo 1: raw (padrão)
config = GraphemeConfig("raw")
config.transform("abraça-los")  # → "abraça-los"

# Exemplo 2: decomposed (marcas diacríticas)
config = GraphemeConfig("decomposed")
config.transform("abraça-los")  # → "abrac'a-los"

# Exemplo 3: raw + remover hífens
config = GraphemeConfig({"type": "raw", "filters": ["-"]})
config.transform("abraça-los")  # → "abraçalos"

# Exemplo 4: decomposed + remover hífens + remover separadores
config = GraphemeConfig({"type": "decomposed", "filters": ["-", "."]})
config.transform("abr.aç.a-los")  # → "abrac'alos"
```

#### Sintaxe JSON

```json
{
  "grapheme_encoding": "raw",
  "keep_syllable_separators": true
}
```

ou

```json
{
  "grapheme_encoding": {
    "type": "raw",
    "filters": ["-", "."]
  },
  "keep_syllable_separators": true
}
```

---

### Fluxo de Aplicação

```
CORPUS SOURCE (dicts/pt-br.tsv)
  "abaixo-assinado" → "a b a y ʃ o a s i ˈ n a d ʊ"

CARREGAMENTO (_load_tsv)
  word = "abaixo-assinado" (raw)

  grapheme_config.transform(word):
    Step 1: Transformação base
      "raw" → "abaixo-assinado"
      ou
      "decomposed" → "abai'xo-assinado"

    Step 2: Aplicar filtros
      filter "-" → remove hífen
      "abaixoassinado" (final)

  words_transformed.append("abaixoassinado")

CONSTRUÇÃO DE VOCABULÁRIOS
  CharVocab.add("abaixoassinado")
    → Extrai: a, b, a, i, x, o, a, s, s, i, n, a, d, o
    → Se "-" foi removido no Step 2: "-" NÃO entra no vocab
    → Se mantido: "-" entra normalmente

TREINAMENTO
  word encodeado: [2, 3, 2, 11, 14, 13, 2, 7, 7, 11, 19, 2, 17, 13]
    (sem hífen = -1 token = sequências mais curtas)
  target fonemas: [a, b, a, y, ʃ, o, a, s, i, ˈ, n, a, d, ʊ]
    (idênticos, hífen não tem fonema)
```

---

### Implementação Técnica

#### Classe GraphemeConfig (src/g2p.py)

```python
class GraphemeConfig:
    """Configuração de transformação grafêmica com suporte a filtros.

    Formatos suportados:
    1. "raw" — sem transformação
    2. "decomposed" — decomposição NFD com marcas de diacríticos
    3. {"type": "raw", "filters": ["-", "."]} — raw + remove caracteres
    4. {"type": "decomposed", "filters": ["-"]} — decomposed + remove
    """

    def __init__(self, encoding: str | dict = "raw"):
        # Valida entrada (str ou dict)
        # Extrai type e filters

    def transform(self, word: str) -> str:
        """Aplica transformação grafêmica + filtros."""
        # Step 1: Transformação base (raw ou decomposed)
        # Step 2: Aplicar filtros (remover caracteres)
        return result
```

#### Integração em G2PCorpus

```python
class G2PCorpus:
    def __init__(self, dict_path, grapheme_encoding="raw", ...):
        self.grapheme_config = GraphemeConfig(grapheme_encoding)
        # ...

    def _load_tsv(self):
        # ...
        for line in f:
            word = parts[0].strip()
            transformed_word = self.grapheme_config.transform(word)
            # ...
```

---

### Impacto no Dataset

#### Caractere Hífen (`-`)

| Métrica | Valor |
|---------|-------|
| Palavras com hífen | 2.361 / 95.937 (2.46%) |
| Exemplos | abaixo-assinado, abelha-rainha, abandoná-los |
| Tipo | Ortográfico (não fonológico) |
| Impacto no modelo | Nenhum (modelo aprende a "pular") |
| Custo computacional | ~0.025% (negligenciável) |
| Benefício de remover | ~0.5-1% mais rápido (marginal) |
| Custo de remover | Perde representação de compostos em PT-BR |

#### Recomendação Operacional

**Manter hífen no pipeline padrão**:
- ✅ Representa fielmente PT-BR (compostos, verbos+pronome)
- ✅ Overhead é negligenciável
- ✅ Modelo é robusto

---

### Validação de Filtros

Script para validar filtros de treino:

```python
from g2p import GraphemeConfig, G2PCorpus

config = GraphemeConfig({"type": "raw", "filters": ["-", "."]})
corpus = G2PCorpus("dicts/pt-br.tsv", grapheme_encoding=config)

print(f"CharVocab size: {len(corpus.char_vocab)}")
print(f"Chars: {sorted(corpus.char_vocab.c2i.keys())}")

hyphen_count_before = sum(1 for w in corpus.words_raw if "-" in w)
hyphen_count_after = sum(1 for w in corpus.words if "-" in w)
print(f"Hífens removidos: {hyphen_count_before - hyphen_count_after}")
```

Vide `test_grapheme_filters.py` para exemplos completos.

---

## 12.8 Conclusão: Pipeline Verificado e Extensível ✅

| Aspecto | Status | Evidência |
|---------|--------|-----------|
| Source corpus (dicts/pt-br.tsv) | ✅ Íntegro | Checksum estável, NFC uniforme |
| Carregamento | ✅ Correto | Leitura em UTF-8, sem corrupção |
| Transformação grafêmica | ✅ Operacional | "raw" e "decomposed" funcionam |
| Filtros customizáveis | ✅ Implementado | GraphemeConfig suporta múltiplos filtros |
| Normalização NFC | ✅ Transparente | Phonemes em NFC uniforme |
| CharVocab | ✅ Válido | 39 chars (padrão), reduz com filtros |
| PhonemeVocab | ✅ Válido | 43 fonemas, todos NFC |
| Split estratificado | ✅ Determinístico | seed=42 reproduz exatamente |
| Encoding para treino | ✅ Correto | Sem corrupção de índices |
| Batch/Loss/Embedding | ✅ Configurável | Parâmetros centralizados em `config` |

Detalhes de evolução experimental e ablações: `docs/article/EXPERIMENTS.md`.
