# IPA Reference — Auditoria de Símbolos do FG2P

**Princípio científico**: a referência normativa é o **IPA oficial**.

**Objetivo deste arquivo**: documentar, com base no corpus atual e no modelo de referência, quais símbolos estão:
- alinhados ao IPA oficial;
- herdados do corpus mas ainda exigindo justificativa explícita;
- reconhecidamente problemáticos ou pendentes de auditoria mais profunda.

**Fontes verificadas**:
- [dicts/pt-br.tsv](dicts/pt-br.tsv)
- predições do Exp104d
- [data/phoneme_map.json](data/phoneme_map.json)

**Resultado da auditoria**:
- `ɾ`, `x` e `ɣ` são símbolos IPA oficiais e podem ser mantidos, desde que descritos corretamente;
- `g` ASCII é um defeito de dado — forma correta é `ɡ`;
- `y`, `ỹ` e `ʊ̃` têm interpretação fonética em aberto; ver tabela de símbolos e Dúvidas Comuns.

---

## Interpretando a Saída

### Exemplo: `computador`

Saída FG2P:
```
k õ . p u . t a . ˈ d o x
```

Leitura token por token:
```
k     = oclusiva velar surda          ("c" de casa)
õ     = vogal nasal                   (vogal nasalizada da primeira sílaba)
.     = separador de sílaba
p     = oclusiva bilabial surda       ("p" de pato)
u     = vogal posterior fechada       ("u" de tudo)
t     = oclusiva alveolar surda       ("t" de tato)
a     = vogal aberta                  ("a" de casa)
ˈ     = marcador de sílaba tônica
d     = oclusiva alveolar sonora      ("d" de dado)
o     = vogal posterior fechada       ("o" de bolo)
x     = rótico forte de coda/final    ("r" final em computador, carta)
```

**Leitura corrida**: `k õ . p u . t a . ˈ d o x`

---

## Símbolos Usados no Corpus Atual

**Leitura da tabela**:
- `IPA OK`: símbolo compatível com o IPA oficial e com o uso documentado aqui;
- `Questão aberta`: símbolo presente no corpus cuja interpretação IPA precisa de justificativa formal;
- `Defeito de dado`: símbolo identificado como forma incorreta frente ao IPA.

### Vogais Orais

| Símbolo | Status | Valor IPA oficial | Exemplo confirmado | Observação |
|---------|--------|-------------------|--------------------|------------|
| a | IPA OK | vogal aberta | `casa` → `ˈ k a . z ə` | compatível com o corpus |
| ə | IPA OK | schwa / vogal média central reduzida | `casa` → `ˈ k a . z ə` | usada em final átono no corpus |
| ɛ | IPA OK | vogal média anterior aberta | contraste com `e` | compatível com o corpus |
| e | IPA OK | vogal média anterior fechada | `borboleta` → `b o ɣ . b o . ˈ l e . t ə` | compatível com o corpus |
| ɪ | IPA OK | vogal quase-fechada anterior | aparece em finais átonos | precisa ser interpretada como escolha fonética do corpus, não erro de símbolo |
| i | IPA OK | vogal fechada anterior | `igreja` → `i . ˈ ɡ ɾ e . ʒ ə` | compatível com o corpus |
| ɔ | IPA OK | vogal média posterior aberta | contraste com `o` | compatível com o corpus |
| o | IPA OK | vogal média posterior fechada | `caso` → `ˈ k a . z ʊ` | compatível com o corpus |
| ʊ | IPA OK | vogal quase-fechada posterior | `caso` → `ˈ k a . z ʊ` | símbolo IPA oficial; interpretação fonológica depende do corpus |
| u | IPA OK | vogal fechada posterior | `computador` → `k õ . p u . t a . ˈ d o x` | compatível com o corpus |

### Vogais Nasais e Ditongos Nasais

| Símbolo | Status | Valor IPA oficial | Exemplo confirmado | Observação |
|---------|--------|-------------------|--------------------|------------|
| ã | IPA OK | `a` nasalizado | `pão` → `ˈ p ã ʊ̃` | compatível com o corpus |
| ẽ | IPA OK | `e` nasalizado | ocorre no corpus | compatível com o corpus |
| ĩ | IPA OK | `i` nasalizado | `ninho` → `ˈ n ĩ . ɲ ʊ` | compatível com o corpus |
| õ | IPA OK | `o` nasalizado | `computador` → `k õ . p u . t a . ˈ d o x` | compatível com o corpus |
| ũ | IPA OK | `u` nasalizado | ocorre no corpus | compatível com o corpus |
| ʊ̃ | Questão aberta | `ʊ` nasalizado | `pão` → `ˈ p ã ʊ̃` | símbolo IPA possível, mas seu papel como glide de ditongo final precisa de justificativa fonética explícita |
| ỹ | Questão aberta | `y` nasalizado | `mãe` → `ˈ m ã ỹ` | se estiver representando glide palatal nasal, o alinhamento ao IPA precisa de auditoria dedicada |

### Consoantes e Glides

| Símbolo | Status | Valor IPA oficial | Exemplo confirmado | Observação |
|---------|--------|-------------------|--------------------|------------|
| p, b, t, d, k, ɡ | IPA OK | oclusivas | `casa`, `computador`, `jogo` | `ɡ` é a forma IPA correta |
| g | Defeito de dado | não é o símbolo IPA correto para a oclusiva velar sonora | aparece em dados legados | deve ser normalizado para `ɡ` |
| f, v, s, z | IPA OK | fricativas labiais/alveolares | `casa`, `ação` | compatível com o corpus |
| ʃ | IPA OK | fricativa pós-alveolar surda | contextos como `ch` e palatalização | compatível com o corpus |
| ʒ | IPA OK | fricativa pós-alveolar sonora | `jogo` → `ˈ ʒ o . ɡ ʊ` | compatível com o corpus |
| m, n, ɲ | IPA OK | nasais | `ninho` → `ˈ n ĩ . ɲ ʊ` | `ɲ` corresponde a `nh` |
| l | IPA OK | lateral alveolar | `borboleta` | compatível com o corpus |
| ʎ | IPA OK | lateral palatal | `ilha`, `milha` | compatível com o corpus |
| ɾ | IPA OK | tepe alveolar | `caro` → `ˈ k a . ɾ ʊ` | símbolo e descrição corretos |
| x | IPA OK | fricativa velar surda | `carta` → `ˈ k a x . t ə`; `computador` → `... d o x` | símbolo IPA oficial; aqui usado como realização do rótico forte de coda |
| ɣ | IPA OK | fricativa velar sonora | `carga` → `ˈ k a ɣ . ɡ ə`; `borboleta` → `b o ɣ . b o . ˈ l e . t ə` | símbolo IPA oficial; aqui usado antes de consoante vozeada |
| w | IPA OK | aproximante labiovelar | aparece em ditongos e vocalização de `l` em coda | símbolo IPA oficial |
| y | Questão aberta | vogal fechada anterior arredondada | `maio` → `ˈ m a y . ʊ` | se o corpus pretende representar glide palatal, o IPA esperado seria `j` ou outra solução explicitamente justificada |

### Marcadores Estruturais

| Símbolo | Função | Observação |
|---------|--------|------------|
| `ˈ` | marca a sílaba tônica | aparece antes da sílaba acentuada |
| `.` | separa sílabas | é estrutural, não é fonema |

---

## Padrões Importantes no Projeto

### 1. Róticos em PT-BR no Corpus Atual

| Contexto | Símbolo | Exemplo confirmado |
|----------|---------|--------------------|
| entre vogais / ataque simples | `ɾ` | `caro` → `ˈ k a . ɾ ʊ` |
| coda antes de consoante surda | `x` | `carta` → `ˈ k a x . t ə` |
| coda antes de consoante vozeada | `ɣ` | `carga` → `ˈ k a ɣ . ɡ ə` |
| final de palavra | `x` | `computador` → `k õ . p u . t a . ˈ d o x` |

### 2. `lh` e `nh`

| Grafia | Saída | Exemplo confirmado |
|--------|-------|--------------------|
| `lh` | `ʎ` | `milha` → `ˈ m i . ʎ ə` |
| `nh` | `ɲ` | `ninho` → `ˈ n ĩ . ɲ ʊ` |

### 3. Final Átono

| Padrão | Exemplo confirmado | Observação |
|--------|--------------------|------------|
| `a` final → `ə` | `casa` → `ˈ k a . z ə` | redução final recorrente no corpus |
| `o` final → `ʊ` | `caso` → `ˈ k a . z ʊ` | alta posterior reduzida |
| `e` final → `ɪ` | frequente no corpus | alta anterior reduzida |

### 4. Terminação `-ão`

| Palavra | Saída confirmada | Leitura prática |
|---------|------------------|-----------------|
| `pão` | `ˈ p ã ʊ̃` | `ã` + fechamento nasal |
| `ação` | `a . ˈ s ã ʊ̃` | mesmo padrão nasal final |
| `mãe` | `ˈ m ã ỹ` | ditongo nasal anterior |

---

## Exemplos Completos

### `borboleta`

```
b o ɣ . b o . ˈ l e . t ə
```

Leitura:
- `b o` = primeira sílaba
- `ɣ` = rótico de coda antes de `b`
- `.` = quebra silábica
- `ˈ l e` = sílaba tônica
- `t ə` = final átono reduzido

**Observação**: a forma acima é a usada no dicionário do projeto e no Exp104d, com separadores silábicos e marcador de tonicidade.

### `casa`

```
ˈ k a . z ə
```

Leitura:
- `ˈ` marca a sílaba tônica inicial
- `k a` = primeira sílaba
- `z` = `s` entre vogais
- `ə` = `a` final reduzido

### `maio`

```
ˈ m a y . ʊ
```

Leitura:
- `ˈ m a` = sílaba tônica inicial
- `y` = glide palatal neste contexto; interpretação fonética em aberto — ver tabela de símbolos
- `ʊ` = vogal final posterior reduzida

---

## Dúvidas Abertas e Ações

Mantivemos aqui apenas as observações já justificadas e corrigidas; as demais questões foram convertidas em tickets de auditoria em `docs/evaluations/open/` para investigação e documentação padronizada.

- `g` no lugar de `ɡ`: **defeito de dado — corrigido**. (Normalização aplicada aos dicionários e predições históricas.)

- Questões convertidas em tickets (abrir para evidências e decisões):
	- `y` vs glide palatal / possível uso de `j` — ver ticket ID 030.
	- `ỹ` e `ʊ̃` em ditongos nasais — ver ticket ID 031.

Observação: sinais estruturais como `.` (separador silábico) e `ˈ` (marca de tônica) permanecem documentados como não-fonemas.

---

**Última atualização**: 2026-03-17
**Encoding**: UTF-8
**Fonte recomendada**: DejaVu Sans, Charis SIL, ou Liberation Sans

