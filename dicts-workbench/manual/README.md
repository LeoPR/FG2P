# dicts-workbench/manual/

> **Status (2026-04-12)**: pasta de palavras manuais que **ainda nao sao
> consumidas pelo pipeline**. Aguarda implementacao da acao `append` no
> `normalize_dicts.py` (Fase 2.5, ticket 080).

## O que vive aqui

Palavras curadas manualmente que:
- Nao existem na fonte ipa-dict (adicoes)
- Existem mas estao erradas na fonte (correcoes — formato futuro)
- Foram levantadas via estudo linguistico dedicado

Cada idioma tem seu proprio arquivo de adicoes, seguindo o padrao
`<group>-additions.tsv`:

```
pt-br-additions.tsv       palavras manuais PT-BR
en-us-additions.tsv       futuro
tpn-additions.tsv         futuro
```

## Formato

Mesmo formato do canonical: TSV com 2 colunas separadas por tab:
```
palavra<TAB>ipa_tokens_separados_por_espaco
```

Exemplo:
```
hars	ˈ x a . ɾ s
porte	ˈ p o x . t e
```

## Regras de formato

As palavras aqui devem **ja estar no formato final** esperado pelo
canonical. Isso significa:

- Tokenizacao char-por-token com espacos (convencao v1.x)
- Normalizacao Unicode NFC (assumida pelo canonical v1.x)
- Uso de `ɡ` (U+0261) nao `g` (U+0067)
- Stress como token separado: `ˈ`
- Separador silabico: `.`

O pipeline **nao reprocessa** estas palavras (Variante 1 da Fase 2.5).
Se voce colocar `g` ASCII aqui, vai para o output assim. Por isso
validacao visual ao adicionar e importante.

## Historico

### 2026-04-12 — 4 palavras iniciais PT-BR

Extraidas do canonical `dicts/pt-br.tsv` durante a Fase 3 de validacao
do pipeline (ticket 079). Sao palavras que existem no canonical mas
nao no `ipa-dict/data/pt_BR.txt`:

| Palavra | IPA | Nota |
|---|---|---|
| `hars` | `ˈ x a . ɾ s` | Adicao manual (possivelmente anglicismo?) |
| `porte` | `ˈ p o x . t e` | Adicao manual |
| `portes` | `ˈ p o x . t e s` | Adicao manual (plural) |
| `teve` | `t e . v y` | Adicao manual (pode ser erro? verificar) |

**Origem das adicoes**: foram inseridas manualmente pelo autor durante
a preparacao do v1.x, razao nao documentada formalmente. Ticket de
auditoria futuro pode investigar se todas sao corretas ou se alguma e
erro de digitacao.

## Pendencias

- Implementar action `append` no `normalize_dicts.py` (ticket 080)
- Auditoria de correcao das 4 palavras (origem, motivo da adicao,
  correcao fonetica se aplicavel)
- Formato futuro: coluna de metadata (origem, data, revisao, categoria
  gramatical). Sera decidido quando o formato do arquivo for repensado
  (ver TOKENIZATION_LAYERS.md Camada 0).
