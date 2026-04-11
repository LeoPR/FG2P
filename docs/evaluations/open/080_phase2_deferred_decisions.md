ID: 080
Title: [v2.0 Fase 2.5] Meta-ticket — decisoes arquiteturais adiadas
Type: meta-research
Priority: Medium
Status: Open
Created: 2026-04-11

## Contexto

Durante a Fase 2 core (submodulo + scripts + rules basico), identificamos
varias decisoes arquiteturais importantes que nao precisam ser tomadas
agora, mas precisam ser registradas para nao esquecer.

A Fase 2 core foi deliberadamente **minima** — so infraestrutura basica
que funciona sem codigo novo. A Fase 2.5 (este ticket) e quando as
decisoes tomadas nesta conversa viram implementacao.

## Decisoes registradas

### 1. Manifest global para mapeamento path → tag BCP 47

**O que e**: Arquivo `dicts/manifest.yaml` gerado automaticamente pelo
`normalize_dicts.py` a cada execucao, mapeando cada arquivo de dados para
sua tag BCP 47 canonica.

**Exemplo**:
```yaml
entries:
  - path: pt/br/pt-br.tsv
    tag: pt-BR
    generated_from: dicts-workbench/rules/pt-br.rules.tsv
    generated_at: 2026-04-11T15:00:00Z
    source: ipa-dict
    mode: full
```

**Por que**: desacopla filesystem (minusculas, simples) de tag BCP 47
(canonica, com maiusculas). Permite renomear pastas sem quebrar tags.

**Implementacao**: adicionar geracao de manifest ao fim de `apply_group()`
em `normalize_dicts.py`. Formato: YAML (mais legivel que JSON para este
caso).

**Dependencia**: nenhuma — pode ser implementado qualquer hora.

### 2. Acao `tag` no `.rules.tsv`

**O que e**: Nova acao declarativa que define a tag BCP 47 associada
ao grupo:

```
pt-br    tag    pt-BR
```

**Por que**: centraliza a definicao de tag no proprio arquivo de regras,
evita duplicacao. O manifest e gerado a partir disso.

**Implementacao**: adicionar case `action == "tag"` em `load_groups()`.
Armazenar em `rules["tag"]`. Usar na geracao do manifest.

**Dependencia**: esta ligado a decisao 1 (manifest).

### 3. Acao `mode` (`full` vs `overlay`)

**O que e**: Flag que define se a saida contem **todas** as palavras
processadas (`full`, default) ou **apenas** as que sofreram alguma
transformacao (`overlay`).

```
pt-br-x-sp    src      ../output/pt-br.tsv
pt-br-x-sp    dst      ../output/pt-br-x-sp.tsv
pt-br-x-sp    mode     overlay                   ← so palavras que mudaram
pt-br-x-sp    regex    x                         ʁ
```

**Por que**: variantes regionais (SP, RJ) precisam de um arquivo pequeno
so com as diferencas fonologicas, nao uma duplicata inteira do canonical.

**Implementacao**: em `apply_group()`, ao escrever cada linha, verificar
se o modo e `overlay` e se a linha foi alterada. Se nao alterada e modo
e overlay, pular.

**Dependencia**: independente.

### 4. Estrutura hierarquica de `dicts/`

**O que e**: Reorganizacao de `dicts/` para acomodar multiplos idiomas
e variantes:

```
dicts/
├── manifest.yaml            gerado, mapeia path → tag
├── pt-br.tsv                v1.x INTOCADO (compat)
├── pt-br/
│   ├── pt-br.tsv            v2 gerado pelo pipeline
│   └── sp/
│       └── pt-br-sp.tsv     variante SP (overlay)
├── en-us/
│   └── en-us.tsv
└── tupinamba/
    └── tpn.tsv
```

**Por que**: permite crescer organicamente para multilingue. Manter
`dicts/pt-br.tsv` na raiz garante compat com v1.x ate a Fase 5.

**Decisao**: estrutura `pt-br/sp/` (2 niveis) em vez de `pt/br/sp/`
(3 niveis) para menos profundidade.

**Dependencia**: precisa do manifest (decisao 1) para mapear paths
hierarquicos para tags corretas. Esta decisao NAO pode ser executada
sem a Fase 5 do plano (migracao atomica de `dicts/pt-br.tsv` + os 9
arquivos hardcoded em `src/`).

### 5. Acao `append` (palavras manuais)

**O que e**: Permite que um grupo adicione palavras de um TSV manual
ao final do output, alem das regras regex:

```
pt-br-x-sp    src       ../output/pt-br.tsv
pt-br-x-sp    dst       ../output/pt-br-x-sp.tsv
pt-br-x-sp    regex     x                      ʁ
pt-br-x-sp    append    ../manual/pt-br-sp-coisas.tsv
```

**Por que**: regionalismos que nao existem no ipa-dict (ex: "trolebus",
giras locais) precisam ser curados manualmente.

**Implementacao**: 5-10 linhas em `apply_group()` — abre arquivo, adiciona
linhas ao fim da saida.

**Status**: baixa prioridade. Implementar so quando houver dados manuais
prontos para adicionar.

### 6. Dependencia implicita vs explicita entre grupos

**O que e**: Como detectar que `pt-br-x-sp` depende de `pt-br` (porque
le o output dele)?

**Opcao A — Implicita** (recomendada): o script analisa `src` de cada
grupo. Se aponta para um path dentro de `../output/`, identifica qual
grupo gera esse output e garante ordem topologica.

**Opcao B — Explicita**: acao `depends_on` no `.rules.tsv`:
```
pt-br-x-sp    depends_on    pt-br
```

**Decisao**: implementar **A primeiro** (mais simples, zero verbosidade).
Se falhar em casos de borda, adicionar B como fallback.

**Dependencia**: pode ser implementado quando houver 2+ grupos
encadeados (nao ha ainda).

### 7. Linha unica vs multi-linha em comentarios de `.rules.tsv`

**O que e**: Detalhe de formato — comentarios longos dentro do .rules.tsv.

**Decisao atual**: comentarios com `#` no inicio da linha, cada comentario
numa linha separada. Simples, sem ambiguidade.

**Nao ha decisao pendente**, so documentacao do padrao.

### 8. Substituicao do canonical v1.x pelo gerado

**O que e**: Em algum momento, o `dicts/pt-br.tsv` gerado pelo pipeline
pode substituir o canonical manual. Isso e a Fase 5 do plano.

**Pre-requisitos**:
- Pipeline reproduz byte-exact o canonical (ou divergencias muito
  pequenas e documentadas)
- Smoke test em `src/g2p.py` etc. passa com o gerado
- Decisao consciente: conservar vs substituir

**Dependencia**: Fase 5. Muito longe de hoje.

### 9. Hipotese: mono vs multi-idioma

**O que e**: questao aberta de pesquisa — sera que um modelo
multilingue com tag BCP 47 ao input supera um modelo monolingue PT-BR
no mesmo PER?

**Tese do usuario**: multilingue pode melhorar PER em palavras
estrangeirizadas (anglicismos) porque "herda" fonologia de outros idiomas.

**Nao e decisao arquitetural**, e experimento de v2.0 futuro. Registrado
aqui para nao esquecer.

## Criterios de aceite (Fase 2.5)

Para fechar este ticket, implementar (na ordem):

1. [ ] Manifest YAML (decisao 1)
2. [ ] Acao `tag` (decisao 2)
3. [ ] Acao `mode full`/`overlay` (decisao 3)
4. [ ] Dependencia implicita entre grupos (decisao 6, opcao A)
5. [ ] Validar com caso sintetico: 2 arquivos `.rules.tsv` encadeados

Decisoes 4, 5, 8, 9 sao **adiadas alem da Fase 2.5** — cada uma vira
seu proprio ticket quando chegar a hora.

## Dependencias

- Fase 2 core concluida (pipeline basico funciona)
- Fase 3 — validacao do pipeline atual (importante fazer antes de
  adicionar complexidade)

## Proximos passos (depois da Fase 2 core)

1. Rodar `normalize_dicts.py --group pt-br` na branch v2 (Fase 3)
2. Comparar output com `dicts/pt-br.tsv` via `audit_dict_diff.py`
3. Documentar divergencias conhecidas
4. Iniciar Fase 2.5 com implementacao das decisoes 1-3
