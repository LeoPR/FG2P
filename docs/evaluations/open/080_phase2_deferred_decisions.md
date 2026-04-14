ID: 080
Title: [v2.0 Fase 2.5] Meta-ticket — decisoes arquiteturais adiadas
Type: meta-research
Priority: Medium
Status: Open (atualizado 2026-04-12)
Created: 2026-04-11

## Contexto

Durante a Fase 2 core (submodulo + scripts + rules basico), identificamos
varias decisoes arquiteturais importantes que nao precisam ser tomadas
agora, mas precisam ser registradas para nao esquecer.

A Fase 2 core foi deliberadamente **minima** — so infraestrutura basica
que funciona sem codigo novo. A Fase 2.5 (este ticket) e quando as
decisoes tomadas nesta conversa viram implementacao.

## Atualizacao 2026-04-12: ciclo de diagnostico do pipeline

Apos rodar o pipeline e comparar com o canonical, fechamos algumas
decisoes que estavam abertas. Ver secao "Decisoes fechadas" abaixo.

Novas questoes criticas sobre tokenizacao foram levantadas e registradas
em documento dedicado: `docs/linguistics/TOKENIZATION_LAYERS.md`. Este
documento mapeia 7 camadas entre arquivo fisico e gradiente da DA Loss,
cada uma com perguntas abertas. **Nao confundir com este ticket** — o
TOKENIZATION_LAYERS e pesquisa linguistica/arquitetural profunda;
este ticket 080 e sobre implementacao incremental do pipeline.

---

## Decisoes FECHADAS (2026-04-12)

Estas decisoes foram tomadas apos o ciclo de diagnostico e **nao
precisam mais de discussao**. Quando a Fase 2.5 for implementada,
elas servem de guia direto.

### F1. Acao `append` — Variante 1 (manuais ja formatadas)

**Decidido**: a acao `append` no `.rules.tsv` concatena um TSV manual
ao **fim do output**, sem reprocessar com os regex. As palavras manuais
ja devem estar no formato final canonical.

**Motivacao**: simplicidade. Se um dia precisarmos de Variante 2
(manuais cruas reprocessadas), evoluimos depois.

**Sintaxe proposta**:
```
pt-br    src       ../sources/ipa-dict/data/pt_BR.txt
pt-br    dst       ../output/pt-br.tsv
pt-br    regex     ...
pt-br    append    ../manual/pt-br-additions.tsv
```

**Arquivo manual ja criado**: `dicts-workbench/manual/pt-br-additions.tsv`
com 4 palavras (`hars`, `porte`, `portes`, `teve`). Nao e consumido ainda.

### F2. Funcao `_expand_escapes` no script

**Decidido**: adicionar suporte a escape sequences `\s`, `\t`, `\n` nos
valores das regras regex. Isso permite representar caracteres whitespace
de forma visivel no `.rules.tsv`, sem depender de trailing whitespace
que editores removem.

**Motivacao**: descoberto que a regra de tokenizacao atual
`(\S)(?=[^\s\u0300-\u036f])	\1` tem `\1` sem espaco, entao nao tokeniza.
Historicamente o espaco trailing deve ter sido removido por algum editor.
Usar `\1\s` resolve de forma explicita e robusta.

**Escopo da mudanca**: ~10 linhas em `normalize_dicts.py`. Funcao
auxiliar `_expand_escapes(value)` chamada antes de usar value1/value2
em acao `regex`.

**Preserva**: regex backreferences (`\1`, `\2`, etc.) devem passar
intactas para `re.sub()`. Apenas `\s`, `\t`, `\n` sao expandidos.

### F3. Fix de encoding no `audit_dict_diff.py`

**Decidido**: adicionar `sys.stdout.reconfigure(encoding='utf-8',
errors='replace')` no topo do script para funcionar no Windows (cp1252
default). E padrao do projeto — 4 outros scripts em `src/` ja usam.

**Escopo**: 2 linhas. Cross-OS seguro.

### F4. Convencao de tokenizacao: token = segmento IPA atomico

**Decidido**: o FG2P usa 1 token por segmento IPA. Digrafos como `tʃ`,
`dʒ` sao **2 tokens** separados. Precomposed como `ã` sao 1 token.
Diacriticos combinantes (NFD) ficam agrupados via regra regex com
lookahead `[^\s\u0300-\u036f]`.

**Motivacao**:
- PanPhon mede vetores de features por segmento atomico — e a base
  do embedding e da DA Loss
- Gestos articulatorios (composicoes) sao responsabilidade de TTS/ASR,
  nao de G2P
- Canonical v1.x ja usa essa convencao (`abadia` -> `a . b a . ˈ d ʒ i . ə`)

**Referencia linguistica**: ver `docs/linguistics/TOKENIZATION_LAYERS.md`
secao H1 para discussao mais profunda sobre gesto vs segmento atomico.

**Nao confundir**: esta e a decisao operacional para Fase 2.5. Discussoes
mais profundas sobre camadas, formatos de arquivo, Unicode normalization,
etc. ficam no TOKENIZATION_LAYERS.md — aquilo e pesquisa de longo prazo.

### F5. Normalizacao NFC no pipeline (Fase 3, implementada 2026-04-13)

**Decidido**: action `nfc true` no `.rules.tsv` normaliza o campo IPA
para NFC apos os regex, antes de escrever o output.

**Motivacao**: a fonte ipa-dict usa NFD (decomposed), o canonical v1.x
e o `g2p.py` usam NFC. Alinhar o arquivo em disco com o modelo elimina
uma transformacao implicita e fecha 42.293 diferencas de auditoria.

**Achados empiricos**:
- PanPhon aceita NFC e NFD e produz vetores 24D **identicos**
- PanPhon internamente converte para NFD (transparente ao codigo)
- Nasalizacao ja e feature `nas` no PanPhon 24D (nao precisa de 25D)
- `ɛ̃` (sem precomposed NFC) e reconhecido como 1 segmento pelo PanPhon

**Implementacao**: 3 linhas em `normalize_dicts.py`:
- `load_groups`: nova action `nfc` → `rules["nfc"] = True`
- `apply_group`: `if apply_nfc: normalized = unicodedata.normalize("NFC")`

**Resultado**: auditoria pipeline vs canonical: `unicode_equivalent = 0`,
`real_content = 2` (uden/uder, correcoes documentadas).

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

Para fechar este ticket, implementar (na ordem sugerida):

### Prioridade 1 — Desbloqueio do diagnostico (fechadas em 2026-04-12)

1. [ ] **F2**: funcao `_expand_escapes` no `normalize_dicts.py`
   - Expande `\s`, `\t`, `\n` em value1/value2 de acoes `regex`
   - Preserva backreferences regex (`\1`, `\2`, etc.) intactas
2. [ ] **F2 (continuacao)**: ajustar `dicts-workbench/rules/pt-br.rules.tsv`
   - Trocar `\1` por `\1\s` na regra de tokenizacao
3. [ ] **F3**: fix encoding `sys.stdout.reconfigure` em `audit_dict_diff.py`
4. [ ] **Validacao P1**: rerodar o pipeline pt-br e comparar com canonical
   - Esperado: cair de 95.933 linhas `real_content` para valor pequeno
   - Numero esperado: 0 linhas divergentes em conteudo real, 4 `only_in_right`

### Prioridade 2 — Acao append (decisao fechada 2026-04-12)

5. [ ] **F1**: acao `append` no `normalize_dicts.py`
   - Variante 1: concatena TSV manual ao fim do output, sem regex
6. [ ] **F1 (continuacao)**: adicionar linha `append ../manual/pt-br-additions.tsv`
   ao `pt-br.rules.tsv`
7. [ ] **Validacao P2**: rerodar pipeline e verificar que as 4 palavras
   aparecem no output; `only_in_right` cai para 0

### Prioridade 3 — Manifest e tags (Fase 4, implementada 2026-04-14)

8. [x] Manifest YAML gerado (`dicts/manifest.yaml`)
9. [x] Acao `tag` (BCP 47 por grupo)
10. [x] Acao `mode full`/`overlay`
11. [x] Dependencia implicita entre grupos (topological sort)
12. [ ] Validar com caso sintetico: 2 arquivos `.rules.tsv` encadeados (Fase 6)

### Fora da Fase 2.5 (ticket proprio quando chegar)

Decisoes originais 4, 5, 8, 9 permanecem adiadas para apos Fase 2.5:
- Estrutura hierarquica de dicts (decisao 4)
- Correcoes via lookup (decisao 5)
- Substituicao do canonical v1.x pelo gerado (decisao 8)
- Experimentos mono vs multi-idioma (decisao 9)

## Dependencias

- Fase 2 core concluida (pipeline basico funciona)
- Fase 3 — validacao do pipeline atual (importante fazer antes de
  adicionar complexidade)

## Proximos passos (depois da Fase 2 core)

1. Rodar `normalize_dicts.py --group pt-br` na branch v2 (Fase 3)
2. Comparar output com `dicts/pt-br.tsv` via `audit_dict_diff.py`
3. Documentar divergencias conhecidas
4. Iniciar Fase 2.5 com implementacao das decisoes 1-3
