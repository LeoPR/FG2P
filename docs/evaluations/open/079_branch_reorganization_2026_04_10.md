ID: 079
Title: Reorganizacao de branches para inicio limpo da v2.0 (Fase 1)
Type: maintenance / infrastructure
Priority: High
Status: In progress (Fase 1 concluida)
Date: 2026-04-10

## Contexto

Apos descobrir que existia uma branch `dev/v2.0` antiga com trabalho de
exploracao (~3 semanas, tickets 034-056 em duplicata com tickets criados
hoje em main), foi decidido adotar o **Caminho C** (nova branch limpa)
em vez de reconciliar as duas linhas de trabalho.

Motivacao para o Caminho C:
1. Trabalho real de v2.0 em `dev/v2.0` era pouco (pipeline declarativo
   + organizacao do ipa-dict). O resto eram "micro-arrumacoes".
2. `main` ja estava em estado cientificamente validado (papers v1.x,
   auditorias, politicas, roadmap consolidado).
3. Conflito de IDs: tickets 034 e 035 tinham significados diferentes
   nas duas branches.
4. Objetivo explicito do usuario: "nao deixar lixo de backup por muito
   tempo".

## Fase 1 — Reorganizacao de branches (EXECUTADA 2026-04-10)

### Estado antes da operacao
```
BRANCHES LOCAIS:
  dev/v2.0   a88b014  (29 commits a frente do merge-base com main)
* main       827072f  (2 commits ahead de origin/main, nao pushados)

BRANCHES REMOTAS:
  origin/dev/v2.0  a88b014
  origin/main      9e68d00  (desatualizado)

STASH:
  stash@{0}: On cleanup/initial-prune-20260318 (contem trabalho de v2.0
             com scripts de pipeline — ver secao "Conteudo preservado")
```

### Operacoes realizadas (ordem cronologica)

**1. Push de main para origin (seguranca do trabalho do dia)**
```
git push origin main
# 9e68d00..827072f  main -> main
```
Commits `f4fce4a` e `827072f` preservados no remote.

**2. Verificacao de sincronismo de dev/v2.0**
```
git log dev/v2.0..origin/dev/v2.0 --oneline  # vazio
git log origin/dev/v2.0..dev/v2.0 --oneline  # vazio
```
Confirmado: local == remote. Zero drift.

**3. Renomeacao da branch local**
```
git branch -m dev/v2.0 archive/dev-v2.0-exploration
```
Mesmo commit `a88b014`, apenas o ponteiro mudou.

**4. Push do archive para o remote**
```
git push origin archive/dev-v2.0-exploration
# * [new branch] archive/dev-v2.0-exploration -> archive/dev-v2.0-exploration
```

**5. Confirmacao de identidade pre-delete**
```
git ls-remote origin | grep -E "dev/v2.0|archive/dev"
# a88b0142b272f8c60ca75cb279121963558ee416 refs/heads/archive/dev-v2.0-exploration
# a88b0142b272f8c60ca75cb279121963558ee416 refs/heads/dev/v2.0
```
Mesmo SHA nas duas refs. Zero perda de dados garantida.

**6. Delete de dev/v2.0 do remote**
```
git push origin --delete dev/v2.0
# - [deleted]  dev/v2.0
```

**7. Criacao da branch v2 a partir de main**
```
git checkout -b v2 main
git push -u origin v2
# * [new branch] v2 -> v2
# branch 'v2' set up to track 'origin/v2'
```

### Estado depois da operacao

```
BRANCHES LOCAIS:
  archive/dev-v2.0-exploration  a88b014  (preservada, sem tracking)
  main                          827072f  [origin/main]  v1.x congelado
* v2                            827072f  [origin/v2]    atual, dev v2.0

BRANCHES REMOTAS:
  origin/archive/dev-v2.0-exploration  a88b014
  origin/main                          827072f
  origin/v2                            827072f
```

### Garantias de rastreabilidade

1. **SHA preservado**: `a88b0142b272f8c60ca75cb279121963558ee416` ainda acessivel
   localmente e no remote via `archive/dev-v2.0-exploration`.
2. **Reflog mantido**: `git reflog archive/dev-v2.0-exploration` mostra toda
   a historia de operacoes, inclusive a renomeacao.
3. **Stash preservado**: `stash@{0}` (On cleanup/initial-prune-20260318) ainda
   existe e contem arquivos que nao chegaram em dev/v2.0.
4. **Remote backup**: archive enviada para `origin/archive/dev-v2.0-exploration`,
   acessivel mesmo se o repositorio local for perdido.

## Conteudo preservado que sera importado na Fase 2

### Scripts funcionais (origem: dev/v2.0 / stash)
1. `scripts/normalize_dicts.py` (114 linhas)
   - Pipeline declarativo que carrega `*.rules.tsv` e aplica regex
     sequenciais por grupo/idioma.
   - Suporta glob para multiplos arquivos de regras.
   - CLI: `python scripts/normalize_dicts.py --group pt-br`

2. `scripts/audit_dict_diff.py` (236 linhas)
   - Auditoria tri-source: compara fonte bruta + output do pipeline + canonical.
   - Categoriza diferencas (unicode_equivalent, real_content, etc).
   - Relatorio TSV em `dicts-workbench/output/*.audit.tsv`.

### Arquivo de regras (origem: dev/v2.0 / stash)
3. `dicts-workbench/pt-br.rules.tsv` (8 linhas, declarativo)
   ```
   # group  action  value1                              value2
   pt-br    src     ./sources/ipa-dict/data/pt_BR.txt
   pt-br    dst     ./output/pt-br.tsv
   pt-br    regex   /                                   (vazio — remove barras)
   pt-br    regex   g                                   ɡ
   pt-br    regex   (\S)(?=[^\s\u0300-\u036f])          \1
   ```

### Ja presente no filesystem (untracked, pre-existente)
4. `dicts-workbench/sources/ipa-dict/` (clone git vivo)
   - Contem `data/pt_BR.txt` e 30+ outras linguas
   - `REFERENCE.tsv` com metadados (data de ultima alteracao por arquivo)
   - `sync_ipa_dict.ps1` (script de sincronismo com `git pull`)
   - O `.git/` interno sera adicionado ao `.gitignore` do FG2P

## Proximas fases (NAO executadas ainda)

### Fase 2 — Extracao seletiva
Trazer os 3 arquivos valiosos do archive/stash para a branch `v2`:
- `scripts/normalize_dicts.py`
- `scripts/audit_dict_diff.py`
- `dicts-workbench/pt-br.rules.tsv`

Adicionar ao `.gitignore`:
- `dicts-workbench/sources/ipa-dict/.git/`
- `dicts-workbench/output/`

### Fase 3 — Validacao do pipeline
Rodar `python scripts/normalize_dicts.py --group pt-br` e comparar
output com `dicts/pt-br.tsv` canonical para confirmar reprodutibilidade.

### Fase 4 — Tickets novos enxutos
Criar tickets renumerados (080+) substituindo o espirito de 036-056 de
dev/v2.0 em versao mais curta e atualizada.

### Fase 5 — Eliminacao do archive (em 1-2 semanas)
Apos confirmar que nada de valor foi perdido:
- `git branch -D archive/dev-v2.0-exploration`
- `git push origin --delete archive/dev-v2.0-exploration`
- `git stash drop stash@{0}`

## Criterios de aceite (Fase 1)

- [x] Main preservada com todos os commits do dia pushados
- [x] Archive criada local e no remote com SHA identico ao dev/v2.0 antigo
- [x] Dev/v2.0 removida do remote sem perda de dados
- [x] Branch v2 criada a partir de main (mesmo commit)
- [x] Tracking de v2 configurado (origin/v2)
- [x] Operacao documentada para rastreabilidade futura

## Comando de emergencia (caso precise reverter)

Se por qualquer motivo for necessario restaurar o estado anterior:

```bash
# Recriar dev/v2.0 a partir do archive
git branch dev/v2.0 archive/dev-v2.0-exploration
git push -u origin dev/v2.0

# Deletar v2
git checkout main
git branch -D v2
git push origin --delete v2
```

O commit `a88b0142b272f8c60ca75cb279121963558ee416` e imutavel e sempre
recuperavel enquanto `archive/dev-v2.0-exploration` existir.
