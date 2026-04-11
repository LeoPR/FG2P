# dicts-workbench

> **Escopo**: area de processamento de fontes brutas para gerar dicionarios
> canonicos em `dicts/`. Nao e consumida diretamente pelo treinamento — os
> outputs sao promovidos manualmente para `dicts/` apos validacao.
>
> **Status (2026-04-11)**: Fase 2 core concluida — submodulo ipa-dict +
> scripts + pipeline declarativo basico. Validacao do pipeline = Fase 3.
> Ver `docs/evaluations/open/079_branch_reorganization_2026_04_10.md`
> e `080_phase2_deferred_decisions.md`.

## Estrutura

```
dicts-workbench/
├── README.md                        este arquivo
├── scripts/
│   ├── normalize_dicts.py           pipeline declarativo (le *.rules.tsv)
│   └── audit_dict_diff.py           auditoria tri-source (compara outputs)
├── rules/
│   └── pt-br.rules.tsv              regras por grupo (idioma ou variante)
├── sources/
│   └── ipa-dict/                    submodulo git pinned (open-dict-data/ipa-dict)
│       └── data/                    30+ arquivos .txt por idioma
└── output/                          gerado pelo pipeline, .gitignore
    └── pt-br.tsv                    saida do normalize_dicts.py
```

## Uso basico

### Aplicar regras de um grupo
```bash
python dicts-workbench/scripts/normalize_dicts.py --group pt-br
```

Sem `--group`, processa todos os grupos em ordem alfabetica:
```bash
python dicts-workbench/scripts/normalize_dicts.py
```

Glob default: `dicts-workbench/rules/*.rules.tsv`

### Auditar divergencias com canonical v1.x

Comparar o output do pipeline com o `dicts/pt-br.tsv` canonical:
```bash
python dicts-workbench/scripts/audit_dict_diff.py \
  dicts-workbench/output/pt-br.tsv \
  dicts/pt-br.tsv
```

Auditoria tri-source (fonte bruta + output + canonical):
```bash
python dicts-workbench/scripts/audit_dict_diff.py \
  dicts-workbench/output/pt-br.tsv \
  dicts/pt-br.tsv \
  --source dicts-workbench/sources/ipa-dict/data/pt_BR.txt
```

## Formato do `.rules.tsv`

Arquivo TSV com 4 colunas: `group`, `action`, `value1`, `value2`.

Acoes implementadas (Fase 2 core):

| Acao | Descricao | Exemplo |
|---|---|---|
| `src` | Arquivo de entrada (relativo ao `.rules.tsv`) | `../sources/ipa-dict/data/pt_BR.txt` |
| `dst` | Arquivo de saida | `../output/pt-br.tsv` |
| `regex` | Substituicao regex Python | `g` -> `ɡ` |

Regras sao aplicadas **sequencialmente** na ordem em que aparecem no arquivo.
Comentarios com `#` sao ignorados.

### Exemplo: `pt-br.rules.tsv`
```
# group   action   value1                          value2
pt-br     src      ../sources/ipa-dict/data/pt_BR.txt
pt-br     dst      ../output/pt-br.tsv
pt-br     regex    /
pt-br     regex    g                               ɡ
pt-br     regex    (\S)(?=[^\s\u0300-\u036f])      \1
```

Acoes planejadas para Fase 2.5 (ver ticket 080):
- `tag` — tag BCP 47 associada ao grupo
- `mode` — `full` (default) ou `overlay`
- `append` — adiciona palavras manuais de outro TSV
- `depends_on` — explicita dependencia entre grupos

## Atualizar fonte ipa-dict

O ipa-dict e submodulo git pinned (commit especifico). Para atualizar:

```bash
# entrar no submodulo e atualizar
cd dicts-workbench/sources/ipa-dict
git pull origin master

# voltar para o FG2P e registrar a nova versao
cd ../../..
git add dicts-workbench/sources/ipa-dict
git commit -m "deps: atualiza ipa-dict para <commit>"
```

## Ver historico de mudancas do ipa-dict

Dentro do submodulo, o git log funciona normalmente:

```bash
cd dicts-workbench/sources/ipa-dict
git log --oneline data/pt_BR.txt
git log -1 --format="%cI %s" data/pt_BR.txt
```

Para listar todas as linguas e suas ultimas alteracoes:
```bash
cd dicts-workbench/sources/ipa-dict
for f in data/*.txt; do
  echo -e "$f\t$(git log -1 --format='%cI %s' -- $f)"
done
```

## Promocao de output para dicts/

Apos validar que o output do pipeline esta correto, promover manualmente:

```bash
# Em v2, quando Fase 5 estiver autorizada:
cp dicts-workbench/output/pt-br.tsv dicts/pt-br/pt-br.tsv   # futuro
```

**Nota importante**: em v2, `dicts/pt-br.tsv` (raiz) continua intocado ate
a Fase 5 do plano de migracao. Ver ticket 079 e memoria
`rule_ptbr_tsv_migration.md`.
