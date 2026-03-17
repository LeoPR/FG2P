---
ID: 032
Title: Organizar fontes originais (`ipa-dict`) e tornar scripts de mapeamento mais visíveis
Type: maintenance / infrastructure
Priority: high
Status: open
---

Resumo:
Criar uma entrada canônica que documente as fontes originais (ex.: `backups/ipa-dict/`), reorganizar a pasta `dicts/` com um README e scripts de mapeamento/normalização claros (ex.: g→ɡ), e expor um CLI/script top-level para aplicar correções e gerar artefatos normalizados.

Motivação:
Atualmente a correção g→ɡ e outras normalizações estão implementadas em código (`src/phonetic_features.py`), e há backups de `ipa-dict` em `backups/ipa-dict/`. Precisamos tornar essa infraestrutura evidente para futuros colaboradores e automatizar a aplicação de normalizações na fonte (`dicts/`), com documentação sobre licenças e origem das fontes.

Tarefas propostas:
1. Inventariar fontes: listar conteúdos de `backups/ipa-dict/`, `dicts/pt-br.tsv` e `data/phoneme_map.json` com origem e licença.
2. Criar `dicts/README.md` com: fonte(s), versão, licença, instruções para regenerar/normalizar (com exemplos de comando).
3. Extrair ou criar um wrapper CLI `scripts/normalize_dicts.py` (ou `scripts/data_normalize.py`) que:
   - aplique as normalizações conhecidas (g→ɡ, NFC, etc.) usando a função central em `src/phonetic_features.py` ou replicando lógica equivalente;
   - gere hashes e um relatório (conteagens antes/depois) e um CSV com instâncias modificadas;
   - escreva saída em `dicts/pt-br.normalized.tsv` por padrão.
4. Adicionar um pequeno teste/unit test que verifica g→ɡ normalização (ex.: 10.252 instâncias corrigidas no histórico).
5. Documentar no `README.md` do projeto e em `IPA_REFERENCE.md` o local e o comando único para (re)aplicar normalizações.

Critérios de aceite:
- `dicts/README.md` criado e referenciado no topo-level `README.md`.
- `scripts/normalize_dicts.py` presente com CLI (`--input --output --report`) e exemplo de uso documentado.
- Relatório CSV gerado com contagens antes/depois e amostra de linhas modificadas.
- Ticket/PR linking e uma entrada no changelog descrevendo a mudança.

Próximos passos imediatos:
- [ ] Confirmar convenção de path preferida (`scripts/` vs `src/scripts/`).
- [ ] Eu posso gerar um rascunho do `scripts/normalize_dicts.py` e um `dicts/README.md` — quer que eu os crie agora como PR draft no workspace?
\n+Implementação (estado atual):

- `scripts/normalize_dicts.py` criado — CLI simples que aplica NFC e a normalização conhecida `g`(U+0067) → `ɡ`(U+0261), produz `dicts/pt-br.normalized.tsv` e `reports/normalize_dicts.csv` com as linhas alteradas.
- `dicts/README.md` criado com instruções de uso e referência a `backups/ipa-dict/`.

Exemplo de uso:

```bash
python scripts/normalize_dicts.py --input dicts/pt-br.tsv --output dicts/pt-br.normalized.tsv --report reports/normalize_dicts.csv
```

Próximos passos (revisão):
- Mover script para `src/` se preferir integração como módulo importável; atualmente está em `scripts/` para visibilidade.
- Adicionar testes unitários que chequem contagens históricas (ex.: 10.252 g→ɡ).
- Atualizar `README.md` top-level para referenciar `dicts/README.md` (opcional, posso fazer).

Status: implmentação inicial criada; revisão e testes pendentes.