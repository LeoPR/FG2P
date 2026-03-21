---
ID: 032
Title: Organizar fontes originais (`ipa-dict`) e tornar scripts de mapeamento mais visíveis
Type: maintenance / infrastructure
Priority: high
Status: open
---

Resumo:
Criar uma entrada canônica que documente as fontes originais (ex.: snapshot local do `ipa-dict`), reorganizar a rastreabilidade dos insumos sem poluir `dicts/`, e expor um CLI/script top-level para aplicar correções e gerar artefatos normalizados.

Motivação:
Atualmente a correção g→ɡ e outras normalizações estão implementadas em código (`src/phonetic_features.py`), e há uma cópia do `ipa-dict` em `backups/ipa-dict/`. Precisamos tornar essa infraestrutura evidente para futuros colaboradores e automatizar a aplicação de normalizações sem transformar `dicts/` em depósito de insumos brutos, com documentação sobre licenças, origem e checksums.

Tarefas propostas:
1. Inventariar fontes: listar conteúdos do snapshot local do `ipa-dict`, `dicts/pt-br.tsv` e `data/phoneme_map.json` com origem, licença, checksum e data de captura.
2. Definir área canônica para fontes brutas e proveniência, preferencialmente `dicts-workbench/sources/ipa-dict/`.
3. Criar `dicts/README.md` com foco em consumo: corpus canônicos, origem resumida e ponteiro para o workbench/proveniência.
4. Extrair ou criar um wrapper CLI `scripts/normalize_dicts.py` (ou `scripts/data_normalize.py`) que:
   - aplique as normalizações conhecidas (g→ɡ, NFC, etc.) usando a função central em `src/phonetic_features.py` ou replicando lógica equivalente;
   - gere hashes e um relatório (conteagens antes/depois) e um CSV com instâncias modificadas;
   - escreva saída em área de build intermediária antes da promoção para `dicts/`.
5. Adicionar um pequeno teste/unit test que verifica g→ɡ normalização (ex.: 10.252 instâncias corrigidas no histórico).
6. Documentar no `README.md` do projeto e em `IPA_REFERENCE.md` o local e o comando único para (re)aplicar normalizações.
7. Criar manifest por snapshot/regra, por exemplo `manifest.json` ou `recipe.yaml`, contendo input, licença, versão, checksum, regras e output esperado.

Critérios de aceite:
- `dicts/README.md` criado e referenciado no topo-level `README.md`, deixando claro que `dicts/` contém apenas artefatos canônicos de uso.
- Área de fontes/proveniência definida fora de `dicts/`.
- `scripts/normalize_dicts.py` presente com CLI (`--input --output --report`) e exemplo de uso documentado.
- Relatório CSV gerado com contagens antes/depois e amostra de linhas modificadas.
- Manifest de proveniência/rule-set criado para pelo menos o caso PT-BR.
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