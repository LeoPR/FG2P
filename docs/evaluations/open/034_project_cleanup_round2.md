# 034 - Segunda rodada de avaliação: organização do projeto

## Objetivo
Executar uma segunda varredura de organização que cubra:
- Estrutura de pastas e convenções de nomes (source, data, models, results, etc.)
- Armazenamento e versionamento de artefatos grandes (`models/*.pt`, backups, .venv, caches)
- Dependências e ambiente reproduzível (requirements, .venv, Dockerfile, scripts de setup)
- Documentação e guias mínimos de onboarding (README / QUICKSTART / docs/ )
- Automação de qualidade (pre-commit, lint, CI checks, verificadores de tamanho > 50MB)

## Por que isso importa
- Reduz a dívida técnica de manutenção e onboarding de novos colaboradores.
- Evita múltiplas decisões ad-hoc sobre onde colocar artefatos grandes.
- Dá base para estabilizar o fluxo de desenvolvimento e avaliação.

## Critérios de aceitação
- Lista documentada de itens adicionais de limpeza/estruturação gerada e priorizada.
- Objetivos e próximos passos do ticket 033 incorporados (limpeza inicial): finalizado e registrado.
- Proposta de política de arquivo/binário (p.ex., `models/` somente metadados, checkpoints em LFS ou armazenamento externo).

## Primeiros passos
1. Rever incisos do ticket 033 e validar que não há itens pendentes.
2. Executar um inventário de `models/`, `backups/`, `results/`, `logs/`, `cache/`, `dicts/` com `du` e `git status`.
3. Verificar se os artefatos gerados geram commits não intencionais (extensões grandes, por exemplo `.pt`, `.dll`, `.onnx`, `.pth`).
4. Atualizar `docs/` com regras de non-goal (o que NÃo deve estar no repo) e uma versão de `git clean`/`git prune` usada.

## Observação final
Depois dessa segunda rodada, o ticket 034 pode ser fechado e o projeto fica com base clara para o ciclo seguinte (release/train/infer).

## Relatório local (executado em 2026-03-20)

- Inventário completo gerado com `scripts/inventory_calc.ps1`.
- Resultado:
  - models: 0.52 GB, 40 arquivos
  - backups: 0.27 GB, 67 arquivos
  - results: 0.48 GB, 880 arquivos
  - .venv: 3.03 GB, 26.531 arquivos
  - logs: 0.00 GB, 0 arquivos
  - cache: 0.00 GB, 1 arquivo
  - dicts: 0.00 GB, 3 arquivos
  - docs: 0.00 GB, 53 arquivos
  - src: 0.00 GB, 72 arquivos

- Status da limpeza: arquivos duplicados movidos de `backups/protected` para `backups/archived_models`.
- Verificação de integridade: SHA256 verificada para os backups duplicados.
- Política (a ser implementada): não rastrear `*.pt`, `*.dll`, `*.onnx`, `*.pth` na lógica do repositório; manter solo `models/` com metadados preferenciais.

## Próximos passos (local)

1. Refinar `requirements` / `README` com instruções de ambiente, incluindo a exclusão de `.venv` e uso de `pip install -r requirements.txt`.
2. Lista de candidatos para remoção em `results/` (arquivos gerados que podem ser reconstruídos).
3. Avaliar migração `models` para Git LFS ou armazenamento por release externo.
4. Criar política de pre-commit/CI para barrar rastreamento de arquivos grandes e conversão a LFS. 


## Inventário imediato (2026-03-20)

- models: 0.52 GB, 40 arquivos
- backups: 0.27 GB, 67 arquivos
- results: 0.48 GB, 880 arquivos
- logs: 0.00 GB, 0 arquivos
- cache: 0.00 GB, 1 arquivo
- dicts: 0.00 GB, 3 arquivos
- docs: 0.00 GB, 53 arquivos
- src: 0.00 GB, 72 arquivos
- .venv: 3.03 GB, 26531 arquivos

### Prioridade para segunda rodada
1. Consolidar `models/` e `backups/`:
   - manter apenas os checkpoints necessários para a etapa de pesquisa/experimentos ativa;
   - mover checkpoins não usados para `backups/archived_models` (já iniciado no ticket 033);
   - considerar Git LFS para `models/*.pt` (essa parte ainda não feita).
2. Verificar `results/` grandes: identificar quais resultados são gerados e podem ser removidos/regen.
3. Excluir/ignorar qualquer arquivo binário grande em `logs/` / `cache/` / `.venv` (já `.venv` está no .gitignore).
4. Documentar políticas em `README`/`docs/` para evitar reintrodução: `*.pt`, `*.pkl`, `*.onnx`, `*.dll` não devem ser tracker.

---

## Relatório de Auditoria Completa (2026-03-20)

### Git pack size

- **316 MB** (size-pack: 323.396 KB)
- 1.410 objetos empacotados

### 1. CRÍTICO — Benchmark CSVs rastreados no git

| Métrica | Valor |
|---------|-------|
| Arquivos | 172 |
| Tamanho total | **~403 MB** |
| Padrão | `results/*/benchmark_*_cuda_raw*.csv` |
| Regenerável? | **Sim** — gerados pelo script de benchmark a partir dos modelos |

**Problema**: Esses CSVs contêm dados de latência bruta por batch size (b4, b8, b16, ..., b512) para cada experimento. São 9 variantes de batch × ~19 experimentos. Cada arquivo tem ~2,35 MB e é 100% regenerável.

**Ação recomendada**: Adicionar ao `.gitignore` e remover do tracking com `git rm --cached`.

Regra sugerida:
```
results/*/benchmark_*_cuda_raw*.csv
```

### 2. ALTO — Modelos .pt rastreados (intencional, mas pesado)

| Métrica | Valor |
|---------|-------|
| Arquivos tracked | 3 |
| Tamanho | **~139 MB** |
| Modelos | `exp104d` (65 MB), `exp9` ×2 (37 MB cada) |

**Status**: Whitelisted no `.gitignore` intencionalmente. São os best models (best PER e best WER).

**Recomendação futura**: Migrar para Git LFS ou release assets quando o repo for publicado. Não é urgente para uso acadêmico local.

### 3. MÉDIO — Checkpoints .pt locais não rastreados

| Métrica | Valor |
|---------|-------|
| Arquivos | 16 |
| Tamanho | **~393 MB** |

Esses são checkpoints de experimentos que estão corretamente ignorados pelo `.gitignore` mas ocupam disco local.

**Lista de checkpoints locais (não rastreados)**:

| Experimento | Tamanho |
|------------|---------|
| exp0_baseline_70split | 16,5 MB |
| exp0_legacy_simple (×2) | 33,0 MB |
| exp0_training_regime | 16,5 MB |
| exp103_intermediate_sep_distance_aware | 36,9 MB |
| exp104b (×2) | 73,9 MB |
| exp104c_structural_tokens | 65,5 MB |
| exp105_reduced_data_50split | 36,9 MB |
| exp1_baseline_60split | 16,5 MB |
| exp3_panphon_trainable | 16,5 MB |
| exp4_panphon_fixed_24d | 15,2 MB |
| exp7_lambda_lower_bound_0.05 | 16,5 MB |
| exp7_lambda_mid_candidate_0.20 | 16,5 MB |
| exp7_lambda_upper_bound_0.50 | 16,5 MB |
| exp8_panphon_distance_aware | 16,5 MB |

**Decisão pendente**: Quais manter localmente para reprodução rápida? Quais mover para `backups/archived_models/`?

### 4. MÉDIO — backups/ipa-dict/ (cópia completa desnecessária)

| Métrica | Valor |
|---------|-------|
| Arquivos | 35 |
| Tamanho | **~190 MB** |
| Conteúdo | Dicionários IPA de 20+ idiomas (ar, de, en_*, es_*, fr_*, ja, ko, etc.) |

**Problema**: É uma cópia completa do repositório open-dict-data/ipa-dict. O único arquivo relevante (`pt_BR.txt`) já existe em `dicts/pt-br/pt_BR.txt`.

**Ação recomendada**: Remover a pasta completa. Se necessário futuramente, clonar de novo do upstream.

### 5. BAIXO — Configs órfãos (sem diretório de modelo)

14 configs em `conf/` não têm diretório correspondente em `models/`:

| Config | Provável status |
|--------|----------------|
| `config_exp0_legacy_s1.json` | Teste de seed — descartado |
| `config_exp0_legacy_s100.json` | Teste de seed — descartado |
| `config_exp0_legacy_s7.json` | Teste de seed — descartado |
| `config_exp0_legacy_s999.json` | Teste de seed — descartado |
| `config_exp10_extended_distance_aware.json` | Resultado em `results/` mas sem modelo salvo |
| `config_exp101_baseline_60split_separators.json` | Idem |
| `config_exp102_intermediate_60split_separators.json` | Idem |
| `config_exp104_intermediate_sep_da_custom_dist.json` | Superseded por exp104b |
| `config_exp106_no_hyphen_50split.json` | Resultado em `results/` mas sem modelo |
| `config_exp107_maxdata_95train.json` | Nunca executado (sem results/) |
| `config_exp11_baseline_decomposed.json` | Resultado em `results/` |
| `config_exp2_extended_512hidden.json` | Idem |
| `config_exp5_intermediate_60split.json` | Idem |
| `config_exp6_distance_aware_loss.json` | Idem |

**Ação recomendada**: Manter os configs (são leves, < 1KB cada). Documentar quais são legacy/descartados via prefixo ou anotação.

### 6. OK — Pastas bem configuradas

| Pasta | Status |
|-------|--------|
| `logs/` | Vazia, ignorada |
| `cache/` | 1 arquivo (1,5 MB panphon pkl), ignorada |
| `data/` | 38 arquivos, todos ignorados corretamente |
| `.venv/` | 3,03 GB, ignorada |
| `docs/` | 53 arquivos, PDFs/HTML/DOCX gerados ignorados |
| `src/` | 72 arquivos, nenhum problema |
| `dicts/` | 4 arquivos (3 em pt-br/ + 1 Tupi), tracking OK |

### Matriz de ações recomendadas

| # | Ação | Severidade | Esforço | Impacto | Status |
|---|------|-----------|---------|---------|--------|
| A1 | Adicionar `results/*/benchmark_*_cuda_raw*.csv` ao `.gitignore` | CRÍTICO | Baixo | -403 MB do tracking | Concluído em 2026-03-20 |
| A2 | `git rm --cached` nos 172 benchmark CSVs | CRÍTICO | Baixo | Limpa o index | Concluído em 2026-03-20 |
| A3 | Remover `backups/ipa-dict/` do disco | MÉDIO | Baixo | -190 MB disco | Pendente |
| A4 | Decidir destino dos 16 .pt locais não rastreados | MÉDIO | Médio | -393 MB potencial | Pendente |
| A5 | Documentar status dos 14 configs órfãos | BAIXO | Baixo | Clareza | Pendente |
| A6 | Avaliar Git LFS para os 3 .pt tracked | BAIXA | Alto | Necessário para publicação | Backlog (adiável) |
| A7 | Purge do histórico git/GitHub para remover blobs antigos (filter-repo/BFG) | BAIXA | Alto | Reduz histórico remoto, sem efeito funcional local | Backlog (pode ser abandonado por ora) |

### Status do .gitignore atual

O `.gitignore` está **bem configurado**, incluindo agora benchmark CSVs regeneráveis. Regras existentes:
- ✅ `models/*/*.pt` (com whitelists para best models)
- ✅ `results/predictions_*.tsv`
- ✅ `results/*/benchmark_*_cuda_raw*.csv`
- ✅ `results/*.pptx`
- ✅ `/backups`, `/cache`, `/data`, `/logs`, `results/_reports`
- ✅ `.venv/`, `__pycache__/`, `*.pyc`
- ✅ `docs/*.pdf`, `docs/*.html`, `docs/*.docx`

### Publicação/Sincronização sugerida (2026-03-20)

- Recomenda-se **não sincronizar `dicts/` agora** enquanto a migração de subpastas não estiver estabilizada.
- Se for sincronizar no curto prazo, priorizar apenas `docs/evaluations/` (tickets) e, separadamente, o ajuste do `.gitignore`.
- O untracking dos benchmarks (A2) deve entrar no mesmo ciclo de sincronização do A1 para evitar reintrodução em merges futuros.
- Purge histórico (A7) não é necessário para funcionamento; tratar apenas quando houver objetivo explícito de reduzir histórico remoto.
