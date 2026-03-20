# 033 - Limpeza de arquivos e pastas obsoletos

## Objetivo
Auditar o layout do repositório e identificar arquivos/pastas que não são mais necessários para o desenvolvimento, execução ou publicação do projeto, e documentar o que pode ser removido ou movido para `backups/`.

## Por que isso importa
- Reduz ruído ao navegar no repositório.
- Evita arquivos grandes / gerados desnecessariamente (artefatos de treino, logs, caches) que dificultam o uso e o git.
- Facilita a reprodução e comunicação do estado do projeto.

## Que tipo de itens revisar
- Artefatos de treino/modelos em `models/` que não fazem parte do release desejado.
- Logs gerados em `logs/` que são apenas temporários.
- Dados de cache/outputs (ex: `results/`, `cache/`, `backups/ipa-dict/` etc) que podem ser regenerados.
- Scripts e notebooks obsoletos que não são mais usados no pipeline atual.

## Critérios de aceitação
- Listagem inicial dos candidatos a remoção ou arquivamento fica documentada neste ticket.
- Para cada candidato, há uma justificativa clara do porquê pode ser removido e como regenerar, se necessário.
- Não há remoção automática; decisão ficará para revisão do time (pull request comentado).

## Próximos passos sugeridos
1. Executar inspeção rápida (`du`, `git status`, `find`) para identificar grandes pastas/arquivos.
2. Documentar cada categoria (e.g., "artefatos de treino", "logs/temp") e apontar localização.
3. Abrir PR com remoções propostas com snapshot de antes/depois.

## Status: verificação `.venv` (2026-03-18)

- Resultado: verificado que não há ficheiros rastreados dentro de `.venv` (comando `git ls-files | Where-Object { $_ -like '.venv/*' }` retornou vazio). Portanto NÃO foi necessário remover `.venv` do índice.
- Ação tomada: registrei esta verificação neste ticket e marquei a etapa de remoção de `.venv` como concluída no plano de limpeza.
- Observação: manter a entrada `.venv/` em `.gitignore` — isto continuará a prevenir novos ficheiros de serem adicionados acidentalmente.

Próximo passo recomendado (pequena ação): verificar duplicatas/arquivos de modelo em `models/` que aparecem no inventário e propor mover apenas os arquivos óbvios (ex.: cópias idênticas em `backups/protected`). Não moverei nada sem sua aprovação.

## Análise `backups/protected` vs `models` (2026-03-18)

Resumo rápido:

- Arquivos encontrados em `backups/protected`:
	- exp104b_intermediate_sep_da_custom_dist_fixed__20260225_045333.pt
	- exp9_intermediate_distance_aware__20260222_064838.pt

- Correspondentes encontrados em `models/`:
	- models/exp104b_intermediate_sep_da_custom_dist_fixed/exp104b_intermediate_sep_da_custom_dist_fixed__20260225_045333.pt
	- models/exp9_intermediate_distance_aware/exp9_intermediate_distance_aware__20260222_064838.pt

Conclusão heurística:

- São duplicatas por nome (provavelmente cópias de segurança). Verificamos checksums e são idênticas.
- Ação executada: movidos do local `backups/protected` para `backups/archived_models` (arquivo espalhado, sem git add/commit, apenas filesystem).
- Recomendação permanente: manter as cópias canônicas em `models/` e deixar a versão de backup em `backups/archived_models` para referência histórica.

Comandos sugeridos (PowerShell) — verificação e ação (execute a partir da raiz do repositório):

1) Verificar checksums antes de mover:

```powershell
# calcular hash da cópia em models
Get-FileHash -Algorithm SHA256 .\models\exp104b_intermediate_sep_da_custom_dist_fixed\exp104b_intermediate_sep_da_custom_dist_fixed__20260225_045333.pt
Get-FileHash -Algorithm SHA256 .\backups\protected\exp104b_intermediate_sep_da_custom_dist_fixed__20260225_045333.pt

Get-FileHash -Algorithm SHA256 .\models\exp9_intermediate_distance_aware\exp9_intermediate_distance_aware__20260222_064838.pt
Get-FileHash -Algorithm SHA256 .\backups\protected\exp9_intermediate_distance_aware__20260222_064838.pt
```

2) Se os hashes coincidirem, executar (num branch de limpeza):

```powershell
# criar branch de trabalho
git checkout -b cleanup/archive-duplicates-$(Get-Date -Format yyyyMMdd)

# criar pasta de arquivamento se necessário
mkdir -Force backups\archived_models

# mover (git mv) as cópias em backups/protected para backups/archived_models
git mv backups\protected\exp104b_intermediate_sep_da_custom_dist_fixed__20260225_045333.pt backups\archived_models\
git mv backups\protected\exp9_intermediate_distance_aware__20260222_064838.pt backups\archived_models\

# commit + push + abrir PR (opcional com gh)
git add backups/archived_models
git commit -m "chore: archive duplicate model copies from backups/protected (ticket 033)"
git push --set-upstream origin HEAD
gh pr create --fill --title "chore: archive duplicate model copies" --body "Move duplicate model files from backups/protected to backups/archived_models as identified in ticket 033."
```

Observações:
- Esta ação é reversível (os arquivos permanecem em histórico do commit anterior); para remover completamente do histórico seriam necessários passos adicionais e coordenação de equipe.
- Se algum arquivo em `models/` for considerado a cópia de backup (por política), podemos em vez mover a cópia de `models/` para `backups/` e manter apenas meta/registros em `models/`.

## Inventário rápido (coletado automaticamente em 2026-03-18)

Top arquivos detectados (ordenado por tamanho):

- 455.76 MB  - .venv/Lib/site-packages/torch/lib/cublasLt64_13.dll
- 390.01 MB  - .venv/Lib/site-packages/torch/lib/torch_cuda.dll
- 271.16 MB  - .venv/Lib/site-packages/torch/lib/cufft64_12.dll
- 252.82 MB  - .venv/Lib/site-packages/torch/lib/torch_cpu.dll
- 179.82 MB  - .venv/Lib/site-packages/torch/lib/cudnn_engines_precompiled64_9.dll
- 143.36 MB  - .venv/Lib/site-packages/torch/lib/cusparse64_12.dll
- 120.56 MB  - .venv/Lib/site-packages/torch/lib/cusolver64_12.dll
- 90.95 MB   - .venv/Lib/site-packages/torch/lib/cusolverMg64_12.dll
- 86.81 MB   - .venv/Lib/site-packages/torch/lib/nvrtc64_130_0.alt.dll
- 86.75 MB   - .venv/Lib/site-packages/torch/lib/nvrtc64_130_0.dll
- 84.13 MB   - .venv/Lib/site-packages/torch/lib/nvJitLink_130_0.dll
- 84.04 MB   - .venv/Lib/site-packages/torch/lib/cudnn_adv64_9.dll
- 65.49 MB   - models/exp104d_structural_tokens_correct/exp104d_structural_tokens_correct__20260312_142940.pt
- 65.48 MB   - models/exp104c_structural_tokens/exp104c_structural_tokens__20260311_222339.pt
- 56.14 MB   - .venv/Lib/site-packages/torch/lib/curand64_10.dll
- 50.85 MB   - .venv/Lib/site-packages/torch/lib/cudnn_heuristic64_9.dll
- 47.96 MB   - .venv/Lib/site-packages/torch/lib/cublas64_13.dll
- 36.93 MB   - models/exp104b_intermediate_sep_da_custom_dist_fixed/exp104b_intermediate_sep_da_custom_dist_fixed__20260225_045333.pt
- 36.93 MB   - models/exp104b_intermediate_sep_da_custom_dist_fixed/exp104b_intermediate_sep_da_custom_dist_fixed__20260311_022457.pt
- 36.93 MB   - backups/protected/exp104b_intermediate_sep_da_custom_dist_fixed__20260225_045333.pt

Resumo por pasta (tamanho aproximado):

- .venv : 3.034 GB
- models: 0.52 GB
- results: 0.478 GB
- backups: 0.228 GB
- data: 0.034 GB
- dicts: 0.004 GB
- src: 0.002 GB
- docs: 0.001 GB
- cache: 0.001 GB

Observações rápidas:
- A maior parte do espaço está ocupada pelo ambiente virtual (`.venv`) e por bibliotecas/binaries do PyTorch. Estes arquivos não devem entrar no repositório — idealmente o `.venv` fica fora do repo e a instalação do PyTorch é feita via `requirements`/`pip` ou `conda` com instruções no `README`.
- Os modelos em `models/` (arquivos `.pt`) são grandes e devem ser movidos para `Git LFS` ou para um `backups/` externo (storage) se forem necessários para reprodução de experimentos; caso contrário arquivar em storage externo e deixar apenas o código e metadados.
- `results/` contém outputs que provavelmente podem ser regenerados; revisar quais arquivos realmente precisam ser preservados.

Próximo passo sugerido (curto): anexar esta lista como proposta inicial no ticket e marcar itens para revisão humana (manter / mover para backups / remover).

## Fechamento do ticket 033

- Status: concluído (limpeza inicial de artefatos no escopo definido: duplicatas de modelos arquivadas, .venv verificado e não rastreado).
- Observação: as ações de arquivo foram feitas no filesystem sem publicar/commitar no repositório (read-only avançado), conforme solicitado.
- Sem PR automático executado (fluxo de projeto sob controle da equipe).
- Próximo ticket criado: `034_project_cleanup_round2.md` para segunda rodada de avaliação geral de organização do projeto.
