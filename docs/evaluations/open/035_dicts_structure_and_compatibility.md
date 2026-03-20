ID: 035
Title: Estrutura de dicionarios multilíngue com compatibilidade
Type: feature
Priority: High
Status: Open

Descrição:
Definir e implementar a organização física de `dicts/` para suportar múltiplas línguas/variantes sem quebrar o fluxo atual de treino com `dicts/pt-br/pt-br.tsv`.

Escopo:
- Propor árvore canônica por língua/variante: `dicts/<lang>/<region>/<variant?>`.
- Manter compatibilidade temporária com caminhos legados.
- Não alterar conteúdo linguístico dos arquivos nesta etapa.

Entregáveis:
- Documento canônico de estrutura de pastas em `dicts/README.md`.
- Mapeamento de caminhos legado -> novo.
- Plano de depreciação de `dicts/pt-br/`.

Critérios de aceite:
- Estrutura aprovada e documentada.
- Nenhuma regressão no treino existente por mudança de path.
- Plano explícito de rollback.

Dependências:
- 026 (épico)
