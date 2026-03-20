ID: 042
Title: Limpeza de legado e depreciação controlada de arquivos antigos
Type: maintenance
Priority: High
Status: Open

Descrição:
Reduzir ruído operacional removendo/arquivando artefatos e documentos legados que induzem leitura equivocada ou manutenção em caminhos não canônicos.

Problema:
- Existem arquivos legados e caminhos históricos coexistindo com caminhos novos.
- Isso aumenta risco de editar arquivo errado e introduzir regressões.

Escopo:
- Inventariar arquivos/pastas legadas por categoria: ativo, legado-mantido, legado-para-arquivo, legado-para-remoção.
- Definir convenção de sufixo/marker para legado (ex.: `LEGACY_` em docs ou seção explícita de depreciação).
- Consolidar links para caminhos canônicos e remover referências ambíguas.
- Criar política de retenção: o que fica, por quanto tempo, e quando remover.
- Registrar purge de histórico git/GitHub como ação opcional de baixa prioridade (pode ser adiada/abandonada sem impacto funcional).

Critérios de aceite:
- Matriz de depreciação publicada (arquivo -> status -> destino -> prazo).
- Pastas canônicas declaradas para cada domínio (`dicts`, `results`, `backups`, `docs`).
- Pelo menos um ciclo de limpeza executado sem afetar treino/inferência.

Dependências:
- 034 (auditoria de organização)
- 037 (garantia anti-regressão no PT-BR)
