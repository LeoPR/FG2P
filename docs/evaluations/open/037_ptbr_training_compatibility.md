ID: 037
Title: Compatibilidade do PT-BR com treino atual
Type: validation
Priority: Critical
Status: Open

Descrição:
Garantir que o corpus PT-BR consolidado continue 100% compatível com o treino e inferência atuais durante toda a migração estrutural.

Escopo:
- Congelar `dicts/pt-br/pt-br.tsv` como referência.
- Registrar checksum SHA256.
- Executar smoke tests de treino/inferência antes e depois de qualquer mudança de path.

Critérios de aceite:
- Mesmo dataset efetivo carregado pelo pipeline (ou diffs explicados).
- Treino curto e inferência de sanidade sem regressão funcional.
- Registro de evidências no ticket.

Dependências:
- 035
