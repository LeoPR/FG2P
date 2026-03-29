ID: 044
Title: Contrato canônico de corpus e metadados de proveniência
Type: feature
Priority: High
Status: Open

Descrição:
Definir contrato mínimo obrigatório para qualquer corpus de entrada/saída no pipeline de dicionários, com foco inicial em PT-BR.

Objetivo:
- Tornar explícito se o corpus está em NFC/NFD.
- Registrar proveniência e checksums de forma reproduzível.
- Separar decisão de armazenamento/representação da decisão de treino.

Escopo:
- Especificar metadados mínimos por snapshot/regra:
  - `unicode_form` (NFC/NFD),
  - `token_scheme`,
  - `source` (repo/URL/commit/data),
  - `license`,
  - `input_checksum`,
  - `ruleset` e ordem,
  - `output_checksum`,
  - `generated_at`,
  - `known_diffs`.
- Definir formato (YAML ou JSON) e local padrão de manifesto.
- Integrar contrato ao fluxo do workbench (036/038).

Critérios de aceite:
- Especificação documentada e versionada.
- Exemplo PT-BR preenchido e validado.
- Auditoria consegue explicar diferenças representacionais sem confundir com diferença fonética.

Dependências:
- 036, 038, 043
