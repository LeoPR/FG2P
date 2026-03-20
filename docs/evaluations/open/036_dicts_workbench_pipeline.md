ID: 036
Title: Workbench de dicionarios e pipeline de limpeza
Type: feature
Priority: High
Status: Open

Descrição:
Projetar uma área de trabalho separada para estudos e transformação de dicionários brutos até corpus final, sem contaminar a pasta oficial `dicts/`.

Proposta inicial:
- Pasta de suporte: `dicts-workbench/`.
- Subpastas sugeridas:
  - `raw/` (fontes originais)
  - `rules/` (regras declarativas de correção)
  - `scripts/` (pipeline de transformação)
  - `reports/` (diferenças/qualidade)
  - `build/` (saídas intermediárias)

Objetivo:
- Script genérico: lê corpus bruto + regras -> gera corpus final limpo para `dicts/`.

Critérios de aceite:
- Estrutura do workbench definida.
- Contrato de I/O do pipeline documentado.
- Exemplo mínimo de execução ponta-a-ponta.

Dependências:
- 026, 032
