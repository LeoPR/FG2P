ID: 038
Title: Reproduzir PT-BR a partir do `pt_BR` (ipa-dict)
Type: feature
Priority: High
Status: Open

Descrição:
Formalizar a história de derivação do corpus `pt-br.tsv`: partir de `pt_BR` bruto (ipa-dict), aplicar regras corretivas e gerar corpus limpo equivalente ao canônico.

Objetivo técnico:
- Implementar pipeline reproduzível: `raw -> rules -> cleaned`.
- Tornar as regras auditáveis (não hardcoded opaco).

Critérios de aceite:
- Script reproduzível gera saída equivalente ao canônico (ou diff pequeno e explicado).
- Relatório de diferenças por categoria (ex.: charset, IPA symbols, tokenização).
- Processo reutilizável para novas línguas.

Dependências:
- 036, 037
