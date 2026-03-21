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
- Preservar proveniência científica sem transformar `dicts/` em pasta de insumos brutos.

Decisão arquitetural proposta:
- Input bruto do `ipa-dict` deve viver em `dicts-workbench/sources/ipa-dict/`.
- Regras/manifest devem viver em `dicts-workbench/recipes/`.
- Saída intermediária e relatórios devem viver em `dicts-workbench/build/` e `dicts-workbench/reports/`.
- Apenas o artefato aprovado/final é promovido para `dicts/pt-br/pt-br.tsv`.

Recomendação crítica:
- Não usar `dicts/raw/` como base do desenho principal.
- Não depender de um único tarball/zstd monolítico de todos os idiomas como unidade lógica do processo; preferir snapshots por fonte/língua com manifest explícito.
- Manter cópia local compactada do `pt_BR.txt` é totalmente razoável, desde que vinculada a checksum, data de captura e recipe declarativa.

Metadados mínimos recomendados por recipe/snapshot:
- origem (`repo`, URL, commit ou data da captura)
- licença
- input file e checksum
- regras aplicadas e ordem
- output esperado e checksum
- data de geração
- observações sobre divergências conhecidas

Critérios de aceite:
- Script reproduzível gera saída equivalente ao canônico (ou diff pequeno e explicado).
- Relatório de diferenças por categoria (ex.: charset, IPA symbols, tokenização).
- Processo reutilizável para novas línguas.
- Recipe/manifest PT-BR documentada e versionada.

Dependências:
- 036, 037
