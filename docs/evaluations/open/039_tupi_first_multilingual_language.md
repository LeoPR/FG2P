ID: 039
Title: Tupi como primeira língua piloto multilíngue
Type: feature
Priority: High
Status: Open

Descrição:
Usar o Tupi como primeiro caso real de expansão multilíngue após estabilizar o pipeline genérico.

Contexto:
- Arquivo detectado: `dicts/tpw_latn_broad.tsv`.
- Código de língua deve ser validado e fixado no projeto (`tpw` vs alternativa), com justificativa documental.

Escopo:
- Ingestão do corpus Tupi no workbench.
- Limpeza mínima com regras explícitas.
- Export para estrutura oficial de `dicts/`.

Critérios de aceite:
- Corpus Tupi carregável pelo pipeline.
- Checklist de qualidade mínima definido.
- Sem regressão no fluxo PT-BR.

Dependências:
- 036, 038
