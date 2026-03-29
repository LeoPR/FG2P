ID: 047
Title: Taxonomia lexical opcional para expansão incremental
Type: feature
Priority: Medium
Status: Open

Descrição:
Definir suporte opcional a metadados lexicais (classe, raiz, derivação, variante) para enriquecer futuros datasets sem quebrar o pipeline atual.

Objetivo:
- Preparar expansão de vocabulário por grupos (substantivos, verbos, plurais, derivados etc.).
- Preservar compatibilidade total com entradas sem metadados.

Escopo:
- Propor esquema mínimo de metadados lexicais opcionais.
- Definir fallback quando metadado não existir.
- Integrar com planos de loader multilíngue e overlays PT-BR.

Critérios de aceite:
- Especificação de metadados documentada.
- Pipeline atual continua aceitando datasets legados sem alterações.
- Exemplo PT-BR com e sem metadados processado pelo mesmo fluxo.

Dependências:
- 041, 043, 046
