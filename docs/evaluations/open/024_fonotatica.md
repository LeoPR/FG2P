ID: 024
Title: Pipeline Fonotático (4 fases)
Type: feature
Priority: Medium
Status: Open

Descrição:
Implementar o pipeline fonotático recomendado no ARTICLE §9.1 em quatro fases: diagnóstico, N-gram fonotático, autômato de estados finitos (FSA) e integração via reranking/penalização.

Critérios de aceite:
- Documento de diagnóstico com métricas que quantifiquem violações fonotáticas no test set.
- Implementação de prova de conceito do N-gram fonotático (treino + API para scoring de sequência).
- Especificação do FSA (estados, transições) e protótipo de validação.
- Estratégia de integração descrita (reranking vs loss penalty), com casos de risco/rollback.

Próximos passos:
1. Coletar exemplos de violações (script de extração de erros atuais).
2. Treinar N-gram sobre o corpus e fornecer exemplos de scorings.
3. Projetar e validar o FSA em poucos exemplos.
