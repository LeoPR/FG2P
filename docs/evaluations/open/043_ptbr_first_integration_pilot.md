ID: 043
Title: Piloto de integração PT-BR first para base multilíngue
Type: feature
Priority: Critical
Status: Open

Descrição:
Executar um piloto de integração incremental começando por `pt-br.tsv`, preservando 100% do comportamento atual de treino/inferência, enquanto prepara a base técnica para evolução multilíngue.

Escopo:
- Coordenar fases de implementação dos tickets 044-047.
- Definir gates obrigatórios de anti-regressão PT-BR.
- Amarrar fronteiras explícitas entre:
  - padronização do corpus (Unicode/proveniência),
  - canonização fonética para treino (classes segmentais),
  - renderização de saída para consumo.

Fases do piloto:
1. Baseline + gate PT-BR (checksum e smoke tests).
2. Contrato de corpus e canonização fonética (sem quebrar pipeline atual).
3. Fechamento do rebuild PT-BR com critérios de promoção workbench -> canônico.
4. Agregação inicial PT-BR multi-arquivo (base + overlays), mantendo modo legado.
5. Preparação do loader para tags multilíngues, ainda com PT-BR como referência de estabilidade.

Critérios de aceite:
- Gate de regressão PT-BR obrigatório executando antes/depois de mudanças estruturais.
- Evidências documentadas de compatibilidade funcional no ticket 037.
- Integração inicial concluída sobre `pt-br.tsv` sem regressão de comportamento.
- Plano de rollout para multilíngue pronto e rastreável por dependências.

Dependências:
- 036, 037, 038, 041

Subtarefas vinculadas:
- 044 - Contrato de corpus canônico e metadados.
- 045 - Canonização fonética para treino e DA Loss.
- 046 - Agregação PT-BR multi-arquivo (base + overlays).
- 047 - Taxonomia lexical opcional para expansão incremental.
