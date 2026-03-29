ID: 055
Title: Expansão de corpus para consoantes geminadas (empréstimos)
Type: research
Priority: Low
Status: Open

Descrição:
A avaliação OOV revelou que o modelo falha sistematicamente em consoantes geminadas de empréstimos (italianos, ingleses): "lazzaretti" → prediz `z z` em vez de `z`, "mozzarela" → `z z`. A causa é clara: o corpus pt-br.tsv não contém exemplos suficientes (ou nenhum) de geminadas com redução documentada.

Objetivo:
- Expandir o corpus com exemplos de palavras com geminadas de empréstimo e suas transcrições corretas (redução a consoante simples em PT-BR).
- Validar que o modelo aprende a regra de redução após expansão.

Análise do problema:
- PT-BR não tem geminadas fonémicas produtivas no léxico nativo.
- Empréstimos com geminadas ortográficas (pizza, mozzarella, lazzaretti, cappuccino) devem reduzir: zz→z, pp→p, tt→t, cc→k, nn→n.
- Regra de redução é produtiva e regular — o modelo *deveria* aprender se tiver exemplos.
- Estimativa de palavras afetadas: ~200-500 palavras de empréstimo em português corrente com geminadas.

Passos:
- [ ] Auditar pt-br.tsv: quantas palavras com geminadas ortográficas existem? Com transcrições corretas?
- [ ] Identificar palavras candidatas (pizza, pizzaria, mozzarella, cappuccino, attore, etc.)
- [ ] Verificar se as transcrições existentes já reduzem geminadas ou duplicam
- [ ] Se ausentes: adicionar transcrições via dicts-workbench ou fonte externa (ex: Wiktionary PT-BR)
- [ ] Treinar modelo com corpus expandido e medir impacto na categoria "Consoantes Duplas" do banco OOV

Métricas de sucesso:
- Categoria "Consoantes Duplas" melhora de 1/5 (20%) para ≥3/5 (60%)
- PER/WER global não regride (expansão não introduz ruído)

Critérios de aceite:
- Corpus expandido com ≥100 exemplos de empréstimos com geminadas corretamente reduzidas.
- Experimento documentado mostrando impacto na avaliação OOV.

Dependências:
- dicts-workbench (pipeline de adição de entradas)
- docs/data/generalization_test.tsv (banco de avaliação OOV)
- 036, 037 (pipeline de corpus)
