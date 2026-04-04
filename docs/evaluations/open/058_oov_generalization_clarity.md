ID: 058
Title: §7 OOV/Generalizacao — clareza da avaliacao, contagem, dois TSV
Type: documentation
Priority: High
Status: Open

## Problemas identificados

### A) ~~Discrepancia de contagem~~ VERIFICADO — contagem correta
~~ARTICLE.md diz 31 palavras no banco de generalizacao.~~
Verificado em 2026-04-04: generalization_test.tsv tem exatamente **31 linhas de dados**
(56 linhas totais incluindo header e comentarios). neologisms_test.tsv tem **35 palavras**.
Contagem do ARTICLE.md esta correta.

### B) Justificativa do N pequeno
O artigo nao explica bem PORQUE sao 31 (ou 54) palavras. Comparar com:
- O test set principal: 28.782 palavras (robusto, estratificado)
- Este banco: curadoria manual, diagnostico qualitativo

Sugestao: explicitar que este banco NAO substitui o test set — e uma sonda
diagnostica complementar para testar generalizacao a tipos especificos de dificuldade.

### C) Dois TSV nao documentados claramente
O artigo menciona generalization_test.tsv mas nao menciona neologisms_test.tsv.
Se existe, deveria ser documentado ou referenciado.

### D) Clareza narrativa OOV vs test set
O test set (28.782 palavras) prova estabilidade do modelo em palavras reais de dicionario.
O banco OOV prova uso pratico: neologismos, nomes, emprestimos, desafios fonologicos.
Essa distincao precisa estar explicita.

## Verificacao

- [x] Contagem no ARTICLE.md corresponde ao arquivo real (31 palavras confirmado)
- [x] Justificativa do N explicita no texto (adicionada em §7.1)
- [x] Ambos os TSV documentados (neologisms_test.tsv referenciado em §7.1)
- [x] Distincao test set vs banco OOV clara (adicionada em §7.1)

Dependencias: nenhuma
