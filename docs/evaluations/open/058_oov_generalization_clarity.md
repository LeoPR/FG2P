ID: 058
Title: §7 OOV/Generalizacao — clareza da avaliacao, contagem, dois TSV
Type: documentation
Priority: High
Status: Open

## Problemas identificados

### A) Discrepancia de contagem
ARTICLE.md diz 31 palavras no banco de generalizacao.
Arquivo real generalization_test.tsv tem **54 palavras** em 6 categorias.
Arquivo neologisms_test.tsv tem **50 palavras** em 5 categorias.

Precisa esclarecer:
- O ARTICLE.md usa um subconjunto de 31? Quais 31?
- Ou o arquivo foi expandido depois e o artigo nao atualizou?
- As 54 palavras foram todas avaliadas? Os resultados (17/31) sao de quais?

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

- [ ] Contagem no ARTICLE.md corresponde ao arquivo real
- [ ] Justificativa do N explicita no texto
- [ ] Ambos os TSV documentados (ou um descartado com justificativa)
- [ ] Distincao test set vs banco OOV clara

Dependencias: nenhuma
