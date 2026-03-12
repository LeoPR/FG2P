# 004 - Validade Externa, Comparacao Classica e Generalizacao

Status: em aberto

## Problema

Como formular corretamente a validade externa do FG2P sem subestimar a comparacao classica com a literatura nem exagerar a extrapolacao para contextos fora do corpus-base?

## Ponto de partida consolidado

1. A comparacao classica com LatPhon e justa no nivel da fonte de dados, porque ambos se apoiam em `ipa-dict`.
2. O FG2P usa um conjunto de teste muito maior e estratificado, o que melhora a confianca da comparacao.
3. O projeto e mais forte ao comparar desempenho dentro do universo `ipa-dict` do que ao extrapolar para qualquer palavra do mundo real sem qualificacao.

## Tensao central

Ha dois sentidos diferentes de validade externa aqui:

### Validade externa classica

Capacidade de comparar honestamente com outros trabalhos que usam a mesma fonte de dados.

Neste sentido, a validade externa e alta.

### Validade externa de generalizacao ampla

Capacidade de afirmar desempenho robusto fora do universo efetivamente coberto pelo corpus.

Neste sentido, a validade externa ainda depende de evidencias adicionais, como testes OOV, neologismos e estabilidade com menos dados.

## Tese em elaboracao

O argumento mais forte talvez nao seja apenas "treinou em muito dado", mas:

- o modelo aprende com pouco dado relativamente bem;
- permanece estavel com muito dado;
- essa estabilidade sob ampliacao sugere aprendizado de estrutura, nao apenas ajuste local.

## O que falta

1. Decidir a redacao final para separar claramente os dois sentidos de validade externa.
2. Ligar essa nota aos testes de neologismos e palavras reais ineditas ja citados no artigo.
3. Definir se a evidencia atual ja basta para afirmar generalizacao robusta ou apenas generalizacao promissora.