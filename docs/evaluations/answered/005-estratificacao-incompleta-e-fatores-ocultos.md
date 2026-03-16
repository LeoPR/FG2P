# 005 - Estratificacao Incompleta e Fatores Ocultos

Status: respondida

## Problema

A estratificacao atual e boa, mas provavelmente nao representa 100% das dimensoes relevantes do espaco fonetico real. Como documentar isso sem desvalorizar o resultado ja obtido?

## Hipotese

As 24 dimensoes articulatorias do PanPhon capturam uma base forte, mas nao esgotam toda a dinamica que participa da realizacao sonora e da sua impressao grafica indireta.

Exemplos de fatores potencialmente sub-representados:

- coarticulacao;
- velocidade de articulacao;
- intensidade e prosodia;
- restricoes biomecanicas e dinamicas de transicao entre sons.

## Consequencia pratica

Mesmo que um calculo teorico diga que um certo numero de palavras seria o minimo aceitavel, ainda pode ser prudente treinar com um multiplicador acima desse minimo para capturar combinacoes escondidas ou nao modeladas.

## Formula conceitual de trabalho

`N_pratico = N_minimo_estrutural * fator_de_seguranca`

onde o fator de seguranca pode ser testado em faixas como 2x, 4x e 8x.

## O que falta

1. Escolher um criterio para `N_minimo_estrutural`.
2. Definir como justificar o `fator_de_seguranca` sem arbitrariedade excessiva.
3. Relacionar essa ideia com os experimentos de reducao de dados ja realizados.

## Fechamento editorial (sem novo experimento)

Decisao adotada:

1. O conceito de `N_pratico = N_minimo_estrutural * fator_de_seguranca` fica mantido como **principio prudencial**.
2. Nao sera fixado um unico fator universal (2x/4x/8x) nesta versao, para evitar arbitrariedade sem validacao dedicada.
3. A justificativa textual passa a ser: a estratificacao atual e forte, mas dimensoes ocultas podem existir; por isso, conclusoes de minimo amostral sao apresentadas com cautela.