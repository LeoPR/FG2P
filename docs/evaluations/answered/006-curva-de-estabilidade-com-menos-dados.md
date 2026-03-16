# 006 - Curva de Estabilidade com Menos Dados

Status: respondida

## Problema

Como demonstrar, de forma clara e reproduzivel, que o modelo continua estavel quando a quantidade de dados de treino diminui?

## Motivacao

Se o modelo permanecer competitivo mesmo com reducao relevante de dados, isso reforca a tese de que ele aprendeu regularidades do sistema e nao apenas memorizou o corpus.

## Evidencias ja observadas

1. Exp104b funciona como referencia principal de desempenho.
2. Exp105 reduz a quantidade de treino e apresenta degradacao pequena.
3. Exp106 continua na mesma familia de desempenho mesmo com alteracao adicional no charset.

## Pergunta principal

Existe uma faixa de estabilidade em que a perda de desempenho por reducao de dados permanece pequena o suficiente para sustentar a tese de generalizacao estrutural?

## Caminho de resposta

1. Organizar os experimentos por percentual de treino.
2. Medir delta de PER, WER e metricas graduadas por reducao relativa de dados.
3. Estimar inclinacao da curva desempenho x volume de treino.
4. Identificar se existe um joelho de degradacao acentuada.

## Resultado desejado

Uma narrativa do tipo:

- acima de certo patamar, o modelo esta em regime estavel;
- abaixo desse patamar, a degradacao acelera;
- esse ponto pode servir como referencia empirica para `N_pratico`.

## O que falta

1. Consolidar uma tabela unica com `% treino`, `N treino`, `PER`, `WER` e possiveis CIs.
2. Ver se faltam pontos intermediarios entre 50% e 60%.
3. Decidir se a analise final sera apenas descritiva ou tambem modelada por regressao simples.

## Fechamento editorial (sem novo experimento)

Decisao adotada para esta versao:

1. A analise fica **descritiva**, usando os pontos ja disponiveis (Exp104b, Exp105, Exp106).
2. Conclusao editorial: ha evidencia de estabilidade no intervalo observado, sem declarar joelho universal.
3. Pontos intermediarios adicionais (ex.: 55%, 65%, 75%) ficam registrados como extensao futura opcional.