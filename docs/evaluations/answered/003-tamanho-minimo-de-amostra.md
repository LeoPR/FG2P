# 003 - Tamanho Minimo de Amostra para Teste e Treino

Status: respondida

## Problema

Qual e a menor quantidade de palavras necessaria para:

1. testar com confianca estatistica suficiente;
2. treinar sem cair abaixo do limiar de representatividade do espaco fonetico.

## Intuicao central

Nem todo aumento de dados agrega nova informacao relevante na mesma taxa. Se um subconjunto pequeno ja representar bem as distribuicoes essenciais, um modelo ideal poderia induzir boa parte das regras do sistema a partir dele.

## Hipotese de trabalho

Ha dois minimos diferentes:

### Minimo de teste

Definido pela precisao desejada do intervalo de confianca.

Pergunta operacional:

- quantas palavras sao necessarias para que o IC do PER ou do WER tenha largura aceitavel?

### Minimo de treino

Definido por cobertura suficiente de regularidades estruturais, incluindo:

- estratos ja modelados;
- cobertura de fonemas;
- cobertura de bigramas e trigramas;
- margem para fatores nao observados explicitamente.

## Ideia da margem multiplicativa

Mesmo com estratificacao valida, o espaco real pode ter dimensoes ocultas ou apenas nao modeladas, como:

- coarticulacao;
- dinamica temporal;
- variacoes prosodicas;
- restricoes articulatorias indiretas que nao aparecem integralmente nas 24D do PanPhon.

Por isso, o minimo teorico nao deve ser usado sem folga. Uma margem multiplicativa de 2x, 4x ou 8x acima do minimo estrutural pode ser uma forma prudente de garantir que o modelo nao esteja operando no limite inferior de percepcao estatistica.

## Evidencias internas ja disponiveis

1. Exp104b usa 60% de treino e serve como referencia principal.
2. Exp105 reduz o treino para 50% e sofre degradacao pequena de PER.
3. Exp106 adiciona filtro de hifen e tambem continua em faixa competitiva.

Esses pontos sugerem que o projeto esta acima do limiar minimo de treino, mas ainda nao definem formalmente onde esse limiar comeca.

## O que falta

1. Escolher um criterio estatistico para tamanho minimo de teste.
2. Escolher um criterio estrutural para tamanho minimo de treino.
3. Definir a margem multiplicativa desejada e a justificativa conceitual.
4. Decidir se a conclusao final sera teorica, empirica ou hibrida.

## Fechamento editorial (sem novo experimento)

Decisao adotada para publicacao atual:

1. Conclusao **hibrida**:
	- teste: criterio estatistico (precisao de IC);
	- treino: criterio estrutural + evidencia empirica interna de estabilidade.
2. O trabalho **nao afirma minimo universal** de amostra.
3. Formula-se apenas um criterio pratico: o regime usado no FG2P esta acima do limiar observado para estabilidade no recorte atual.
4. A estimativa formal de `N_minimo` (com modelagem dedicada) fica como extensao metodologica futura.