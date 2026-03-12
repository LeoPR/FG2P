# 002 - Intervalo de Confianca do WER

Status: em aberto

## Problema

O projeto enfatiza corretamente o PER como metrica principal, mas ainda e desejavel documentar o intervalo de confianca do WER para completar o quadro estatistico.

## Por que importa

1. Fecha a apresentacao estatistica das metricas classicas.
2. Facilita comparacoes com leitores que pensam primeiro em acerto por palavra.
3. Ajuda a mostrar que o WER e auxiliar, mas nao arbitrario.

## Hipotese atual

WER pode receber intervalo de confianca de Wilson diretamente, porque no nivel da palavra ele e uma proporcao Bernoulli:

- sucesso: palavra totalmente correta;
- falha: palavra com pelo menos um erro.

## Formula de trabalho

Seja:

- `n = total de palavras`
- `k = numero de palavras erradas`
- `p = k / n`

Entao o WER e `100 * p`, e o IC 95% pode ser calculado pela mesma funcao `wilson_ci()` ja usada em `src/analyze_errors.py`.

## O que falta

1. Confirmar se o projeto quer reportar IC para `WER` ou para `Accuracy`, com conversao posterior.
2. Decidir se o CI sera apenas mostrado no relatorio textual ou salvo tambem em JSON/CSV.
3. Implementar no codigo, caso seja desejado, e revisar o texto do artigo para evitar redundancia.

## Possivel saida futura

- `WER: 5.43% [5.15%, 5.72%]`
- `Accuracy: 94.57% [94.28%, 94.85%]`

## Observacao

Se isso for considerado excesso de detalhe para a narrativa principal, pode ficar apenas documentado aqui e no apendice metodologico.