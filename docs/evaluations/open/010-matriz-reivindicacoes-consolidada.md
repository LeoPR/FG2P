# 010 - Matriz Consolidada de Reivindicacoes

Status: em aberto

Objetivo desta nota:

1. consolidar a tabela do novo avaliador com o que ja foi auditado em `001`, `007`, `008` e `009`;
2. reduzir redundancia entre notas;
3. definir fechamento por prioridade com criterios verificaveis.

## Matriz consolidada (status + evidencia + acao)

| Reivindicacao | Status | Evidencia atual | Risco de overclaim | Acao para fechar |
|---|---|---|---|---|
| DA Loss redistribui Classe D -> B | Sustentada | Exp1 vs Exp9 (mesma estrutura de output): D -0.39pp, B +0.19pp | Baixo | Manter como evidencia principal de qualidade de erro |
| FG2P supera LatPhon em PT-BR | Parcial | PER com ICs nao sobrepostos no setup reportado | Medio | Qualificar no texto: comparacao em PER no escopo `ipa-dict`; evitar universalizacao |
| Metodo > arquitetura (BiLSTM > Transformer) | Fragil | Sem controle de variavel de foco (mono vs multi) | Alto | Trocar por formulacao condicional: "neste setup, com estes dados" |
| Split 60/10/30 supera 70/10/20 | Sustentada | Comparacoes internas com mesmo pipeline e melhoria consistente | Baixo | Manter como contribuicao metodologica |
| Modelo aprende regras, nao memoriza | Sustentada com ressalva | OOV + estabilidade com menos dados apontam na direcao correta | Medio | Declarar tamanho pequeno do OOV set como limite atual |
| DA Loss e contribuicao original | Nao verificada | Nao ha revisao de related work fechada nesta trilha | Alto | Abrir revisao dedicada de originalidade com fontes primarias |
| PER_w e WER_g agregam informacao real | Sustentada | Metrica graduada capta severidade alem do PER classico | Baixo | Manter e linkar a `FORMULAS.md` |

## Ajustes de consistencia com notas anteriores

1. Alinha com `007`: LatPhon escolheu PER como metrica central (nao e "omissao acidental" de WER).
2. Alinha com `008`: ganho de PER classico depende do contexto (DA isolada vs override estrutural).
3. Alinha com `009`: qualquer claim comparativo precisa de metrica comum + condicao + calculo + limite.

## Pontos que podem ser considerados "quase fechados"

1. Redistribuicao de erro por DA Loss (classe D -> B).
2. Robustez do split 60/10/30 como decisao metodologica.
3. Utilidade de metricas graduadas como complemento do PER classico.

## Pontos ainda realmente abertos

1. Originalidade da DA Loss frente a trabalhos proximos (ASR/G2P com distancia fonologica).
2. Matizacao final da frase "metodo > arquitetura" para evitar causalidade nao isolada.
3. Validacao perceptual (MOS/ABX) para sustentar afirmacoes de impacto auditivo.

## Prioridade de fechamento

1. Alta: remover/ajustar claims frageis em README/ARTICLE (arquitetura, superioridade universal).
2. Alta: abrir nota de originalidade com checklist de related work e criterio de novidade.
3. Media: definir plano minimo de validacao perceptual (mesmo que como trabalho futuro formalizado).
4. Media: transformar esta matriz em secao curta no artigo, com linguagem "evidencia + limite".

## Criterio de aceite para marcar "respondida"

A nota 010 so passa para "respondida" quando:

1. cada item da tabela tiver status final + justificativa curta;
2. os itens "parcial" e "fragil" tiverem redacao corrigida no README/ARTICLE;
3. houver registro explicito do que ficou para trabalho futuro, sem ambiguidade.