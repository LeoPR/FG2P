# Avaliacoes do Projeto

Esta pasta organiza avaliacoes tecnicas do FG2P em formato incremental.

Objetivo:
- registrar perguntas de avaliacao que surgem durante a pesquisa;
- separar o que ja foi respondido do que ainda precisa de evidencias ou calculos;
- manter um historico de raciocinio que possa depois ser consolidado no artigo ou removido se ficar obsoleto.

Convencao:
- `answered/`: perguntas com resposta documentada, ainda que sujeita a refinamento;
- `open/`: perguntas, riscos, hipoteses e pendencias que merecem investigacao adicional.

## Indice de Perguntas

| ID | Tema | Status | Observacao |
|----|------|--------|------------|
| 001 | Robustez da pesquisa: pontos fortes e fracos | Respondida parcialmente | Sintese inicial ja consolidada; pode ser refinada depois |
| 002 | Intervalo de confianca do WER | Respondida | Implementado em analyze_errors.py; formulas centralizadas em docs/article/FORMULAS.md |
| 003 | Tamanho minimo de amostra para teste e treino | Em aberto | Hipotese teorica + caminho para estimativa estatistica |
| 004 | Validade externa, comparacao classica e generalizacao | Em aberto | Comparacao com ipa-dict e limites de extrapolacao |
| 005 | Estratificacao incompleta e fatores ocultos | Em aberto | Questao sobre margem multiplicativa acima do minimo teorico |
| 006 | Curva de estabilidade com menos dados | Em aberto | Relacionar Exp104b, Exp105, Exp106 e possiveis novos pontos |
| 007 | Por que LatPhon nao reporta WER | Respondida | Paper usa PER como metrica principal e estatistica no nivel de fonema |
| 008 | DA Loss: forca, limites e narrativa justa | Em aberto | Matizar ganhos de PER classico, papel do override estrutural e validacao perceptual |
| 009 | Regras para comparacoes empiricas justas | Em aberto | Definir padrao de prova (dados, CI, hardware, escopo e limite da afirmacao) |
| 010 | Matriz consolidada de reivindicacoes | Em aberto | Revisao integrada (status, evidencia, risco e acao) para fechar pendencias por prioridade |
| 011 | Questionario de fechamento por etapas | Em aberto | Caminho feliz: perguntas decisorias para encerrar notas abertas sem ambiguidade |
| 012 | Clareza de nomenclatura no README | Respondida | Diferenciado FG2P (familia), ExpXXX (ID interno) e LatPhon (ano do paper) nos blocos iniciais |

## Ordem sugerida de aprofundamento

1. Formalizar o criterio de tamanho minimo de teste com alvo de precisao estatistica.
2. Formalizar o criterio de tamanho minimo de treino com cobertura estratificada + margem multiplicativa.
3. Fechar a nota 008 (DA Loss) com linguagem de contribuicao tecnica sem overclaim.
4. Definir e aplicar a nota 009 (padrao de comparacao empirica) no README e no artigo.
5. Fechar a nota 010, priorizando as reivindicacoes com status parcial/fragil.
6. Responder a nota 011 (questionario) e converter respostas em decisoes editoriais.

## Uso esperado

Quando surgir uma nova pergunta, adicionar primeiro uma nota em `open/` com:
- problema;
- por que importa;
- evidencias ja conhecidas;
- o que falta para responder com mais rigor.

Quando a pergunta estiver madura, mover o conteudo para `answered/` ou consolidar no artigo.