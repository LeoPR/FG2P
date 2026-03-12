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
| 002 | Intervalo de confianca do WER | Em aberto | Formula e plano documentados; implementacao pode entrar no codigo depois |
| 003 | Tamanho minimo de amostra para teste e treino | Em aberto | Hipotese teorica + caminho para estimativa estatistica |
| 004 | Validade externa, comparacao classica e generalizacao | Em aberto | Comparacao com ipa-dict e limites de extrapolacao |
| 005 | Estratificacao incompleta e fatores ocultos | Em aberto | Questao sobre margem multiplicativa acima do minimo teorico |
| 006 | Curva de estabilidade com menos dados | Em aberto | Relacionar Exp104b, Exp105, Exp106 e possiveis novos pontos |

## Ordem sugerida de aprofundamento

1. Fechar o CI do WER e decidir se entra no pipeline principal ou apenas na documentacao.
2. Formalizar o criterio de tamanho minimo de teste com alvo de precisao estatistica.
3. Formalizar o criterio de tamanho minimo de treino com cobertura estratificada + margem multiplicativa.
4. Revisar a narrativa de validade externa para deixar claro: mesmo dataset-base, subset diferente, comparacao ainda justa.
5. Decidir se as notas abertas viram secao do artigo ou permanecem apenas como caderno de avaliacao.

## Uso esperado

Quando surgir uma nova pergunta, adicionar primeiro uma nota em `open/` com:
- problema;
- por que importa;
- evidencias ja conhecidas;
- o que falta para responder com mais rigor.

Quando a pergunta estiver madura, mover o conteudo para `answered/` ou consolidar no artigo.