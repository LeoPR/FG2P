# 007 - Por que LatPhon nao reporta WER

Status: respondida

## Pergunta

Por que o paper LatPhon (2509.03300v1) nao usa WER nas explicacoes principais?

## Resposta curta

Porque o desenho de avaliacao do paper foi centrado em PER (erro por fonema) desde o resumo ate os testes estatisticos. Nao ha indicio de que WER tenha sido metrica-alvo do trabalho.

## Evidencias no paper (texto extraido)

Fonte usada nesta nota:

- `docs/2509.03300v1.pdf`
- extracao local para busca textual: `docs/latphon_2509_03300v1_extracted.txt`

Evidencias diretas:

1. O resumo e as contribuicoes falam em PER como metrica central.
   - "mean phoneme error rate (PER)" (linhas iniciais do resumo)
   - "Head-to-head PER comparison" nas contribuicoes.

2. A tabela principal de resultados (Table II) e somente PER com IC de Wilson.
   - "PER (%) WITH WILSON 95% CONFIDENCE INTERVALS".

3. A significancia estatistica reportada e no nivel de erro fonemico.
   - "two-proportion z-test on phoneme-level error counts".

4. O paper explicita limitacao de disponibilidade de contagens para comparacoes externas.
   - Para ByT5, eles afirmam que nao havia contagens brutas publicadas para construir CIs.

## Interpretacao metodologica

1. O trabalho LatPhon e orientado a comparacao em PER, que e padrao em G2P.
2. O foco estatistico no nivel de fonema e coerente com a tarefa modelada no paper.
3. A ausencia de WER nao implica falha; indica escopo de metrica escolhido.

## Implicacao para FG2P

1. Comparar FG2P x LatPhon em PER continua apropriado.
2. Incluir WER em FG2P e um plus de transparencia, mas nao pode ser usado para confronto direto quando a outra ponta nao reporta WER.
3. Toda comparacao deve explicitar: metrica comum, tamanho amostral, incerteza e limites de escopo.

## Proximo passo recomendado

Usar a nota 009 (comparacoes empiricas justas) como checklist obrigatorio antes de declarar vantagem em qualquer metrica.