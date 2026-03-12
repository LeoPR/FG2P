# 001 - Robustez da Pesquisa

Status: respondida parcialmente

Escopo desta nota:
- registrar os pontos fortes e fracos ja identificados na avaliacao da robustez da pesquisa;
- preservar as respostas discutidas durante a revisao tecnica;
- deixar claro o que e conclusao atual e o que ainda pode ser revisado.

## Pergunta

Quais sao os principais pontos fortes e fracos da robustez metodologica do projeto FG2P, sem modificar o codigo neste momento?

## Resposta consolidada

### Pontos fortes

1. O split 60/10/30 foi validado com criterios complementares de distribuicao, incluindo qui-quadrado e tamanho de efeito. Isso fortalece a validade interna do protocolo experimental.
2. O conjunto de teste e grande para o padrao da literatura em PT-BR. O uso de 28.782 palavras reduz a incerteza estatistica e torna a comparacao mais estavel.
3. A reprodutibilidade operacional esta bem amarrada por checksum do TSV, seed persistida em metadata e configuracoes salvas por experimento.
4. A avaliacao nao depende apenas de WER. O projeto mede qualidade dos erros com PER, metricas graduadas e classes A/B/C/D baseadas em distancia articulatoria.
5. A funcao de custo com distancia fonetica foi acompanhada de analise conceitual e de ablacoes, em vez de ser apresentada como caixa-preta.
6. A documentacao experimental distingue comparacoes limpas de comparacoes confundidas, o que reduz risco de conclusoes indevidas.
7. O projeto documenta comportamento com menos dados e mostra degradacao pequena entre configuracoes proximas, o que sugere estabilidade acima do limiar minimo de aprendizado.

### Pontos fracos

1. O vocabulario e construido a partir do corpus completo, o que elimina uma classe de OOV estrutural durante a avaliacao principal. Isso nao invalida o resultado, mas limita a interpretacao sobre robustez a simbolos realmente ausentes.
2. Ainda faltava, no momento desta nota, o intervalo de confianca formal do WER no pipeline principal. Como WER e auxiliar, isso nao compromete a tese central, mas seria um complemento metodologico util.
3. O projeto depende de uma unica fonte-base principal. A comparacao classica com a literatura e justa porque usa o mesmo universo `ipa-dict`, mas a generalizacao para fora desse universo ainda precisa ser delimitada com cuidado.
4. Ainda nao existe uma formalizacao fechada do tamanho minimo de treino e de teste que preserve distribuicoes e confianca estatistica sob hipoteses explicitas.
5. A estratificacao atual captura fatores importantes, mas nao exaure todas as dimensoes possiveis do espaco fonetico. Fatores ocultos ou nao observados podem exigir margem acima do minimo teorico.

## Ajustes importantes surgidos depois da primeira resposta

### Validade externa

A avaliacao anterior subestimava a validade externa da comparacao classica. O ponto correto e:

- FG2P e LatPhon usam o mesmo universo de referencia, `open-dict-data/ipa-dict`.
- A comparacao e justa no nivel da fonte de dados.
- A diferenca principal esta no tamanho e na selecao do conjunto de teste, nao na origem do corpus.

Conclusao atual:

- para comparacao classica com a literatura baseada em `ipa-dict`, a validade externa e alta;
- para extrapolacao a dominios fora desse universo, a validade externa ainda precisa de qualificacao.

### Papel do PER e do WER

- O PER e a metrica principal porque o problema e fonetico.
- O WER funciona como controle auxiliar de acerto integral por palavra.
- As metricas graduadas corrigem a cegueira do WER para a qualidade dos erros.

## Pendencias ligadas a esta pergunta

1. Decidir se o CI do WER entra no codigo principal ou apenas na documentacao.
2. Formalizar a narrativa de tamanho minimo de amostra.
3. Transformar a intuicao sobre estabilidade com menos dados em argumento estatistico reproduzivel.
4. Revisar se esta nota deve ser absorvida parcialmente por `docs/article/ARTICLE.md` depois.

## Referencias internas uteis

- `docs/article/ARTICLE.md`
- `docs/article/EXPERIMENTS.md`
- `docs/report/performance.json`
- `src/analyze_errors.py`
- `src/g2p.py`