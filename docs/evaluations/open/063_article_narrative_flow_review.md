ID: 063
Title: Revisao de fluxo narrativo — ordem, duplicacoes, spoilers
Type: documentation
Priority: Low
Status: Open

## Problema

Checagem geral da forma como a historia do artigo esta sendo contada.
Sem reescrever o artigo, apenas identificar:
- Explicacoes fora de ordem (conceito usado antes de ser apresentado)
- Duplicacoes (mesmo conceito explicado em dois lugares diferentes)
- "Spoilers" (resultados antecipados antes de serem devidamente contextualizados)
- Forward references que precisam de resumo minimo antes de apontar para capitulo futuro

## Pontos ja conhecidos

1. §2.25 "Memorizacao vs Aprendizado" menciona Exp107 (0.46%) antes de §5 (Resultados)
   — pode ser um spoiler; avaliar se precisa de nota "ver §5 para detalhes"

2. §3.5 "Protocolo de Treinamento" discute Exp0 (0.38%) e regime de treino antes
   dos resultados experimentais — e contexto de design, nao spoiler, mas verificar

3. §4.2 "DA Loss" tem exemplo numerico detalhado (§4.2 fluxo passo a passo) que
   e pedagogico mas longo — avaliar se pertence ao corpo ou a apendice

4. §5.3 "Metricas Graduadas" repete definicoes de PER_w/WER_g ja dadas em §5.1
   — verificar se ha duplicacao real ou apenas resumo

5. §5.5 "Principais Descobertas" resume tudo de §5.1-5.4 — avaliar se e redundante
   ou se funciona como "executive summary" util

## Acao

1. Ler §1-§9 na sequencia, anotando inconsistencias de fluxo
2. Classificar cada uma como: spoiler, duplicacao, fora de ordem
3. Para cada: sugerir correcao minima (mover, adicionar forward ref, ou deletar)
4. NAO reescrever o artigo — apenas produzir lista de ajustes pontuais

## Verificacao

- [ ] Lista de ajustes produzida (pode ser adicionada neste ticket)
- [ ] Nenhuma secao faz referencia a conceito nao introduzido sem forward ref
- [ ] Duplicacoes identificadas e resolvidas (manter ou remover uma copia)

Dependencias: tickets 057-062 (resolver questoes factuais antes de revisar fluxo)
