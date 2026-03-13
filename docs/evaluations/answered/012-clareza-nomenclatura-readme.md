# 012 - Clareza de nomenclatura no README

Status: respondida

## Pergunta

As tabelas comparativas do README usavam eixos de nomeacao inconsistentes: "FG2P (Exp104b)" (ID interno) ao lado de "LatPhon (2025)" (ano de publicacao). Isso confunde o leitor e nao segue a convencao de papers cientificos.

## Convencao cientifica de referencia

Em papers de NLP/speech (ACL, Interspeech, ICASSP), tabelas comparativas usam o mesmo eixo para todos os metodos:
- **Nome (Ano)** ou **Nome (Autores, Ano)** para identificar trabalhos
- Configuracao especifica vai em nota de rodape, caption ou coluna separada
- Trabalho proprio: "Ours", "This work" ou "Nome (Ano)"

## Decisao adotada

1. Tabelas de **comparacao externa** (FG2P vs LatPhon/WFST/ByT5):
   - "FG2P (2026)" e "LatPhon (2025)" — mesmo eixo (ano)
   - Nota curta antes da tabela explicando que os numeros vem da melhor configuracao (Exp104b) do estudo de ablacao

2. Tabelas **internas** (ablacao entre variantes do FG2P):
   - Mantidos IDs de experimento (Exp1, Exp9, Exp104b etc.)
   - Uma frase introdutoria explica o sistema de IDs antes das tabelas internas

3. Secao "Model Selection":
   - Frase introdutoria explica que aliases (best_per, best_wer) apontam para IDs internos

## O que foi removido

- Bloco verbose "How to read names in this section" (tentativa anterior, excessivamente literal)
- Rotulos longos como "FG2P (reference run: Exp104b)" e "LatPhon (paper, 2025)"

## Aplicacao

- README.md: tabelas Key Results e Baselines atualizadas
- ARTICLE.md: tabela comparativa atualizada
- docs internos (evaluations, experiments): mantidos como estao (sao notas de trabalho, nao publicacao)
