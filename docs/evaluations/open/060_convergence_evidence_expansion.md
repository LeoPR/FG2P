ID: 060
Title: §8.4 — Expandir evidencias de convergencia rapida
Type: documentation
Priority: Low
Status: Open

## Problema

§8.4 "Convergencia Rapida como Sinal de Qualidade" (linha 939) tem apenas 2 linhas.
Afirma que modelos melhores convergem mais rapido, mas nao apresenta:
- Dados numericos de convergencia por experimento (epocas ate "joelho")
- Comparacao entre modelos (Exp1 vs Exp9 vs Exp104b vs Exp104d)
- Referencia a literatura sobre velocidade de convergencia em G2P/seq2seq

## Dados disponiveis (verificar)

Os graficos de convergencia existem em results/ como curvas de loss.
Os metadados dos modelos (models/*_metadata.json) registram epocas de convergencia.

O "joelho" de treinamento (convergencia rapida inicial seguida de plateau)
e visivel nos graficos — falta quantificar.

## Acao

1. Verificar metadados: extrair epoca de early stopping para cada experimento
2. Tabela comparativa: Exp1, Exp5, Exp9, Exp104b, Exp104d — epocas, val_loss final
3. Se possivel, comparar com literatura (velocidade tipica de convergencia em BiLSTM G2P)
4. Se NAO houver dados comparaveis na literatura, marcar como "observacao sem baseline externo"

Nota: nao precisa justificar se e "rapido" ou "lento" sem referencia.
Basta mostrar que modelos melhores convergem consistentemente mais cedo.

## Verificacao

- [ ] §8.4 tem tabela com dados de convergencia
- [ ] Claim "convergencia rapida = qualidade" tem evidencia numerica
- [ ] Ausencia de comparacao com literatura esta declarada (se nao encontrar)

Dependencias: models/*_metadata.json
