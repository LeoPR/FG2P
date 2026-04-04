ID: 059
Title: §8.2 — Revisar hipotese de memorizacao em modelos 17.2M
Type: documentation
Priority: High
Status: Open

## Problema

§8.2 (linha 915) afirma:
"Em 17,2M parametros (Exp10), DA Loss interfere negativamente — o modelo grande tem
capacidade suficiente para memorizar, e a penalizacao fonetica atrapalha esse processo."

Porem, Exp104d (tambem 17.2M parametros) alcanca o MELHOR PER (0.48%).

### Tabela de evidencias 17.2M

| Exp   | Params | Loss          | Sep | PER   | Nota |
|-------|--------|---------------|-----|-------|------|
| Exp2  | 17.2M  | CE            | nao | 0.60% | CE puro |
| Exp10 | 17.2M  | DA 0.2        | nao | 0.61% | DA "falha" |
| Exp104d| 17.2M | DA 0.2 + dist | sim | 0.48% | MELHOR PER |

O que mudou entre Exp10 e Exp104d NAO foi o tamanho do modelo — foi:
1. Separadores silabicos (sim vs nao)
2. Distancias customizadas corrigidas (override pos-normalizacao)
3. Possivelmente interacao com o split de dados

### Conclusao necessaria

A explicacao "17.2M memoriza, DA atrapalha" esta INCOMPLETA:
- Exp104d prova que 17.2M + DA pode ser o melhor
- A variavel confundida e a presenca de separadores + correcao de distancias
- O tamanho do modelo NAO e a explicacao unica

## Acao

Revisar §8.2 para:
1. Reconhecer que Exp104d refuta a hipotese simples de "tamanho = memorizacao"
2. Atribuir o resultado de Exp10 a AUSENCIA de separadores + distancias, nao a tamanho
3. Reformular: "DA Loss em 17.2M sem separadores nao melhora (Exp10), mas COM
   separadores e distancias corrigidas produz o melhor PER (Exp104d)"
4. Marcar a hipotese de memorizacao como "observacao preliminar que requer
   ablacao dedicada" — nao como conclusao

## Verificacao

- [x] §8.2 nao afirma categoricamente que 17.2M memoriza (reformulado 2026-04-04)
- [x] Exp104d citado como contra-evidencia (§8.2 e §5.5 Fase 2 atualizados)
- [x] Hipotese de memorizacao marcada como "requer ablacao dedicada" (§8.2)
- [x] Tabela §5.2 Exp10 atualizada: "DA sem sep: sem ganho vs CE (ver §8.2)"

Dependencias: nenhuma
