ID: 061
Title: Experimentos controle como argumento narrativo (Exp0 bias, Exp107 inflacao)
Type: documentation
Priority: Medium
Status: Open

## Contexto

Varios experimentos foram "fracassados de proposito" — nao provam melhor ou pior,
existem como controle de parametros e artefatos de treinamento.
Eles reforcariam a argumentacao do artigo se documentados corretamente.

## Experimentos controle identificados

### A) Exp0 — Vicio de semente aleatoria (split bias)

Exp0 com semente nao-estratificada produziu PER 0.38% — artificialmente melhor que TUDO.
README.md documenta isso como "Metric Inflation" mas o ARTICLE.md NAO usa isso como argumento.

**Argumento que poderia fazer**: "sem estratificacao, a separacao aleatoria pode
concentrar palavras dificeis no treino e faceis no teste, inflando metricas.
Exp0 (PER 0.38%, nao-estratificado) vs Exp1 (PER 0.66%, estratificado) demonstra
que a diferenca de -41% PER vem da qualidade do split, nao do modelo."

Isso reforca diretamente a secao §2.2 (Estratificacao) e §5.5 Fase 1.

### B) Exp107 — Inflacao por treino excessivo (95% split)

Exp107 usa 95% treino / 4% val / 1% teste (~960 palavras).
PER 0.46% — aparentemente melhor que Exp104d (0.48%), mas:
- IC Wilson em 960 palavras: +/-3% (vs +/-0.03% em 28.782)
- Diferenca 0.46 vs 0.48 esta DENTRO do ruido
- Risco de memorizacao muito maior

**Argumento**: reforcar porque 60/10/30 e melhor que maximizar treino.

### C) Tamanho minimo de amostra

Ticket respondido: docs/evaluations/answered/003-tamanho-minimo-de-amostra.md

Formula conceitual documentada:
  N_practical = N_minimum_structural x safety_factor (2x-8x)
  
Criterios: cobertura de fonemas, bigramas, trigramas + margem para fatores ocultos.

Verificar se essa formula/conceito esta citado no ARTICLE.md ou se ficou so no ticket.

## Acao

1. Verificar se §2.2, §2.25 ou §5.5 ja mencionam Exp0 como controle de bias
2. Se nao, adicionar 1-2 paragrafos usando Exp0 e Exp107 como evidencia negativa
3. Citar ou referenciar ticket 003 (tamanho minimo) se a formula for relevante
4. Manter como "comentario adicional" se nao justificar secao propria

## Verificacao

- [ ] Exp0 como exemplo de split bias mencionado em §2.2 ou §5.5
- [ ] Exp107 como exemplo de inflacao por treino excessivo mencionado em §2.25
- [ ] Formula de tamanho minimo referenciada ou citada (se existir)

Dependencias: docs/evaluations/answered/003-tamanho-minimo-de-amostra.md
