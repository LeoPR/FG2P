# 009 - Regras para Comparacoes Empiricas Justas

Status: respondida

## Problema

Como garantir que comparacoes empiricas (ex.: "GPU 4x mais rapida", "modelo A melhor") sejam justificadas por criterios objetivos e verificaveis?

## Principio geral

Afirmacao comparativa so entra no texto principal quando tiver:

1. definicao de metrica;
2. dados de entrada e condicoes de medicao;
3. calculo explicito;
4. limite de escopo declarado.

## Checklist minimo (obrigatorio)

1. **Metrica comum**
   - Exemplo: PER para G2P, throughput para inferencia.
2. **Mesmo tipo de unidade**
   - Exemplo: palavras/s com batch=1 em ambos lados.
3. **Contexto de hardware declarado**
   - GPU/CPU, precisao (fp32/fp16), batch, ambiente.
4. **Incerteza quando aplicavel**
   - CI, desvio, repeticao, variacao entre runs.
5. **Formula de comparacao explicita**
   - Razao, delta absoluto, delta relativo.
6. **Escopo e limite da conclusao**
   - "neste setup" em vez de "universalmente".

## Formulas uteis para padronizar texto

Seja `x_A` o resultado do modelo A e `x_B` do modelo B:

1. **Delta absoluto**
   - `Δ = x_A - x_B`
2. **Delta relativo (%)**
   - `Δ_rel = (x_A - x_B) / x_B * 100`
3. **Fator multiplicativo (speedup)**
   - `f = x_A / x_B`

Exemplo de frase valida:

- "No setup X (batch=1, fp32), A atingiu 31.4 w/s e B 7.8 w/s, razao 4.0x (31.4/7.8)."

Exemplo de frase invalida:

- "A e 4x mais rapido" (sem metrica, sem condicao, sem calculo).

## Politica de redacao recomendada

1. Substituir termos absolutos por condicionais:
   - "melhor" -> "menor PER neste setup"
   - "mais rapido" -> "maior throughput nas condicoes medidas"
2. Sempre anexar uma linha de caveat em comparacoes cruzadas.
3. Quando nao houver dado equivalente na outra referencia, usar `n/d` e evitar inferencia cruzada.

## O que falta para fechar esta nota

1. Aplicar checklist em todo o README (secoes de metricas e performance).
2. Espelhar o mesmo padrao no ARTICLE.md.
3. Criar mini-template de frase comparativa permitida e proibida para edicoes futuras.