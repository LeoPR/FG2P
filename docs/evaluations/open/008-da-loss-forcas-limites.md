# 008 - DA Loss: Forcas, Limites e Narrativa Justa

Status: em aberto

## Problema

Como documentar a DA Loss de forma tecnicamente forte, sem overclaim, separando:

1. o que foi demonstrado com evidencias limpas;
2. o que depende de condicoes especificas;
3. o que ainda precisa de validacao perceptual externa.

## Base tecnica da contribuicao

Formula usada:

`L = L_CE + λ · d_PanPhon(ŷ, y) · p(ŷ)`

Onde:

- `d_PanPhon`: distancia euclidiana normalizada em 24 features articulatorias;
- `p(ŷ)`: probabilidade softmax do fonema predito (escala a penalidade com confianca);
- `λ`: hiperparametro de ponderacao (sweep em Exp7, λ em {0.05, 0.10, 0.20, 0.50}).

## Pontos fortes (ja sustentados)

1. A multiplicacao por `p(ŷ)` e motivada: erros confiantes pesam mais que erros incertos.
2. Ha evidencia limpa na comparacao Exp1 vs Exp9 (mesma estrutura de output):
   - Classe D cai 0.39pp;
   - Classe B sobe 0.19pp.
3. Isso sustenta a tese de melhora de qualidade do erro (nao apenas contagem bruta).

## Pontos de atencao (devem aparecer na narrativa)

1. Magnitude do ganho de PER classico depende do contexto de output.
   - Exp102 (CE + sep) vs Exp103 (DA + sep): PER igual (0.53%).
2. O salto para Exp104b (0.49%) inclui efeito de distancia customizada para tokens estruturais (`.` e `ˈ`).
3. Portanto, a narrativa "DA Loss sozinha vence X" precisa ser matizada quando houver co-fatores ativos.

## Questao perceptual em aberto

DA Loss usa distancia articulatoria como proxy de impacto perceptual.

- Articulacao e percepcao sao correlacionadas, mas nao identicas.
- Afirmacoes como "Classe B imperceptivel em TTS" pedem validacao auditiva (ex.: MOS/ABX).

## O que falta para fechar esta nota

1. Tabela unica com comparacoes limpas (somente um fator variado por vez).
2. Separar explicitamente ganho por DA Loss e ganho por override estrutural.
3. Definir redacao final para README/ARTICLE sem extrapolacao causal indevida.
4. Registrar plano de validacao perceptual (quando aplicavel).