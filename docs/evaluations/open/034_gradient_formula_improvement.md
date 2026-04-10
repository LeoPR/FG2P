ID: 034
Title: [v2.x] Melhoria da formula de gradiente (balanceamento DA Loss)
Type: research
Priority: Medium
Status: Open

## Contexto

A DA Loss atual tem a forma:

```
L = L_CE + lambda * d_panphon(y_hat, y) * p(y_hat)
```

Onde:
- `L_CE = -log p_correto` (cross-entropy padrao)
- `d_panphon in [0, 1]` (distancia articulatoria normalizada)
- `p(y_hat) in (0, 1]` (confianca na predicao errada)
- `lambda = 0.20` (fixo, otimizado por sweep no v1.x)

**Observacao do autor (2026-04-10)**: este balanceamento e considerado
**basico**. Funciona como prova de conceito mas tem limitacoes estruturais:

## Limitacoes identificadas

### 1. Lambda fixo e nao-adaptativo
O peso `lambda=0.20` foi encontrado empiricamente, mas:
- Nao se ajusta a dificuldade do exemplo
- Nao se ajusta a fase de treinamento (early vs late epochs)
- Pode ser subotimo em outras linguas ou outros tamanhos de vocabulario

### 2. Produto multiplicativo simples
A forma `d * p(y_hat)` assume que os dois fatores sao linearmente independentes.
Na pratica:
- Quando `p(y_hat)` e alto (modelo confiante), pode amplificar demais o gradiente
- Quando `p(y_hat)` e baixo (modelo na duvida), o sinal fica fraco demais

### 3. Nao considera a distribuicao completa da softmax
So olha para o argmax (`y_hat`). Ignora:
- Probabilidade atribuida a fonemas proximos do alvo (bom sinal)
- Entropia da distribuicao (medida de incerteza)
- Candidatos secundarios (top-k)

### 4. Distance override manual para tokens estruturais
A correcao `d=1.0` para pares envolvendo `.` e `ˈ` e manual e nao se propaga
naturalmente. Uma formulacao melhor evitaria essa excecao.

## Direcoes exploratorias

### Opcao A: Lambda adaptativo
```
lambda(t) = lambda_0 * f(CE_t)
```
Onde `f` e uma funcao da loss atual — DA ativa mais forte na zona de transicao
(CE entre 0.3 e 1.5) e menos nos extremos.

### Opcao B: Formula baseada em expectativa
```
L = L_CE + lambda * E_{y' ~ softmax}[d(y', y)]
```
Integra sobre toda a distribuicao, nao so o argmax. Mais caro computacionalmente
mas teoricamente mais fundamentado.

### Opcao C: Divergencia KL com target "phonologically smoothed"
Em vez de one-hot, o target seria uma distribuicao suave sobre fonemas proximos
(inspirado em label smoothing mas com distancia articulatoria como pesos):
```
target_smooth(y_i) = (1 - alpha) * 1_{y_i=y} + alpha * (1 - d(y_i, y)) / Z
L = KL(target_smooth || softmax_output)
```
Elimina completamente o termo multiplicativo e unifica CE e DA em uma unica
perda KL.

### Opcao D: Focal-inspired weighting
Inspirado em Focal Loss (Lin et al. 2017), modular por dificuldade:
```
L = L_CE + lambda * (1 - p_correto)^gamma * d(y_hat, y)
```
Penaliza mais exemplos dificeis.

## Criterios de aceite

- [ ] Literatura revisada sobre loss functions para structured prediction
  (label smoothing, focal loss, distillation losses, knowledge distillation)
- [ ] Pelo menos 3 variantes implementadas e comparadas em baseline PT-BR
- [ ] Variante vencedora comparada com v1.x em mesmas condicoes
- [ ] Analise teorica da nova formula (gradiente, propriedades de convergencia)
- [ ] Contribuicao publicavel no Paper C (v2.x)

## Dependencias

- Ticket 025 (espaco 7D): nova metrica de distancia pode mudar a formula
- Ticket 035 (Transformer): arquitetura pode interagir com a formula
- Supplementary `DA_LOSS_ANALYSIS.md`: contem analise matematica atual

## Proximos passos

1. Levantamento bibliografico sobre structured prediction losses
2. Implementacao das 4 variantes como loss classes em `src/losses/`
3. Sweep comparativo em PT-BR (mesmo setup do v1.x)
4. Selecao de variante vencedora e analise teorica
5. Experimentos multilingues (apos ticket 026 estar maduro)
