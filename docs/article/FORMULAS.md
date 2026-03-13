# Fórmulas e Métrica s — Documentação Técnica Detalhada

**Propósito desta nota**: Centralizar todas as definições matemáticas, fórmulas, intervalos e implementações de métricas usadas no projeto FG2P.

Outros documentos (README, ARTICLE, evaluations) referenciam esta nota para detalhes técnicos, mantendo a narrativa principal legível.

**Navegação**:
- [1. Métricas Clássicas de G2P](#1-métricas-clássicas-de-g2p)
- [2. Intervalos de Confiança (Wilson)](#2-intervalos-de-confiança-wilson)
- [3. Métricas Graduadas (PanPhon)](#3-métricas-graduadas-panphon)
- [4. Comparações Estatísticas](#4-comparações-estatísticas)

---

## 1. Métricas Clássicas de G2P

### 1.1 Phoneme Error Rate (PER)

#### Definição
Taxa de erros no nível do fonema. Usa distância de edição de Levenshtein.

#### Fórmula
$$\text{PER} = \frac{\sum_i \text{edit\_distance}(pred_i, ref_i)}{\sum_i |ref_i|} \times 100$$

Onde:
- $pred_i$ = sequência de fonemas predita para a palavra $i$
- $ref_i$ = sequência de fonemas de referência para a palavra $i$
- $|ref_i|$ = comprimento da referência em fonemas
- $\sum$ é sobre todas as $N$ palavras do teste

#### Interpretação
- PER = 0: todas as transcrições perfeitas
- PER = 100: nenhum fonema correto
- Recomendado para: comparações com literatura em G2P, porque usa a mesma fórmula que a comunidade (Morris 2004, Bisani & Ney 2008)

#### Implementação
[src/analyze_errors.py:200-214](c:/Users/leona/OneDrive/Documents/Projects/Acadêmicos/FG2P/src/analyze_errors.py#L200)

```python
def calculate_per(predictions, references):
    """Calcula PER: soma(edit_distance) / soma(len_ref) * 100"""
    import editdistance
    total_errors = 0
    total_phonemes = 0
    for pred, ref in zip(predictions, references):
        total_errors += editdistance.eval(pred, ref)
        total_phonemes += len(ref)
    return (total_errors / total_phonemes * 100) if total_phonemes > 0 else 0.0
```

---

### 1.2 Word Accuracy

#### Definição
Taxa de palavras com transcrição fonética 100% correta (sem erros).

#### Fórmula
$$\text{Accuracy} = \frac{|\{i : pred_i = ref_i\}|}{N} \times 100$$

Onde:
- $N$ = total de palavras
- A igualdade é exata no nível de string (token-a-token)

#### Interpretação
- Accuracy = 100: todas as palavras perfeitas
- Accuracy = 0: nenhuma palavra correta
- Complemento do WER

#### Implementação
[src/analyze_errors.py:219-236](c:/Users/leona/OneDrive/Documents/Projects/Acadêmicos/FG2P/src/analyze_errors.py#L219)

```python
def calculate_accuracy(predictions, references):
    """Calcula acurácia exata por palavra"""
    if not predictions:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions) * 100
```

---

### 1.3 Word Error Rate (WER)

#### Definição
Taxa de palavras com **qualquer** erro (complemento da Accuracy em contexto G2P).

#### Fórmula
$$\text{WER} = 100 - \text{Accuracy} = \frac{|\{i : pred_i \neq ref_i\}|}{N} \times 100$$

#### Contexto do projeto
Em G2P, "Word Error Rate" difere do WER padrão de ASR:
- **ASR WER**: distância de edição em sequência de palavras numa frase (nível de sentença)
- **G2P WER**: proporção de palavras inteiras com qualquer erro (nível de palavra isolada)

O FG2P segue a definição de G2P, equivalente ao "String Error Rate" ou "Sentence Error Rate" de um fonema-por-palavra.

#### Interpretação
- WER = 0: todas as palavras perfeitas
- WER = 100: nenhuma palavra correta
- WER é auxiliar; PER é a métrica principal

#### Implementação
[src/analyze_errors.py:239-256](c:/Users/leona/OneDrive/Documents/Projects/Acadêmicos/FG2P/src/analyze_errors.py#L239)

```python
def calculate_wer(predictions, references):
    """Calcula WER = 100 - Accuracy"""
    return 100.0 - calculate_accuracy(predictions, references)
```

---

## 2. Intervalos de Confiança (Wilson)

### 2.1 Motivação

As métricas PER e WER são estimativas pontuais. Um intervalo de confiança quantifica a incerteza devida ao tamanho finito da amostra.

Existem várias fórmulas para calcular CIs para proporções:
1. **Wald CI** (simples): $p \pm z \sqrt{p(1-p)/n}$
2. **Wilson CI** (melhor): fórmula assimétrica mais conservadora
3. **Clopper-Pearson CI** (exato): mais conservador ainda, mas computacionalmente caro

O FG2P usa **Wilson CI** porque:
- Válido para amostras pequenas (n < 30)
- Válido para proporções extremas (p próximo de 0 ou 1)
- Exato sob distribuição binomial
- Sem o problema do Wald CI (pode gerar limite inferior negativo)

Referências: Wilson (1927), Brown, Cai & DasGupta (2001).

---

### 2.2 Wilson CI — Fórmula e Derivação

#### Fórmula Compacta

Para uma proporção Bernoulli com $k$ sucessos em $n$ observações:

$$\hat{p} = \frac{k}{n}$$

O intervalo de confiança $(L, U)$ para nível de confiança $\alpha$ é:

$$L = \frac{\hat{p} + \frac{z^2}{2n} - z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

$$U = \frac{\hat{p} + \frac{z^2}{2n} + z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

Onde $z = z_{\alpha/2}$ é o quantil da distribuição Normal padrão.

#### Valores de $z$ Comuns

| Confiança | $\alpha$ | $z$ |
|-----------|----------|-----|
| 90% | 0.10 | 1.6449 |
| 95% | 0.05 | 1.9600 |
| 99% | 0.01 | 2.5758 |

O FG2P usa $z = 1.95996$ para 95% de confiança.

#### Intuição Geométrica

1. Numenador: corrige a proporção amostral pela variância e margem
2. Denominador: normaliza pelo tamanho da amostra
3. Resultado: intervalo enviesado (assimétrico) que é mais conservador que Wald

---

### 2.3 Wilson CI para PER

O PER é mais complexo que uma proporção simples, porque é uma **razão de somas**:

$$\text{PER} = \frac{\sum_i d_i}{\sum_i n_i}$$

Onde:
- $d_i$ = edit distance da palavra $i$
- $n_i$ = comprimento da referência da palavra $i$

Para o intervalo de confiança do PER, o projeto usa uma **aproximação Normal**:

1. Calcular $\text{PER}_{\text{ponto}} = \frac{\sum d_i}{\sum n_i}$
2. Estimar variância como $p(1-p)/N_{\text{fonemas}}$
3. Aplicar Wilson CI com $k = \sum d_i$ e $n = \sum n_i$ (total de fonemas)

#### Implementação
[src/analyze_errors.py:259-295](c:/Users/leona/OneDrive/Documents/Projects/Acadêmicos/FG2P/src/analyze_errors.py#L259)

```python
def wilson_ci(count: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Intervalo de confiança de Wilson para uma proporção.
    
    Args:
        count: Número de "erros" (fonemas errados para PER)
        n:     Total de observações (total de fonemas para PER)
        confidence: Nível de confiança (padrão 0.95)
    
    Returns:
        (ci_lower, ci_upper) em % (multiplicado por 100)
    """
    import math
    if n == 0:
        return 0.0, 100.0
    
    # z-score para o nível de confiança
    if confidence == 0.95:
        z = 1.95996
    elif confidence == 0.99:
        z = 2.57583
    else:
        z = 1.95996  # fallback
    
    # Proporção amostral
    p_hat = count / n
    z2 = z * z
    denom = 1 + z2 / n
    
    # Centro e margem
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    
    # Limites, garantindo [0, 1]
    lower = max(0.0, center - margin) * 100
    upper = min(1.0, center + margin) * 100
    
    return lower, upper
```

#### Exemplo Operacional

Se o modelo tiver N_fonemas = 181.000 e errors = 1.050 (PER ≈ 0.58%):

```
wilson_ci(1050, 181000) → (~0.50%, 0.67%)
```

Interpretação: com 95% de confiança, o verdadeiro PER está entre 0.50% e 0.67%.

---

### 2.4 Wilson CI para WER

WER é uma proporção Bernoulli simples no nível da palavra:

$$\text{WER} = \frac{k}{n}$$

Onde:
- $k$ = número de palavras erradas
- $n$ = número total de palavras

Aplicar Wilson CI diretamente:

```python
wer_ci_low, wer_ci_high = wilson_ci(k, n, confidence=0.95)
```

#### Exemplo Operacional

Se o modelo tiver N_palavras = 28.782 e erros = 1.567 (WER ≈ 5.44%):

```
wilson_ci(1567, 28782) → (~5.17%, 5.70%)
```

Interpretação: com 95% de confiança, o verdadeiro WER está entre 5.17% e 5.70%.

#### Implementação no Código
[src/analyze_errors.py] — seção "MÉTRICAS CLÁSSICAS" (após cálculo do WER)

```python
# Após calcular PER, WER, Accuracy
per = calculate_per(predictions, references)
accuracy = calculate_accuracy(predictions, references)
wer = 100.0 - accuracy

# Calcular CIs
total_fonemas = sum(len(r) for r in references)
errors_fonemas = sum(editdistance.eval(p, r) for p, r in zip(predictions, references))
per_ci_low, per_ci_high = wilson_ci(errors_fonemas, total_fonemas)

total_palavras = len(predictions)
errors_palavras = sum(1 for p, r in zip(predictions, references) if p != r)
wer_ci_low, wer_ci_high = wilson_ci(errors_palavras, total_palavras)

# Exibir
logger.info(f"  PER:  {per:.2f}%  [{per_ci_low:.2f}%, {per_ci_high:.2f}%]")
logger.info(f"  WER:  {wer:.2f}%  [{wer_ci_low:.2f}%, {wer_ci_high:.2f}%]")
logger.info(f"  Accuracy: {accuracy:.2f}%")
```

---

## 3. Métricas Graduadas (PanPhon)

### 3.1 Espaço Fonético de PanPhon

O PanPhon (Donegan & Stampe, 2009; Mortillaro et al., 2014) representa fonemas como vetores de 24 features articulatórias binárias:

| Feature | Significado |
|---------|-------------|
| syl | syllabic |
| son | sonorant |
| cons | consonantal |
| cont | continuant |
| delrel | delayed release |
| nasal | nasal |
| voice | voice |
| ... | (18 features adicionais) |

Cada fonema é um vetor $\vec{v} \in \{0, 1\}^{24}$.

#### Distância Euclidiana Normalizada

$$d(f_i, f_j) = \frac{\sqrt{\sum_{k=1}^{24} (v_i^{(k)} - v_j^{(k)})^2}}{24}$$

Intervalo: $d \in [0, 1]$

---

### 3.2 Classificação de Erro por Distância

Os erros são classificados em 4 classes baseadas em distância PanPhon:

| Classe | Nome | Distância | Interpretação |
|--------|------|-----------|---------------|
| A | Exato | 0.000 | Fonema correto |
| B | Leve | ≤ 0.125 | ≤ 3 features diferentes |
| C | Médio | ≤ 0.250 | 4–6 features diferentes |
| D | Grave | > 0.250 | 7+ features diferentes (categorias diferentes) |

#### Motivação Fonológica

- **Classe B**: Alofones, neutralizações esperadas (e.g., /ɛ/→/e/, /o/→/ɔ/)
- **Classe C**: Erros dentro de famílias (e.g., /t/→/s/, fricativa vs oclusiva)
- **Classe D**: Erros entre categorias (e.g., /p/→/m/, oclusiva oral vs nasal)

---

### 3.3 Métricas Graduadas: Definições

#### WER Graduado (Graduated WER)

Pesa o erro de cada palavra pela pior classe fonológica nela:

$$\text{WER}_{\text{grad}} = 100 - \text{Score}_{\text{grad}}$$

Onde:

$$\text{Score}_{\text{grad}} = \frac{1}{N} \sum_i s_i$$

E $s_i$ é calculado como:

$$s_i = \begin{cases}
1.0 & \text{se } \text{worst\_class}(i) = A \text{ (palavra corre tal)} \\
0.75 & \text{se } \text{worst\_class}(i) = B \\
0.50 & \text{se } \text{worst\_class}(i) = C \\
0.0 & \text{se } \text{worst\_class}(i) = D
\end{cases}$$

#### PER Ponderado (Weighted PER)

Pondera cada fonema errado por sua distância:

$$\text{PER}_{\text{weighted}} = \frac{\sum_i d(pred_i, ref_i)}{N_{\text{fonemas}}} \times 100$$

Onde $d(\cdot, \cdot)$ é a distância normalizada em [0, 1].

#### Implementação
[src/phonetic_features.py](c:/Users/leona/OneDrive/Documents/Projects/Acadêmicos/FG2P/src/phonetic_features.py) — funções `graduated_metrics()` e correlatas.

---

### 3.4 Distribuição de Classes

Proporção de erros por classe:

$$p_{\text{class}} = \frac{|\{(p, r) : \text{class}(p, r) = \text{class}\}|}{N_{\text{erros}}}$$

Interpretação:
- Alto $p_B$: modelo erra inteligentemente (confunde sons similares)
- Alto $p_D$: modelo erra grave (confunde sons distantes)

---

## 4. Comparações Estatísticas

### 4.1 Comparação com LatPhon

| Métrica | LatPhon | FG2P Exp104b | Cálculo |
|---------|---------|--------------|---------|
| PER | 0.86% [0.56%, 1.16%] | 0.51% [0.48%, 0.54%] | via Wilson CI |
| N_teste | ~500 | 28.782 | 57× maior |
| IC Amplitude | ±0.30pp | ±0.03pp | 10× mais preciso |

Conclusão: Intervalos não se sobrepõem → diferença estatisticamente significativa a 95%.

### 4.2 Teste de Hipótese (não implementado, para trabalho futuro)

Para comparar dois modelos $M_1$ e $M_2$, pode-se usar:

1. **Bootstrap**: resampling com reposição dos erros, recompute CI
2. **Teste exato de Fisher**: para comparação de proporções 2×2
3. **Teste de McNemar**: para dados pareados (mesma amostra, dois modelos)

---

## Resumo de Referências

| Métrica | Implementação | Documento de Contexto |
|---------|---------------|----------------------|
| PER | src/analyze_errors.py:200 | ARTICLE.md, evaluations/001 |
| WER | src/analyze_errors.py:239 | ARTICLE.md, evaluations/002 |
| Wilson CI PER | src/analyze_errors.py:259 | evaluations/002 |
| Wilson CI WER | src/analyze_errors.py:259 (universal) | evaluations/002 |
| Graduadas | src/phonetic_features.py | ARTICLE.md §4.2 |

---

**Última atualização**: 2026-03-12
**Mantido por**: [GitHub Copilot — referência técnica]
