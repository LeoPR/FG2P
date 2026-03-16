# 020 — Proporcionalidade chars/s e Condições de Pico por Hardware

## Perguntas

1. Existe uma relação de proporcionalidade entre throughput e comprimento de entrada (chars)? O modelo é N vezes mais rápido com palavras N vezes mais curtas?
2. Em que condições exatas GPU e CPU atingem o pico de throughput — qual o batch_size crítico, e isso varia por modelo (hidden size)?
3. O número de núcleos do CPU influencia diretamente o throughput de inferência do LSTM?

---

## Status: respondida

Os dados do sweep formal (GPU 2026-03-14 + CPU 2026-03-15) e dos CSVs brutos por `chars_in` permitem resposta técnica consolidada para as três perguntas, com ressalva explícita de que a pergunta de núcleos CPU permanece no nível de explicação mecanística/profiling (não como bloqueador de conclusão operacional).

---

## Pergunta 1 — Proporcionalidade throughput ~ chars/s

### Evidência consolidada (batch=1, GPU, Exp104d — raw CSV overnight)

Latência mediana por comprimento de entrada:

| chars | p50 latência | throughput (w/s) | throughput (chars/s) |
|-------|-------------|------------------|---------------------|
| 3 | 13.6 ms | 73 w/s | 221 chars/s |
| 5 | 18.4 ms | 54 w/s | 272 chars/s |
| 7 | 24.0 ms | 42 w/s | 292 chars/s |
| 9 (mediana) | 29.3 ms | 34 w/s | 307 chars/s |
| 12 | 37.2 ms | 27 w/s | 323 chars/s |
| 17 | 53.2 ms | 19 w/s | 320 chars/s |

**Observação imediata**: `chars/s` NÃO é constante — aumenta com o comprimento da palavra e estabiliza em ~310–325 chars/s a partir de ~chars=9.

### Modelo físico esperado

A latência do FG2P é composta por:

```
T(n_chars, n_tokens) = T_overhead + T_encoder(n_chars) + T_decoder(n_tokens)
```

Onde:
- `T_overhead`: constante por chamada (kernel CUDA launch, Python call, tensor allocation) ≈ 5–10 ms em batch=1
- `T_encoder(n_chars)`: BiLSTM O(n_chars) — linear em caracteres
- `T_decoder(n_tokens)`: autoregressivo O(n_tokens) — linear em fonemas de saída

Como `n_tokens ≈ f(n_chars)` (palavras mais longas geram mais fonemas), temos aproximadamente:

```
T ≈ a + b × n_chars
```

Onde `a` = overhead fixo, `b` = marginal cost por char.

**Consequência**: `throughput_chars = n_chars / T = n_chars / (a + b × n_chars) = 1 / (a/n_chars + b)`

Para `n_chars` pequeno: throughput_chars → 0 (overhead domina)
Para `n_chars` grande: throughput_chars → 1/b (constante, throughput máximo por char)

Isso explica a curva observada: chars/s aumenta e estabiliza. Não é uma proporção simples.

### Implicação prática

**A velocidade NÃO é proporcional a chars/s de forma linear.** A unidade mais informativa depende do regime:
- Para palavras curtas (n_chars < 6): o overhead fixo domina → palavras de 3 chars levam >50% do tempo em overhead
- Para palavras longas (n_chars > 10): throughput por char se estabiliza → chars/s é uma boa normalização
- Para comparações entre modelos: usar `tokens/s` (decoder), que reflete melhor o trabalho computacional

### O que falta para resposta completa

- [ ] Ajuste de regressão: estimativa de `a` e `b` da curva `T = a + b × n_chars` com dados por comprimento
- [ ] Separar contribuição encoder vs decoder (n_chars vs n_tokens análise bivariada)
- [ ] Validar com batch>1: em batch grande, o overhead fixo `a` é amortizado; espera-se que chars/s se torne mais estável

---

## Pergunta 2 — Condições de pico por hardware e modelo

### Resposta consolidada (dados sweep formal GPU+CPU)

**GPU (RTX 3060, hidden=384):**

| Ponto crítico | Batch size | Throughput | p50 | Característica |
|--------------|-----------|-----------|-----|----------------|
| Início de aceleração | 1 | 34 w/s | 28 ms | baseline, overhead Python mínimo |
| Aceleração máxima relativa | 16→32 | 264→406 w/s | 3.7→2.4 ms | GPU paralelismo sub-linear |
| Início de saturação | 128→256 | 745→976 w/s | 1.35→1.03 ms | p50 ainda caindo mas devagar |
| **Pico confirmado** | **512** | **1.106 w/s** | **0.89 ms** | plateau |
| Além do pico | 1024 | ~891 w/s | ~1.1 ms | regressão (overhead de gestão) |

**CPU (Xeon 36 cores, hidden=384):**

| Ponto crítico | Batch size | Throughput | p50 | Característica |
|--------------|-----------|-----------|-----|----------------|
| Baseline | 1 | 21.6 w/s | 46 ms | MKL multi-thread já ativo |
| Saturação | 64 | 162.7 w/s | 5.52 ms | p50 estabiliza |
| **Pico** | **128** | **184.0 w/s** | **5.54 ms** | <5% ganho além de batch=64 |

### Dependência por hidden size (síntese operacional)

Modelos com `hidden` maior têm matrizes W_hh maiores:
- hidden=256: W_hh = 4×256×256 = 262K params — pode saturar GPU em batch menor
- hidden=384: W_hh = 4×384×384 = 590K params — satura em batch≈512 (observado)
- hidden=768 (Exp107): W_hh = 4×768×768 = 2.36M params — pode saturar em batch≤128

O sweep inclui 19 modelos (hidden=256 a hidden=768) e confirma que o ponto de saturação varia com a carga computacional por passo e comprimento médio de saída. Para decisão prática de operação, já há base suficiente por família de modelo.

**Range observado entre modelos no batch=512:**
- 1,081–1,500 w/s (min=Exp104d, max=Exp106 ou modelos com menos separadores/tokens de saída)

### O que falta

- [ ] Estratificação do batch_size_crítico por `hidden_size` (dados disponíveis no sweep_summary CSV)
- [ ] Análise formal de ponto de inflexão (segunda derivada do throughput vs batch_size)

---

## Pergunta 3 — Influência do número de núcleos CPU

### Resposta consolidada da pergunta

Esta pergunta foi parcialmente abordada em doc 018 §4 (Xeon 36 cores vs batch scaling). Resumo:

- **Núcleos NÃO são usados em paralelo** para inferência single-word batch=1: o processamento é serial (LSTM autoregressivo)
- **MKL usa múltiplos threads** para as operações matriciais W_hh dentro de cada passo — `num_threads=1` causou −61% (doc 016)
- **O número "útil" de threads** para W_hh (4×384×384) no MKL provavelmente satura em 4–8 threads; threads adicionais contribuem marginalmente

**Questão residual (não bloqueante)**: Quantos dos 36 cores efetivamente participam por chamada em batch=1 vs batch=128?

Isso requer profiling com `torch.profiler` ou `Intel VTune`. O experimento seria:
```python
import torch
torch.set_num_threads(N)  # variar N=1,2,4,8,16,36
# medir throughput para batch=1 e batch=128
```

A hipótese operacional permanece: throughput satura em N≈4–8 para batch=1 e em N≈8–16 para batch=128 (matrizes maiores justificam mais threads).

---

## Conclusão operacional

1. `chars/s` não segue proporcionalidade linear com tamanho da palavra em batch=1; há termo fixo de overhead dominante em entradas curtas.
2. Picos de throughput por hardware estão tecnicamente estabelecidos para o protocolo atual: GPU em batch≈512 e CPU em batch≈64–128.
3. A análise por núcleos de CPU já é suficiente para decisão de engenharia (escalar batch e evitar inferência unitária como proxy de throughput), mesmo sem profiling fino por thread.
4. A pergunta fica encerrada para uso documental/operacional; profiling aprofundado de threads permanece opcional como trabalho futuro de micro-otimização.

---

## Referências de dados existentes

- Raw CSVs por `chars_in`: `results/exp104d_*/benchmark_*_cuda_raw.csv`
- Sweep GPU summary: `results/benchmarks/sweep_summary_20260314_113752.csv`
- Análise de variância e comprimento: `docs/benchmarks/BENCHMARK.md §Análise de variância`
- Métrica chars/s: `src/benchmark_inference.py → _analyze() → throughput_cps_in`
