# 018 — Guia Prático de Inferência: CPU, Batch e Métricas de Desempenho

## Perguntas

1. Quem for usar o inference tem mais vantagem mandando muitas palavras em lote ou uma por vez? Em cada caso, qual o desempenho esperado?
2. Como usar o CPU corretamente para obter máxima performance — quais as vantagens e desvantagens de cada modo?
3. `words/s` é uma boa medida de desempenho? O que `chars/s` e `tokens/s` acrescentam?
4. Como a CPU Xeon com 36 cores se relaciona com o escalamento do batch? O modelo usa os 36 cores?
5. Por que GPU e CPU têm desempenhos diferentes — e isso ainda precisa ser comprovado formalmente para GPU.
6. Quais são os próximos steps de otimização e quais verificações 2×2 são necessárias para fechar as conclusões?

---

## Dados de referência — sweep de batch_size (Exp104d)

Hardware CPU: Xeon 36 processadores lógicos, Windows 10, PyTorch MKL, FP32.
Hardware GPU: RTX 3060 12GB, CUDA, FP32.
Modelo: exp104d (hidden=384, sep=S, DA loss), 17.2M params.

**Sweep formal** (CPU: 2026-03-15, adaptive, warmup=20, words=500 · GPU: 2026-03-14, warmup=20, words=1000):

| batch_size | CPU stable w/s | CPU p50 ms | GPU stable w/s | GPU p50 ms | GPU/CPU |
|-----------|---------------|-----------|---------------|-----------|---------|
| 1 | 23.8 | 41.8 ms | 34.4 | 28.4 ms | 1.45× |
| 4 | 60.3 | 16.4 ms | 95.3 | 10.3 ms | 1.58× |
| 8 | 88.2 | 11.4 ms | 160.5 | 6.1 ms | 1.82× |
| 16 | 123.6 | 8.1 ms | 264.4 | 3.7 ms | 2.14× |
| 32 | 155.2 | 6.4 ms | 406.0 | 2.4 ms | 2.62× |
| 64 | 174.2 | 5.6 ms | 585.8 | 1.7 ms | 3.36× |
| 128 | 190 | 5.2 ms | 745 | 1.35 ms | **3.92×** |
| 256 (GPU) | — | — | 976 | 1.03 ms | — |
| 512 (GPU) | — | — | **1.106** | 0.89 ms | — |

GPU speedup vs GPU batch=1: 1.0×, 2.77×, 4.67×, 7.69×, 11.8×, 17.0×, 21.7×, 28.4×, **32.2×**

---

## Pergunta 1 — Uma palavra por vez vs lote: quando usar cada modo?

### Resposta

**Regra geral:**
- Se você tem **uma palavra por vez** (TTS em tempo real, consulta interativa): use `predict(word)` — latência p50 ~42ms (CPU) / ~28ms (GPU), suficiente para real-time em quase todos os casos.
- Se você tem **N palavras prontas para processar** (pré-processamento de corpus, pipeline de transcrição, normalização de texto): use `predict_batch_native(words, batch_size=32)` — throughput 6.5× maior no CPU, 11.8× maior na GPU (vs batch=1).

**Tabela de decisão (Exp104d, sweeps formais CPU 2026-03-15 / GPU 2026-03-14):**

| Cenário | API recomendada | batch_size | CPU w/s | GPU w/s | Latência CPU | Latência GPU |
|---------|----------------|-----------|---------|---------|-------------|-------------|
| TTS palavra a palavra (interativo) | `predict(word)` | 1 | **24** | **34** | 42 ms | 28 ms |
| Pipeline de baixo volume (até ~100 palavras) | `predict_batch_native` | 16 | **124** | **264** | 8.1 ms | 3.7 ms |
| Processamento de corpus (volume alto) | `predict_batch_native` | 32 | **155** | **406** | 6.4 ms | 2.4 ms |
| Ingestão máxima sem GPU | `predict_batch_native` | 128 | **190** | — | 5.2 ms | — |
| Ingestão máxima com GPU | `predict_batch_native` | 512 | — | **1.106** | — | 0.89 ms |

**Regra de dispositivo:** GPU é superior em todos os cenários se disponível. CPU é a alternativa adequada para ambientes sem GPU ou para serving de baixo custo.

**Desvantagem do lote:**
- Latência da primeira resposta é maior (o chunk inteiro precisa ser preenchido antes de processar).
- Para TTS em streaming, batch=1 pode ser preferível mesmo com menor throughput.

### Código mínimo

```python
p = G2PPredictor.load("best_per")

# Modo unitário (TTS, consulta):
result = p.predict("computador")

# Modo lote (corpus, pipeline):
results = p.predict_batch_native(lista_de_palavras, batch_size=32)
```

---

## Pergunta 2 — Como usar o CPU corretamente para máxima performance?

### Resposta

**O que funciona no Xeon (confirmado experimentalmente):**

| Técnica | Efeito | Status |
|---------|--------|--------|
| Batch inference (`batch_size≥16`) | +5 a 8× throughput | ✅ Confirmado |
| MKL default threads | Ótimo como está | ✅ Não mexer |
| `model.eval()` único no load | Neutro/correto | ✅ Implementado |

**O que NÃO funciona no Xeon (experimentalmente falsificado):**

| Técnica | Efeito medido | Motivo |
|---------|--------------|--------|
| `quantize_dynamic` INT8 | −38% | Dequantização por passo > ganho compute |
| `set_num_threads(1)` | −61% | Elimina paralelismo MKL real para W_hh |

**Recomendação operacional — CPU Xeon:**
1. Carregar modelo com `G2PPredictor.load()` (padrões já otimizados).
2. Para throughput: agrupar palavras em lotes de 32–128 e usar `predict_batch_native()`.
3. Não aplicar quantização ou controle manual de threads sem validar A/B no hardware específico.

**Para hardware diferente (ARM, laptop sem MKL):** testar com `--quantize` e `--threads N` — os experimentos foram feitos no Xeon e os resultados podem ser invertidos em outra arquitetura.

---

## Pergunta 3 — `words/s`, `chars/s` e `tokens/s`: qual a melhor métrica?

### Análise

**`words/s` (palavras por segundo):**
- Métrica mais intuitiva para o usuário final.
- Responde diretamente: "quantas palavras esse sistema processa por segundo?"
- Mais relevante para SLA de sistemas (TTS, NLP pipeline).
- Limitação: ignora que palavras têm comprimentos diferentes — "a" e "extraordinariamente" custam tempos diferentes para o modelo.

**`chars/s` (caracteres de entrada por segundo):**
- Melhor para normalização por comprimento de entrada.
- Revela se há penalidade para palavras longas (encoder BiLSTM: O(n) em comprimento).
- Útil para comparar modelos com diferentes comprimentos médios de vocabulário.

**`tokens/s` (fonemas de saída por segundo):**
- Mais diretamente relacionado ao trabalho do decoder (cada token = um passo autoregressivo).
- Ideal para comparar custo real de compute entre modelos com diferentes complexidades de saída.
- Exemplo: modelos com separadores silábicos geram mais tokens por palavra → decoder mais custoso.

**Recomendação:**
- Para SLA de sistemas: reportar `words/s` (primário) + `p50 ms` (latência unitária).
- Para análise de modelo e comparação técnica: reportar `tokens/s` (trabalho do decoder) + `chars/s entrada` (trabalho do encoder).
- Os três juntos formam o quadro completo para publicação científica.

---

## Pergunta 4 — Relação entre Xeon 36 cores e escalamento de batch

### Status: parcialmente respondida (análise indireta)

O Xeon Scalable com 36 processadores lógicos usa MKL (Math Kernel Library) para operações BLAS dentro do PyTorch. O comportamento observado:

**Com batch=1:** MKL já usa múltiplos threads para a operação LSTM W_hh (4×384×384). Forçar `threads=1` causou regressão de 61%, confirmando que MKL está ativamente usando parallelismo intra-op.

**Com batch crescente:** O throughput escala sub-linearmente. Possíveis razões:
1. A largura da matriz aumenta (batch × hidden), dando mais trabalho por kernel MKL.
2. MKL redistribui os threads disponíveis para matrizes maiores → melhor utilização.
3. O overhead Python fixo por chamada é amortizado por mais itens.

**Saturação em batch≥64:** p50 estabiliza em ~5.5ms independente do batch. O gargalo migrou de operação matricial para:
- Loop Python de decodificação de índices (por item no batch).
- Alocação de tensores (`torch.zeros` para padding).
- Possível saturação de bandwidth de memória.

**Questão aberta:** Quantos dos 36 cores efetivamente participam por chamada? Isso requer `torch.get_num_threads()` + profiling com vtune/perf. A relação exata entre `num_cores` e throughput ótimo não foi medida diretamente.

---

## Pergunta 5 — Por que GPU e CPU têm desempenhos diferentes?

### Respondida (sweeps formais: GPU 2026-03-14 · CPU 2026-03-15)

**Resultado: GPU superior em todos os batch sizes ≥4. Em batch=1, depende do modelo.**

A hipótese inicial ("CPU vence em batch=1 por overhead de kernel CUDA") foi parcialmente falsificada. Para Exp104d (`best_per`): GPU 1.45× mais rápida em batch=1 (34.4 vs 23.8 w/s). Para Exp9 (`best_wer`, sem separadores): GPU ≈ CPU em batch=1 (40.8 vs 41.2 w/s).

**Por que a vantagem da GPU em batch=1 depende do modelo:**
- Modelos com separadores silábicos + tokens estruturais (Exp104d) geram sequências de saída mais longas → mais passos autoregressivos → MKL trabalha mais em sequência → GPU amortiza melhor.
- Modelos sem separadores (Exp9, Exp4) têm saídas mais curtas → overhead CUDA proporcionalmente maior → CPU é equivalente ou superior em batch=1.

**Por que a vantagem da GPU cresce com batch (consistente em todos os modelos):**
- A GPU escala quase linearmente até batch≈128–256; o CPU satura em batch≈64 — gargalo migra para Python overhead.

**Diferença de saturação:**

| Device | Ponto de saturação | Throughput pico (Exp104d) | Range 19 modelos | Gargalo após saturação |
|--------|-------------------|--------------------------|-----------------|-----------------------|
| CPU | batch≈64 (p50 estável em ~5.5ms) | **~190 w/s** (formal) | 190–736 w/s | Python overhead (alocação, decode loop) |
| GPU | batch≈512 (p50 ~0.89ms) | **~1.106 w/s** (formal) | 1.081–1.500 w/s | Paralelismo CUDA esgotado |

**Nota de variância (sweep noturno 2026-03-14):** O sweep formal com warmup=20, words=1000 confirmou que CV alto nos batch sizes intermediários era artefato de amostra insuficiente (~60 chunks em calibração). Com 800 chunks independentes: CV=9.5%, sem avisos. Ver `docs/benchmarks/BENCHMARK.md §Análise de variância`.

### Verificação 2×2 — completa (Exp104d, sweeps formais)

| | batch=1 | batch=32 | batch=128 | batch=512 |
|--|---------|---------|---------|---------|
| **CPU** | 23.8 w/s ✅ | 155 w/s ✅ | 190 w/s ✅ | — |
| **GPU** | 34.4 w/s ✅ | 406 w/s ✅ | 745 w/s ✅ | 1.106 w/s ✅ |

CPU: sweep formal adaptativo 2026-03-15 (warmup=20, words=500). GPU: sweep overnight 2026-03-14 (warmup=20, words=1000).

**Nota sobre variação por modelo:** a vantagem da GPU em batch=1 varia de 0.78× (CPU vence, Exp4 sem sep) a 1.45× (GPU vence, Exp104d com sep+struct). Para Exp9 (`best_wer`, sem sep): GPU/CPU ≈ 0.99×. A GPU supera o CPU consistentemente a partir de batch≥4 em todos os 19 modelos testados.

---

## Pergunta 6 — Próximos steps de otimização e verificações pendentes

### Steps implementados e status

| Step | Descrição | Resultado |
|------|-----------|-----------|
| 1 | INT8 quantization | ❌ −38% no Xeon (revertido, flag disponível) |
| 2 | Remove redundant eval() | ✅ Neutro/correto |
| 3 | Buffer reuse | Não implementado |
| 4 | num_threads=1 | ❌ −61% no Xeon (revertido, flag disponível) |
| 5 | Batch inference nativo | ✅ **+8.52× em batch=128** |

### Steps pendentes

| Step | Descrição | Hipótese | Prioridade |
|------|-----------|---------|-----------|
| 6 | ~~GPU batch sweep (2×2 com CPU)~~ | ✅ GPU superior em todos os regimes | Concluído |
| 7 | AMP (BF16) no treino | Aumenta throughput de treinamento (não afeta inferência) | Média |
| 8 | Buffer/tensor reuse no predict_batch_native | Elimina `torch.zeros` + alocação por chunk | Baixa |
| 9 | `torch.compile` | Limitado no Windows/LSTM sequencial | Baixa |
| 10 | Rodada GPU com warmup≥20, runs≥50 | IC95 formal para batch=16–64 (CV alto nesta rodada) | ✅ Concluído — sweep overnight 2026-03-14 |

### Verificações pendentes

**Verificação B — Batch × Modelo (OPCIONAL):**
Confirmar que o escalamento de batch é similar para Exp9 (sem separadores, hidden=384) e Exp107 (hidden=768). Modelos com hidden maior devem ter ponto de saturação diferente pois as matrizes são maiores.

---

## Métricas oficiais a reportar após fechamento

Para cada device × batch_size × modelo:
- `stable_wps`: throughput na janela estável (primário)
- `p50_ms`: latência mediana por item
- `p95_ms`: latência no 95º percentil (pior caso representativo)
- `ci95_wps_low / ci95_wps_high`: IC95 para inferência estatística

Condição de promoção a documentação oficial (README/ARTICLE): IC95 não sobrepostos entre condições comparadas.

## Status

Respondida (todas as perguntas centrais fechadas).
Perguntas 1, 2 e 3: respondidas com dados do CPU sweep (2026-03-14).
Pergunta 4: respondida com dados indiretos do CPU sweep (MKL multi-thread confirmado; profiling direto opcional).
Pergunta 5: respondida com GPU sweep completo (2026-03-14) — GPU superior em todos os regimes.
Pergunta 6: etapas concluídas; pendências opcionais registradas (IC95 formal GPU, Batch×Modelo).

Pode ser movido para `answered/` após IC95 formal da GPU (opcional) ou imediatamente se os dados atuais forem suficientes para o artigo.
