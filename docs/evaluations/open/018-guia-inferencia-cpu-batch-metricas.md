# 018 — Guia Prático de Inferência: CPU, Batch e Métricas de Desempenho

## Perguntas

1. Quem for usar o inference tem mais vantagem mandando muitas palavras em lote ou uma por vez? Em cada caso, qual o desempenho esperado?
2. Como usar o CPU corretamente para obter máxima performance — quais as vantagens e desvantagens de cada modo?
3. `words/s` é uma boa medida de desempenho? O que `chars/s` e `tokens/s` acrescentam?
4. Como a CPU Xeon com 36 cores se relaciona com o escalamento do batch? O modelo usa os 36 cores?
5. Por que GPU e CPU têm desempenhos diferentes — e isso ainda precisa ser comprovado formalmente para GPU.
6. Quais são os próximos steps de otimização e quais verificações 2×2 são necessárias para fechar as conclusões?

---

## Dados de referência — sweep de batch_size (CPU Xeon, Exp104d, 2026-03-14)

| batch_size | w/s | stable w/s | p50 ms | Speedup vs batch=1 |
|-----------|-----|-----------|--------|--------------------|
| 1 | 21.5 | 21.6 | 45.9 ms | 1.0× |
| 4 | 55.4 | 55.8 | 17.9 ms | 2.58× |
| 8 | 83.4 | 83.1 | 11.8 ms | 3.85× |
| 16 | 114.6 | 115.5 | 8.68 ms | 5.35× |
| 32 | 143.2 | 142.9 | 6.85 ms | 6.63× |
| 64 | 165.0 | 162.7 | 5.52 ms | 7.67× |
| 128 | 183.0 | 184.0 | 5.54 ms | **8.52×** |

Hardware: Xeon com 36 processadores lógicos, Windows 10, PyTorch MKL.
Modelo: exp104d (hidden=384, sep=S, DA loss), 17.2M params.

---

## Pergunta 1 — Uma palavra por vez vs lote: quando usar cada modo?

### Resposta

**Regra geral:**
- Se você tem **uma palavra por vez** (TTS em tempo real, consulta interativa): use `predict(word)` — latência p50 ~46ms, suficiente para real-time em quase todos os casos.
- Se você tem **N palavras prontas para processar** (pré-processamento de corpus, pipeline de transcrição, normalização de texto): use `predict_batch_native(words, batch_size=32)` — throughput 6.6× maior.

**Tabela de decisão:**

| Cenário | API recomendada | batch_size | Throughput | Latência por item |
|---------|----------------|-----------|-----------|------------------|
| TTS palavra a palavra (interativo) | `predict(word)` | 1 | 21.6 w/s | 46 ms (p50) |
| Pipeline de baixo volume (até ~100 palavras) | `predict_batch_native` | 16 | 115 w/s | 8.7 ms |
| Processamento de corpus (volume alto) | `predict_batch_native` | 32 | 143 w/s | 6.9 ms |
| Ingestão máxima (pipeline batch, sem SLA latência) | `predict_batch_native` | 128 | 184 w/s | 5.5 ms |

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

### Status: parcialmente explicada, ainda precisa de benchmark formal de GPU com batch

**Explicação conhecida (modo unitário, batch=1):**
- GPU tem overhead de kernel launch (~50–150 µs) + transferência H2D/D2H (~10–50 µs) por operação CUDA.
- Para LSTM autoregressive batch=1: cada um dos 50 passos do decoder lança kernels CUDA separados.
- O tempo de compute por passo (~0.3–1 ms para hidden=384) é menor que o overhead → CPU vence.
- Benchmarks anteriores: CPU ~21.6 w/s, GPU ~12–15 w/s em modo single-word.

**O que ainda precisa ser comprovado (GPU com batch):**
- Com batch≥32, a GPU pode inverter o resultado — matrizes maiores amortizam o overhead de kernel.
- O benchmark de GPU atual foi feito apenas com batch=1 (modo padrão).
- Hipótese: GPU pode superar CPU em throughput com batch≥32–64.
- **Pendente:** sweep de batch_size no GPU com o mesmo protocolo do CPU.

### Verificação 2×2 necessária

| | batch=1 | batch=32 | batch=128 |
|--|---------|---------|---------|
| **CPU** | 21.6 w/s ✅ | 143 w/s ✅ | 184 w/s ✅ |
| **GPU** | ~13 w/s ✅ | ? ❌ | ? ❌ |

Sem a linha de GPU com batch, não é possível afirmar qual dispositivo é superior para throughput. A conclusão atual ("CPU é mais rápida") vale apenas para batch=1.

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
| 6 | GPU batch sweep (2×2 com CPU) | GPU pode superar CPU em batch≥32 | Alta |
| 7 | AMP (BF16) no treino | Aumenta throughput de treinamento (não afeta inferência) | Média |
| 8 | Buffer/tensor reuse no predict_batch_native | Elimina `torch.zeros` + alocação por chunk | Baixa |
| 9 | `torch.compile` | Limitado no Windows/LSTM sequencial | Baixa |

### Verificações 2×2 necessárias para fechamento

**Verificação A — Batch × Device (PENDENTE — GPU):**
Rodar o mesmo sweep de batch no GPU para determinar o crossover CPU vs GPU.

**Verificação B — Batch × Modelo (OPCIONAL):**
Confirmar que o escalamento de batch é similar para Exp9 (sem separadores, hidden=384) e Exp107 (hidden=768). Modelos com hidden maior devem saturar mais cedo pois as matrizes já são maiores.

---

## Métricas oficiais a reportar após fechamento

Para cada device × batch_size × modelo:
- `stable_wps`: throughput na janela estável (primário)
- `p50_ms`: latência mediana por item
- `p95_ms`: latência no 95º percentil (pior caso representativo)
- `ci95_wps_low / ci95_wps_high`: IC95 para inferência estatística

Condição de promoção a documentação oficial (README/ARTICLE): IC95 não sobrepostos entre condições comparadas.

## Status

Aberta.
Perguntas 1, 2 e 3: respondidas com dados do sweep de batch (2026-03-14).
Perguntas 4 e 5: parcialmente respondidas — pendente benchmark GPU com batch.
Pergunta 6: em andamento.
