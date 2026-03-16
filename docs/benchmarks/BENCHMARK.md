# FG2P — Benchmark de Inferência

Este documento descreve o protocolo formal de benchmark do FG2P, os resultados obtidos,
as conclusões técnicas e os comandos para reprodução.

---

## Protocolo de medição

### Hardware de referência

| Componente | Especificação |
|-----------|---------------|
| CPU | Intel Xeon (36 processadores lógicos), Windows 10, PyTorch MKL |
| GPU | NVIDIA RTX 3060 12GB, CUDA, FP32 |
| Framework | PyTorch 2.x, Python 3.13 |

### Parâmetros do benchmark

| Parâmetro | Calibração (proporcionalidade) | Sweep formal (overnight) |
|-----------|-------------------------------|--------------------------|
| `--words` | 200 | 1000 (GPU) / 500 (CPU) |
| `--warmup` | 10 | 20 |
| `--runs` | 20–50 (fixo) | `--adaptive` (min=10, max=80, cv_target=3%) |
| `--batch-size` | manual por run | `--sweep` (CPU) / `--sweep-gpu` (GPU) |
| Split | test set estratificado (seed=42, test_ratio=0.3) | idem |
| Overhead | calibrado por loop vazio (2000 iterações) | idem |

O modo `--adaptive` substitui o número fixo de runs por convergência de CV: para quando a janela deslizante de medianas por run atinge CV < 3% (padrão). Aplica rejeição de outliers por IQR de Tukey em dois níveis (run e amostra individual).

### Métricas coletadas

- `stable_wps`: throughput na janela de menor CV (20% das medições) — **métrica primária de throughput**
- `p50_ms`, `p95_ms`: latência mediana e 95º percentil por item — **métricas primárias de latência**
- `global_cv`: coeficiente de variação global (std/mean); sensível a outliers e a cauda longa de comprimento de entrada
- `robust_cv`: CV robusto = IQR/p50; resistente a outliers e à cauda de comprimento de palavra — melhor indicador de dispersão real do hardware
- `thermal_ratio`: razão última janela / primeira janela de 20% (< 0.90 indica warmup insuficiente)
- `ci95_wps_low / ci95_wps_high`: intervalo de confiança 95% via SEM

> **Interpretação de CV > 15%**: o aviso não significa necessariamente contention. Ver §"Análise de variância e contention" para distinção entre variância estrutural (comprimento de entrada), artefato de amostra pequena e contention real.

### Modo batch

Para `batch_size > 1`, a latência reportada por item é calculada como:
```
latência_por_item = tempo_do_chunk / n_itens_no_chunk
```
O encoder processa o batch inteiro em uma passagem com `pack_padded_sequence`.
O decoder executa `max_len` passos autoregressivos sobre `(batch × hidden)` em vez de `(1 × hidden)`.

---

## Resultados — Calibração (2026-03-14)

> **Nota**: estes resultados são de uma rodada de calibração com parâmetros curtos
> (warmup=10, runs=20–50). O objetivo foi verificar proporcionalidade e localizar o
> ponto de saturação antes do sweep formal. Alguns batch sizes na GPU apresentaram
> CV > 15% e thermal drift, indicando que warmup=10 pode ser insuficiente nesses tamanhos.
> Os valores indicam tendência coerente, mas IC95 formal requer o sweep completo abaixo.

### Modelo: Exp104d (`best_per`), hidden=384, sep=S, DA λ=0.2, 17.2M params

| batch_size | CPU stable w/s | CPU p50 ms | GPU stable w/s | GPU p50 ms | GPU/CPU |
|-----------|---------------|-----------|---------------|-----------|---------|
| 1 | 21.6 | 45.9 ms | 29.8 | 32.7 ms | **1.38×** |
| 4 | 55.8 | 17.9 ms | 86.6 | 11.6 ms | **1.55×** |
| 8 | 83.1 | 11.8 ms | 148.8 | 6.6 ms | **1.79×** |
| 16 | 115.5 | 8.68 ms | 250.3 ⚠️ | 3.9 ms | **2.17×** |
| 32 | 142.9 | 6.85 ms | 312.9 ⚠️ | 3.2 ms | **2.19×** |
| 64 | 162.7 | 5.52 ms | 411.9 | 2.1 ms | **2.53×** |
| 128 | 184.0 | 5.54 ms | 729.6 | 1.3 ms | **3.97×** |
| 256 (GPU only) | — | — | 862.5 | 1.2 ms | — |
| 512 (GPU only) | — | — | 900.3 | 1.1 ms | — |
| 1024 (GPU only) | — | — | 891.3 | 1.1 ms | ← saturado |

⚠️ = CV > 15% nesta rodada de calibração por amostra insuficiente (~3 chunks/run em batch≥16 com words=200); ver §"Análise de variância" para diagnóstico completo.
CPU além de batch=128: saturação confirmada em batch=64 (p50 estável em ~5.5ms).
GPU além de batch=512: saturação confirmada (~900 w/s, p50 ~1.1ms); batch=1024 não melhora.

**GPU speedup vs GPU batch=1**: 1.0×, 2.90×, 4.99×, 8.39×, 10.5×, 13.8×, 24.5×, 28.9×, **30.2×** (≈pico)

---

## Resultados — Sweep formal (CPU 2026-03-15 · GPU 2026-03-14)

> Parâmetros: `--warmup 20 --adaptive --words 500` (CPU) / `--words 1000` (GPU), 19 modelos cada. IC95 formal via SEM.

### Modelo: Exp104d (`best_per`), hidden=384, sep=S, 17.2M params

| batch_size | CPU stable w/s | CPU p50 ms | GPU stable w/s | GPU p50 ms | GPU/CPU |
|-----------|---------------|-----------|---------------|-----------|---------|
| 1 | 23.8 | 41.8 ms | 34.4 | 28.4 ms | **1.45×** |
| 4 | 60.3 | 16.4 ms | 95.3 | 10.3 ms | 1.58× |
| 8 | 88.2 | 11.4 ms | 160.5 | 6.1 ms | 1.82× |
| 16 | 123.6 | 8.1 ms | 264.4 | 3.7 ms | 2.14× |
| 32 | 155.2 | 6.4 ms | 406.0 | 2.4 ms | 2.62× |
| 64 | 174.2 | 5.6 ms | 585.8 | 1.7 ms | 3.36× |
| 128 | **189.9** | 5.2 ms | **745.3** | 1.35 ms | **3.92×** |
| 256 (GPU only) | — | — | 975.8 | 1.03 ms | — |
| 512 (GPU only) | — | — | **1.106** | 0.89 ms | — |

IC95: CPU batch=128: [190.4, 191.2] w/s · GPU batch=512: [1.123, 1.124] k-w/s

### Variação entre 19 modelos

| Device | batch | Min w/s | Mediana w/s | Max w/s |
|--------|-------|---------|------------|---------|
| CPU | 1 | 23.8 (Exp104d) | 52 | 55.3 (Exp4) |
| CPU | 128 | 190 (Exp104d) | 697 | 736 (Exp4) |
| GPU | 1 | 30.6 (Exp103) | 41 | 43.2 (Exp4) |
| GPU | 512 | 1,081 (Exp103) | 1,322 | 1,500 (Exp0_legacy) |

### GPU/CPU em batch=1 — depende da sequência de saída

| Modelo | Sep/struct | CPU w/s | GPU w/s | GPU/CPU |
|--------|------------|---------|---------|---------|
| Exp4 (panphon_fixed) | não | 55.3 | 43.2 | **0.78×** (CPU vence) |
| Exp9 (best_wer) | não | 41.2 | 40.8 | **0.99×** (empate) |
| Exp104b | sep | 34.3 | 32.7 | **0.95×** (CPU vence) |
| Exp104d (best_per) | sep+struct | 23.8 | 34.4 | **1.45×** (GPU vence) |

**Padrão:** modelos que geram sequências de saída mais longas (separadores silábicos + tokens estruturais) amplificam a vantagem da GPU mesmo em batch=1, pois o decoder autoregressivo executa mais passos — trabalho computacional que a GPU amortiza melhor que o CPU (MKL sequencial). A GPU supera o CPU em todos os modelos a partir de batch≥4.

---

### Experimentos negativos CPU (excluídos do caminho padrão)

| Técnica | Resultado | Motivo |
|---------|-----------|--------|
| `quantize_dynamic` INT8 | −38% (13.3 w/s) | Dequantização por passo > ganho compute (Xeon AVX-512 + LSTM hidden=384) |
| `set_num_threads(1)` | −61% (8.4 w/s) | Elimina paralelismo MKL real para W_hh (4×384×384 ≈ 590K params) |

Ambos disponíveis como opt-in via `--quantize` e `--threads N` para outros hardwares.

---

## Conclusões técnicas

### 1. GPU vs CPU em batch=1 — depende do modelo

A hipótese inicial de que "CPU vence no modo unitário por overhead de kernel CUDA" foi parcialmente falsificada: a resposta depende da sequência de saída do modelo.

**Sweep formal com 19 modelos — GPU/CPU ratio em batch=1:**
- Modelos **sem separadores** (Exp4, Exp9): CPU vence ou empata (0.78×–0.99×).
- Modelos com **separadores apenas** (Exp104b, Exp103): CPU ligeiramente superior (0.89×–0.95×).
- Modelos com **separadores + tokens estruturais** (Exp104d, Exp104c): GPU vence (1.45×).

**Mecanismo:** o LSTM decoder é autoregressivo — cada token extra de saída é um passo adicional de CPU. Modelos com sequências de saída mais longas fazem o MKL trabalhar mais em sequência, amplificando a vantagem da GPU (que amortiza o overhead de kernel launch sobre mais passos de cômputo paralelo).

**Conclusão correta:** para o modelo de referência `best_per` (Exp104d), GPU já é 1.45× mais rápida em batch=1. Para `best_wer` (Exp9, sem separadores), GPU ≈ CPU em batch=1.

**Por que a vantagem cresce com batch (consistente em todos os modelos):**
- GPU escala quase linearmente até batch≈128–256.
- CPU satura em batch≈64 — gargalo migra de compute BLAS para Python overhead.

### 2. Pontos de saturação diferenciados

| Device | Ponto de saturação | Throughput pico (Exp104d) | Range entre 19 modelos | Gargalo após saturação |
|--------|-------------------|--------------------------|-----------------------|----------------------|
| CPU | batch≈64 (p50 estável em ~5.5ms) | **~190 w/s** (batch=128) | 190–736 w/s | Python overhead (alocação, decode loop) |
| GPU | batch≈512 (p50 ~0.89ms) | **~1.106 w/s** (batch=512) | 1,081–1,500 w/s | Paralelismo CUDA / GDDR6 bandwidth |

### 3. Política de deployment recomendada

Valores do modelo `best_per` (Exp104d), sweep formal 2026-03-15 (CPU) / 2026-03-14 (GPU).

| Cenário | Device | batch_size | stable w/s | p50 ms |
|---------|--------|-----------|-----------|--------|
| TTS palavra a palavra (interativo) | GPU | 1 | **34** | 28 ms |
| TTS palavra a palavra (sem GPU) | CPU | 1 | **24** | 42 ms |
| Pipeline baixo volume (≤100 palavras) | GPU | 16 | **264** | 3.7 ms |
| Corpus / NLP pipeline | GPU | 32 | **406** | 2.4 ms |
| Ingestão máxima | GPU | 512 | **1.106** | 0.89 ms |
| Ingestão máxima (sem GPU) | CPU | 128 | **190** | 5.2 ms |

### 4. Métricas de reporting

- Para SLA de sistema: `stable_wps` (primário) + `p50_ms` (latência unitária).
- Para análise técnica: `tokens/s` (decoder) + `chars/s entrada` (encoder).
- Para publicação científica: os três juntos + IC95 não sobrepostos como critério de significância.

---

## Análise de variância e contention

Durante o desenvolvimento do benchmark, todos os runs de GPU — inclusive sob baixíssima carga do sistema — dispararam o aviso `CV > 15% (possível contention)`. A análise dos dados brutos do sweep noturno (50 runs × 1000 palavras, 50k medições) revelou que **não havia contention real**: as causas eram estruturais e estatísticas.

### Diagnóstico por cenário

#### batch=1 — variância de comprimento de entrada (causa estrutural)

O decoder LSTM é autoregressive: percorre mais passos do encoder e do decoder para palavras longas.
Medindo palavras de 3 a 19 caracteres no mesmo run, a distribuição de latência é naturalmente larga:

| chars | p50 latência GPU | throughput |
|-------|-----------------|-----------|
| 3 | 13.6 ms | ~73 w/s |
| 9 (mediana do dataset) | 29.3 ms | ~34 w/s |
| 17 | 53.2 ms | ~19 w/s |

**Consequência**: CV global ≈ 27%, `robust_cv` ≈ 36% — ambos altos, porque a largura é genuína.
**Sinal de ausência de contention**: apenas 0.44% das medições >2×p50, zero >5×p50.

O aviso de contention era **falso positivo**. A métrica correta para batch=1 é `p50` (reflete palavras de comprimento mediano do dataset) e `stable_wps` (janela mais homogênea do run). Para comparação de hardware, usar sempre as mesmas palavras ou reportar throughput estratificado por comprimento.

#### batch≥16, calibração curta — artefato de amostra pequena

Com `--words 200 --runs 20 --batch-size 64`: apenas 200 ÷ 64 = ~3 chunks/run × 20 runs = **~60 medições independentes**. O estimador de CV com 60 amostras tem altíssima variância — por acaso, aquele conjunto de 60 chunks tinha dispersão aparente de 47.8%.

O sweep noturno com 1000 palavras (800 chunks independentes) para o mesmo modelo e batch size: **CV=9.5%, robust_cv=7.6%** — abaixo do threshold de 15%.

**Regra prática**: para `batch_size > 1`, garantir pelo menos 100 chunks independentes:
`min_words = batch_size × runs_desejados_de_chunks ≈ batch_size × 10` por run, ou `words × runs ÷ batch_size ≥ 200`.

O script detecta e avisa quando há menos de 100 chunks, diferenciando essa causa das demais.

#### Como identificar contention real

Contention real (CPU concorrendo por CPU, processo usando GPU em paralelo) teria as seguintes assinaturas, **ausentes neste benchmark**:

| Indicador | Ausência de contention (observado) | Contention real (não observado) |
|-----------|-----------------------------------|---------------------------------|
| Medições >5×p50 | 0.000% | >0.1% esperado |
| `global_cv` >> `robust_cv` | não (ambos altos, causa comprimento) | sim (spikes puxam global_cv) |
| Distribuição bimodal | não | sim ("modo rápido" + "modo lento") |
| `thermal_ratio` crescente | não | possível (competição por TDP) |

### Implicações para reportar performance

- **`p50` e `stable_wps`** são as métricas corretas em qualquer cenário.
- **`global_cv > 15%`** em batch=1 é estrutural — esperado, não indica problema.
- **`robust_cv`** (IQR/p50) permite distinguir variância de comprimento (robust_cv alto mesmo sem spikes) de spikes raros de OS scheduler (global_cv >> robust_cv).
- Para publicação: reportar `stable_wps [ci95_low, ci95_high]` com a condição "mesmo conjunto de palavras de teste, split estratificado seed=42".

---

## Como rodar o sweep formal

### Palavras recomendadas por batch size

Com batch grande e poucas palavras, cada run tem apenas 1 chunk — estatísticas instáveis.
O script avisa quando `batch_size > words`, mas não ajusta automaticamente.

| Batch size | Mín. words para ≥10 chunks/run | Recomendado |
|-----------|-------------------------------|-------------|
| 1–16 | 160 | 200 |
| 32 | 320 | 500 |
| 64 | 640 | 1000 |
| 128 | 1280 | 2000 |
| 256 | 2560 | 3000 |
| 512 | 5120 | 6000 |

Para o sweep overnight com batch máximo de 512: use `--words 1000` como compromisso entre qualidade estatística e tempo de execução.

### Sweep formal completo — todos os modelos (overnight)

```bash
# GPU — sweep estendido (até batch=512, ponto de saturação da RTX 3060)
python src/benchmark_inference.py --device cuda --sweep-gpu --force --warmup 20 --runs 50 --words 1000

# CPU — sweep padrão (até batch=128; satura em batch=64)
python src/benchmark_inference.py --device cpu --sweep --force --warmup 20 --runs 50 --words 500
```

Saída gerada automaticamente:
- Artefato JSON por (modelo, device, batch_size): `results/benchmarks/benchmark_*_b{N}.json`
- **CSV incremental** (crash-safe): `results/benchmarks/sweep_summary_TIMESTAMP.csv` — escrito linha a linha; se o processo for interrompido, os dados já processados estão salvos
- Artefato agregado por batch_size para rastreabilidade

O CSV pode ser lido diretamente por pandas/matplotlib para gráficos de scaling.

### Sweep em um modelo — calibração rápida

```bash
# Um modelo, GPU estendido (calibração)
python src/benchmark_inference.py --index 18 --device cuda --sweep-gpu --force --warmup 10 --runs 20 --words 500

# Um modelo, CPU+GPU, sweep padrão
python src/benchmark_inference.py --index 18 --device both --sweep --force --warmup 10 --runs 20 --words 200
```

### Batch sizes customizados

```bash
# Só os extremos relevantes
python src/benchmark_inference.py --index 18 --device both --batch-sizes "1,32,128,512" --force --warmup 20 --runs 50 --words 1000
```

### Baseline unitário (batch=1)

```bash
# Execução unitária (batch=1 implícito)
python src/benchmark_inference.py --device both --force --warmup 20 --runs 200 --words 200
```

---

## Reproduzindo a calibração (2026-03-14)

```bash
# CPU sweep de calibração (runs curtos)
python src/benchmark_inference.py --index 18 --device cpu --sweep --force --words 200 --warmup 10 --runs 20

# GPU sweep de calibração (runs curtos)
python src/benchmark_inference.py --index 18 --device cuda --sweep --force --words 200 --warmup 10 --runs 20
```

---

## Arquivos e artefatos

| Tipo | Localização | Formato |
|------|-------------|---------|
| Artefato por modelo/device (batch=1) | `results/benchmarks/benchmark_{exp}_{device}.json` | JSON |
| Artefato por modelo/device/batch (sweep) | `results/benchmarks/benchmark_{exp}_{device}_b{N}.json` | JSON |
| CSV bruto (medições por palavra) | `results/benchmarks/benchmark_{exp}_{device}[_bN]_raw.csv` | CSV |
| Resumo do sweep | `results/benchmarks/sweep_summary_{timestamp}.csv` | CSV |
| Benchmark multi-modelo 2026-03-13 | `results/benchmarks/benchmark_all_models_2026-03-13.txt` | TXT |

---

## Referências

- Protocolo: `docs/evaluations/answered/017-benchmark-latencia-unitaria-vs-volume-cpu-gpu.md`
- Guia de uso (API batch): `docs/evaluations/answered/018-guia-inferencia-cpu-batch-metricas.md`
- Otimizações CPU (INT8, threads): `docs/evaluations/answered/016-estudo-performance-cross-platform-train-inference.md`
- Script de benchmark: `src/benchmark_inference.py`
- API de inferência: `src/inference_light.py` (`G2PPredictor.predict_batch_native()`)
