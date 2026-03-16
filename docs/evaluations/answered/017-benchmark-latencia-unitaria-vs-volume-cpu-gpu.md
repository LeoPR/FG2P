# 017 - Benchmark: Latencia Unitária vs Volume (CPU vs GPU)

## Pergunta

Como avaliar de forma tecnicamente justa a aceleracao CPU vs GPU no FG2P, separando:

1. latencia de palavra individual;
2. throughput em volume;
3. custo de orquestracao (Python + transferencia CPU->GPU);
4. ganhos reais que venham de paralelismo e batching.

## Resposta curta

A GPU nao deve ser avaliada apenas por palavra individual.

1. Para requisicao unitária, overhead fixo (preprocessamento Python, lancamento de kernels, sincronizacao e decodificacao) pode diluir ou anular vantagem da GPU.
2. Para volume (micro-batch e batch), a GPU tende a ganhar por paralelismo, desde que o pipeline realmente use lote e minimize overhead por item.
3. O benchmark deve medir dois regimes explicitamente: "single-item latency" e "sustained throughput"; comparar um so regime mistura sinais e leva a conclusoes erradas.

## Diagnóstico técnico consolidado

### 1) O que os resultados atuais ja mostram

Pelos benchmarks listados no estado atual (p50 tipicamente 24-33 ms em GPU e 19-30 ms em CPU):

1. Em varios modelos, CPU aparece melhor ou muito proxima em latencia unitária.
2. Isso e coerente com inferencia palavra-a-palavra: a GPU fica subutilizada e o custo fixo vira parcela grande do tempo total.
3. A comparacao atual ainda nao isola explicitamente custo de transferencia e custo de orquestracao Python por item.

### 2) Componentes de tempo que precisam ser separados

Para cada predicao, o tempo observado pode ser decomposto como:

$$
T_{total} = T_{prep\_python} + T_{h2d} + T_{kernel} + T_{sync} + T_{decode}
$$

onde:

1. $T_{prep\_python}$: tokenizacao, montagem de tensores e controle em Python;
2. $T_{h2d}$: transferencia host->device;
3. $T_{kernel}$: computo efetivo no modelo;
4. $T_{sync}$: sincronizacao implicita/explicita para coletar resultado;
5. $T_{decode}$: conversao de saida para string/fonemas.

No regime unitário, $T_{prep\_python} + T_{sync} + T_{decode}$ frequentemente domina. No regime em lote, $T_{kernel}$ cresce melhor que linear por item e melhora throughput.

### 3) Sobre "delay CPU -> GPU"

1. Existe custo de transferencia e de launch/sync por chamada.
2. Em entradas pequenas (uma palavra), esse custo pode ser maior que o beneficio do paralelismo da GPU.
3. Em lote, o custo fixo e amortizado por varios itens, aumentando a vantagem da GPU.

## Regimes de performance a medir (obrigatorio)

Para nao "diluir" sinais comuns entre CPU e GPU, medir quatro regimes:

1. R1 - Latencia unitária fria (cold-ish): palavra individual, apos aquecimento minimo.
2. R2 - Latencia unitária quente (steady): palavra individual em sequencia longa.
3. R3 - Micro-batch throughput: lotes pequenos (ex.: 4, 8, 16) para simular baixa fila.
4. R4 - Batch throughput sustentado: lotes maiores (ex.: 32, 64, 128) para carga de volume.

Cada regime deve reportar no minimo:

1. p50, p95, p99 de latencia por item;
2. throughput (words/s);
3. CV global e CV em janela estavel;
4. IC95 para latencia media e throughput.

## Otimizacoes prioritarias para maximizar desempenho

### A) Inferencia

1. Tornar batching real de inferencia primeira classe (API batch nativa no caminho principal).
2. Evitar sincronizacao desnecessaria no miolo de decodificacao (manter sincronizacao apenas onde estritamente necessario para medir).
3. Reusar buffers/tensores para reduzir alocacao por chamada.
4. Introduzir modo de benchmark por batch-size (1/4/8/16/32/64/128).

### B) CPU-only

1. Trilha de quantizacao dinamica INT8 para modulos elegiveis (LSTM/Linear), com comparacao de qualidade.
2. Definir politica de threads/processos por perfil de carga (latencia vs throughput).

### C) GPU

1. Pipeline orientado a lote para amortizar $T_{h2d}$ e overhead Python por item.
2. Avaliar mixed precision de inferencia (quando numericamente seguro).

### D) Benchmark e observabilidade

1. Manter artefato bruto por medicao (CSV sidecar ja implementado).
2. Registrar metadados de regime: batch_size, queue_depth simulada, device, dtype, warmup/runs.
3. Comparar CPU vs GPU por regime, nao por numero unico agregado.

## Plano completo em fases (execucao)

### Fase 1 - Auditoria de baseline (agora)

1. Rodar 1 experimento representativo em GPU e CPU nos regimes R1-R4.
2. Fixar protocolo: palavras, warmup, runs, batch-size e sementes.
3. Consolidar tabela de crossover (quando GPU passa CPU).

### Fase 2 - Provas de ganho (curto prazo)

1. Branch A: batching real de inferencia.
2. Branch B: quantizacao dinamica CPU.
3. Branch C: ajuste de sincronizacao/decodificacao para reduzir overhead unitário.

### Fase 3 - Politica operacional (fechamento)

1. Perfil "latencia unitária minima": priorizar CPU ou GPU conforme crossover medido.
2. Perfil "volume maximo": priorizar GPU com batch recomendado.
3. Perfil "custo/infra": CPU quantizado quando SLA permitir.

## Critérios de aceite para promover mudancas

1. Ganho robusto com IC95 favoravel no regime-alvo.
2. Sem regressao relevante de qualidade (PER/WER).
3. Variancia controlada (CV dentro de limite operacional definido por regime).
4. Reprodutibilidade em Windows e Linux.

## Comandos recomendados (praticos)

1. Selecionar um experimento:
   - `python src/benchmark_inference.py --list`
2. Rodada curta de validacao (pipeline):
   - `python src/benchmark_inference.py --index 18 --device cuda --force --words 50 --warmup 5 --runs 20`
3. Analise offline do artefato:
   - `python src/benchmark_inference.py --analyze results/benchmarks/benchmark_run_index_18_exp104d_structural_tokens_correct__YYYYMMDD_HHMMSS.json`
4. Rodada formal:
   - `python src/benchmark_inference.py --index 18 --device both --force --words 200 --warmup 20 --runs 200`

## Conclusão operacional

1. A pergunta correta nao e "GPU e mais rapida?" de forma unica, e sim "em qual regime ela acelera?".
2. Para palavra individual, CPU pode vencer por overhead fixo menor.
3. Para volume, GPU deve vencer quando batching real estiver no caminho principal.
4. O projeto deve adotar comparacao por regime e politica de execucao por objetivo (latencia, throughput ou custo).

## Atualização operacional (2026-03-13)

Objetivo desta atualização: validar apenas a logica do benchmark (sem rodada longa), usando o indice 18 como modelo.

1. Protocolo efemero validado com timeout curto por comando (smoke).
2. Execucao smoke concluida com sucesso:
   - `--index 18 --device cpu --words 3 --warmup 1 --runs 1`
   - agregado salvo em `results/benchmarks/smoke_idx18_cpu.json`
3. Integridade de artefatos confirmada:
   - `artifact_path` presente no JSON agregado
   - `raw_csv_path` presente no artefato por device
   - CSV bruto existe em disco (`Test-Path = True`)
4. Fase offline confirmada:
   - `--analyze results/benchmarks/smoke_idx18_cpu.json` funcionou
   - redirecionamento agregado -> artefato por device funcionando para resultado unico

Regra operacional acordada para proximos testes automatizados:

1. Somente testes efemeros quando executados pelo agente.
2. Nada em background para experimentos potencialmente longos sem confirmacao explicita.
3. Rodadas formais completas ficam para execucao manual no terminal do usuario apos smoke green.

## Atualização CPU sweep completo (2026-03-14)

### Resultados experimentais CPU — Exp104d, Xeon 36 cores, 200 palavras, warmup=10, runs=20–50

| Regime | batch_size | w/s | stable w/s | p50 ms | Speedup vs batch=1 |
|--------|-----------|-----|-----------|--------|--------------------|
| R1/R2 | 1 (baseline) | 21.5 | 21.6 | 45.9 ms | 1.0× |
| R3 micro | 4 | 55.4 | 55.8 | 17.9 ms | 2.58× |
| R3 micro | 8 | 83.4 | 83.1 | 11.8 ms | 3.85× |
| R3 micro | 16 | 114.6 | 115.5 | 8.68 ms | 5.35× |
| R4 sustentado | 32 | 143.2 | 142.9 | 6.85 ms | 6.63× |
| R4 sustentado | 64 | 165.0 | 162.7 | 5.52 ms | 7.67× |
| R4 máximo | 128 | 183.0 | 184.0 | 5.54 ms | **8.52×** |

Saturação confirmada em batch≥64: p50 estabiliza (~5.5ms) — gargalo migrou de operação matricial para overhead Python (alocação de tensores, loop de decodificação de índices).

### Experimentos negativos realizados (CPU)

| Técnica | Resultado | Motivo |
|---------|-----------|--------|
| INT8 quantize_dynamic | −38% (13.3 w/s) | Dequantização por passo > ganho compute em Xeon AVX-512 |
| set_num_threads(1) | −61% (8.4 w/s) | Elimina paralelismo MKL real para W_hh (4×384×384) |

Ambos revertidos. Flags `--quantize` e `--threads N` mantidos como opt-in para outros hardwares (ARM, x86 sem AVX-512).

### Tabela 2×2 completa — CPU vs GPU, Exp104d (2026-03-14)

| batch_size | CPU stable w/s | GPU stable w/s | GPU/CPU ratio | GPU speedup vs GPU=1 |
|-----------|---------------|---------------|---------------|----------------------|
| 1 | 21.6 | 29.8 | **1.38×** | 1.0× |
| 4 | 55.8 | 86.6 | **1.55×** | 2.90× |
| 8 | 83.1 | 148.8 | **1.79×** | 4.99× |
| 16 | 115.5 | 250.3 | **2.17×** | 8.39× |
| 32 | 142.9 | 312.9 | **2.19×** | 10.5× |
| 64 | 162.7 | 411.9 | **2.53×** | 13.8× |
| 128 | 184.0 | 729.6 | **3.97×** | 24.5× |

Hardware CPU: Xeon 36 cores, MKL, FP32. Hardware GPU: RTX 3060 12GB, CUDA, FP32.

**Achado principal: GPU vence em todos os batch sizes, incluindo batch=1.**

A hipótese de crossover ("CPU vence no unitário por overhead de kernel") foi falsificada: GPU já é 1.38× mais rápida em batch=1 (30 w/s vs 21.6 w/s). O número anterior (~13 w/s GPU batch=1) vinha de benchmark multi-modelo com alta contention/thermal drift — não era comparável.

### Comportamento de saturação diferenciado

- **CPU**: satura em batch≥64 — p50 estabiliza em ~5.5ms. Gargalo = Python overhead (alocação de tensores, loop de decodificação de índices).
- **GPU**: ainda escala em batch=128 (p50=1.28ms) — sem saturação clara. GPU parallelism ainda adiciona valor; gargalo ainda é compute/memória, não overhead Python.

Isso implica que a GPU tem um ponto de saturação mais alto e se beneficia mais de batches maiores.

### Notas sobre variância (CV e thermal drift)

Runs de batch=16, 32 e 64 registraram CV > 15% e/ou thermal drift negativo. Isso indica que warmup=10 (com warmup=1 chamada de batch inteiro) pode ser insuficiente para estabilizar a GPU nesses tamanhos. Os resultados são indicativos e coerentes com a tendência, mas IC95 formal exigiria rodada com warmup maior (≥20) e runs ≥50.

### Próximas otimizações GPU

| Step | Descrição | Hipótese | Prioridade |
|------|-----------|---------|-----------|
| G2 | Mixed precision FP16/BF16 para inferência GPU | Reduz tempo de kernel e largura de banda de memória | Média |
| G3 | Rodada controlada com warmup=20, runs=50 para IC95 formal | Reduzir CV e confirmar números com intervalo | Média |
| G4 | Reduzir round-trips Python→CUDA no decoder (jit.script) | 50 passos = 50 kernel launches; complexo para LSTM variável | Baixa |

## Status

Respondida.
CPU sweep completo (2026-03-14). GPU sweep completo (2026-03-14).
Tabela 2×2 fechada: GPU superior em todos os regimes. Crossover CPU-GPU não existe para este modelo.
Ver doc 018 para guia prático de uso e análise de métricas.
