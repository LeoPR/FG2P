# 016 — Estudo de Performance Cross-Platform (Train + Inference)

## Pergunta

A modelagem atual de processamento (GPU/CPU, paralelismo, dataloaders, dtype e inferência) está no melhor desenho para rodar em Python 3.13+ com PyTorch em Windows 10+ e Linux Ubuntu 25.10+, visando máximo aproveitamento da RTX 3060 12GB e de CPU Xeon com 36 processadores lógicos?

## Resposta curta

Parcialmente.

- O pipeline atual já aplica boas práticas relevantes (TF32, cuDNN benchmark, pin memory, inference_mode, set_to_none no otimizador).
- Há espaço relevante de melhoria para throughput, principalmente em dois pontos: AMP no treino e batch real na inferência.
- A principal limitação cross-platform é que o comportamento de paralelismo muda entre Windows (spawn) e Linux (fork), afetando DataLoader e multiprocessamento.

## Escopo e método

- Escopo: estudo técnico (sem alterações de código).
- Base: auditoria dos núcleos de treino e inferência no estado atual do repositório.
- Critério: desempenho prático para uso geral (latência e throughput), robustez estatística e coerência entre Windows/Linux.

## Diagnóstico técnico consolidado

### 1) O que já está forte no desenho atual

1. Treino com otimizações CUDA relevantes já habilitadas (TF32 e cuDNN benchmark).
2. Avaliação com `torch.inference_mode()`, reduzindo overhead de autograd.
3. DataLoader com `pin_memory=True` em CUDA, coerente para transferência CPU→GPU.
4. Uso de `zero_grad(set_to_none=True)`, reduzindo custo de memória/alocação.
5. Pipeline geral permanece dentro do ecossistema PyTorch nas etapas críticas.

### 2) Gargalos principais identificados

1. AMP ainda não está no loop de treino.
   - Impacto esperado alto na RTX 3060 (especialmente BF16/FP16).
2. Inferência em lote não está realmente batelada no caminho principal de uso geral.
   - O fluxo atual tende a chamar predição palavra a palavra, reduzindo uso efetivo da GPU.
3. CPU inference ainda sem trilha formal de quantização dinâmica (INT8) como modo de produção.
   - Em Xeon, isso tende a deixar desempenho em CPU abaixo do potencial.

### 3) Vazamento de processamento para fora do fluxo Torch

Conclusão do estudo:

1. Não há evidência de vazamento grave no caminho de tensorização central (forward/loss).
2. O maior custo extra observado não é "sair do Torch" por si, e sim o controle Python no regime de inferência unitária repetida.
3. Para throughput alto, o problema dominante é granularidade de execução (batch pequeno), não corretude do grafo.

## Cross-platform: Windows vs Linux

### Diferenças operacionais relevantes

1. Windows usa `spawn` para multiprocessamento.
   - Maior overhead na criação de workers/processos.
2. Linux usa `fork`.
   - Menor overhead, melhor para pipelines com multiprocessamento e dataloading paralelo.
3. `torch.compile` com backend inductor/Triton é mais favorável em Linux.
   - Em Windows, o ganho é mais restrito no cenário atual.

### Implicação prática no projeto

1. A decisão atual de `num_workers=0` é defensável para o desenho existente e para Windows.
2. Em Linux, vale estudo controlado com workers persistentes para confirmar ganho líquido no regime de treino real.
3. Para CPU serving com alto paralelismo, Linux tende a escalar melhor por semântica de processo.

## Dtype, precisão e quantização

### Treino em GPU (RTX 3060)

1. Prioridade técnica: AMP em BF16 (ou FP16 com scaler), com validação de estabilidade das losses customizadas.
2. TF32 já ativo é positivo, mas não substitui AMP.
3. Ganho esperado: aumento de throughput e potencial de batch maior sob o mesmo limite de VRAM.

### Inferência em GPU

1. Para baixa latência unitária, o estado atual é funcional.
2. Para throughput (uso geral em lote), falta caminho batelado robusto como primeira classe.
3. Mixed precision de inferência é candidata de otimização, desde que validada numericamente.

### Inferência em CPU (Xeon 36)

1. Quantização dinâmica INT8 para módulos elegíveis (LSTM/Linear) é prioridade de estudo aplicado.
2. Potencial de ganho em throughput/custo por requisição é alto no perfil CPU-only.
3. Paralelismo por processo tende a ser mais previsível que por thread Python para carga concorrente.

## Plano padronizado (próxima discussão)

### Fase A — Medição controlada (sem mudança estrutural)

1. Consolidar baseline de treino e inferência por dispositivo (CPU/GPU), com repetição e IC95.
2. Separar métricas de latência unitária e throughput em lote.
3. Fixar protocolo por SO para comparabilidade (Windows vs Linux).

### Fase B — Provas de ganho em branches curtos

1. Treino com AMP (BF16/FP16) e validação de estabilidade.
2. Inferência batelada real para throughput.
3. Inferência CPU com quantização dinâmica INT8.

### Fase C — Política operacional

1. Definir perfil recomendado por cenário:
   - qualidade máxima,
   - latência unitária,
   - throughput em lote,
   - custo em CPU-only.
2. Publicar matriz de decisão com critérios objetivos (ganho mínimo, estabilidade, variância).

## Critérios de aceite para promover mudança

1. Ganho estatisticamente robusto (IC95 não sobreposto no indicador alvo).
2. Sem regressão relevante em qualidade fonética (PER/WER).
3. Sem regressão de robustez cross-platform.
4. Reprodutibilidade em pelo menos um run Linux e um run Windows.

## Conclusão operacional

1. O estado atual é tecnicamente sólido como baseline.
2. O maior ganho provável de curto prazo está em AMP no treino e batching real na inferência.
3. Para CPU Xeon, quantização dinâmica + paralelismo por processo devem entrar como trilha prioritária de validação.
4. A recomendação final de configuração por ambiente deve ser fechada após rodada controlada de benchmark comparativo por SO/dispositivo.

## Atualização: Experimento de quantização dinâmica INT8 (2026-03-14)

### Protocolo executado

Benchmark A/B com `exp104d` (Exp18), CPU, 200 palavras, warmup=5, runs=10:

| Condição | w/s global | stable w/s | p50 ms |
|----------|-----------|-----------|--------|
| Baseline FP32 | 21.6 | 21.4 | 45.6 ms |
| INT8 quantize_dynamic | 13.3 | 13.3 | 76.7 ms |

Resultado: **quantização tornou o modelo 38% mais lento** (inversão total da hipótese).

### Diagnóstico técnico

1. `torch.quantization.quantize_dynamic` é "dinâmica" por design: os pesos são armazenados em INT8 mas **dequantizados para FP32 a cada forward pass**, imediatamente antes do cálculo matricial.
2. Para cada predição de palavra, o decoder executa até 50 passos autoregressivos. Em cada passo, o custo de dequantização (INT8 → FP32, por camada LSTM + Linear) é pago integralmente.
3. O modelo tem hidden_dim=384 — as matrizes de peso do LSTM são ~384×(2×384+128) ≈ 350K elementos por camada, por direção. A operação FP32 sobre esse tamanho é O(1 µs) em Xeon AVX-512, enquanto o overhead de dequantização é da mesma ordem ou maior.
4. Conclusão: o ganho computacional de INT8 (menos operações de ponto flutuante) não compensa o overhead de empacotar/desempacotar pesos a cada chamada, para modelos neste tamanho.

### Análise de portabilidade (quando INT8 dynamic poderia ajudar)

| Hardware/SO | Expectativa | Razão |
|-------------|-------------|-------|
| Xeon + Windows (ambiente atual) | ❌ Mais lento | AVX-512 FP32 já é rápido; dequantização domina para modelos pequenos |
| Intel/AMD x86 sem AVX-512 (gen antiga) | ⚠️ Neutro/marginal | FP32 mais lento → ganho INT8 maior, mas overhead ainda relevante |
| ARM (Apple Silicon, mobile) | ✅ Possível ganho | NEON INT8 nativo; dequantização mais barata; sem AVX-512 concorrente |
| GPU CUDA | ❌ Não aplicável | `quantize_dynamic` não tem suporte CUDA; modelo fica em CPU mesmo |
| Modelos maiores (hidden ≥ 1024, transformers) | ✅ Ganho confirmado na literatura | Compute domina; overhead de dequantização é percentagem menor do total |

### Por que a implementação não é trivialmente corrigível

- `quantize_static` (alternativa sem dequantização em runtime) exige passagem de calibração com dados reais e não funciona bem com a dependência temporal do LSTM autoregressivo (estado oculto sequencial).
- `torch.ao.quantization` com backends FBGEMM/QNNPACK não adiciona VNNI path automaticamente — mesmo com Xeon Scalable que tem VNNI, PyTorch 2.x não usa esse caminho para LSTM quantizado por padrão.
- O loop autoregressivo em Python (50 passos por palavra) amplia o overhead: cada passo tem custo de dequantização, totalizando 50× o overhead por palavra.

### Experimento de controle de threads intra-op (2026-03-14)

Hipótese: para inferência unitária sequencial (batch=1), `torch.set_num_threads(1)` elimina overhead de sincronização e melhora latência.

| Condição | w/s | p50 ms |
|----------|-----|--------|
| Baseline sistema (MKL default) | 21.6 | 45.6 ms |
| `set_num_threads(1)` | 8.4 | 117.6 ms |

Resultado: **61% mais lento.** O Xeon com MKL otimizado beneficia multi-threading mesmo para LSTM hidden=384. Matrizes W_hh (4×384×384 ≈ 590K params) são suficientemente grandes para MKL distribuir com ganho real. Forçar 1 thread elimina esse paralelismo.

Conclusão análoga à quantização: o sistema padrão (MKL auto-tuning em Xeon) já é ótimo para este hardware. Configurações manuais que beneficiam hardware menor (laptop sem MKL, ARM) causam regressão no Xeon.

### Padrão adotado após ambos os experimentos

| Flag | Padrão | Motivo |
|------|--------|--------|
| `quantize=False` | desativado | INT8 piora 38% em Xeon+LSTM hidden=384 |
| `num_threads=None` | sistema/MKL | threads=1 piora 61% em Xeon |

Ambos os flags permanecem disponíveis como opt-in explícito (`--quantize`, `--threads N`) para desenvolvedores testarem em seu hardware (ARM, x86 sem AVX-512, modelos maiores). A decisão é feita uma única vez no `load()`, sem ramificação no caminho quente de predição.

### Experimento de batch inference nativo (2026-03-14) — POSITIVO

Hipótese: agrupar N palavras em uma chamada ao modelo reduz overhead Python por item e aumenta utilização de MKL via matrizes maiores no batch dimension.

Protocolo: `exp104d` (Exp18), CPU Xeon, 200 palavras, sweep de batch_size.

| batch_size | w/s global | stable w/s | p50 ms | Speedup | Ganho da dobrada |
|-----------|-----------|-----------|--------|---------|-----------------|
| 1 (baseline) | 21.5 | 21.6 | 45.9 | 1.0× | — |
| 4 | 55.4 | 55.8 | 17.9 | 2.58× | 2.58× |
| 8 | 83.4 | 83.1 | 11.8 | 3.85× | 1.50× |
| 16 | 114.6 | 115.5 | 8.68 | 5.35× | 1.37× |
| 32 | 143.2 | 142.9 | 6.85 | 6.63× | 1.24× |
| 64 | 165.0 | 162.7 | 5.52 | 7.67× | 1.16× |
| 128 | 183.0 | 184.0 | 5.54 | **8.52×** | 1.11× |

Resultado: **ganho real e significativo, escalamento sub-linear com saturação confirmada em batch≥64.**

Mecanismo confirmado: o encoder (`pack_padded_sequence(enforce_sorted=False)`) processa o batch inteiro em uma passagem; cada passo do decoder roda matmul sobre `(batch, hidden)` em vez de `(1, hidden)`, melhorando utilização de MKL. O decoder autoregressivo continua sequencial (até `max_len` passos), limitando o escalamento.

Saturação observada em p50: batch=64 → 5.52ms, batch=128 → 5.54ms (idêntico). O gargalo a partir de batch≥64 migrou de operação matricial para overhead Python (alocação de tensores, loop de decodificação de índices).

Ponto de operação recomendado: **batch=32** para throughput com latência controlada; batch=128 para throughput máximo sem restrição de latência.

### Decisão sobre o código

Padrões ajustados:
- `quantize=False`, `num_threads=None` — experimentos negativos revertidos
- `predict_batch_native(words, batch_size=32)` adicionado como API de throughput
- `--batch-size N` adicionado ao benchmark para sweep controlado
- Nenhuma ramificação no hot path — todas as decisões ocorrem no `load()` ou no agrupamento de entrada

## Status

Respondida.
Atualizado em 2026-03-14 com resultados completos de três experimentos de otimização CPU:
- INT8 quantization: −38% (falsificada para Xeon+LSTM hidden=384)
- threads=1: −61% (falsificada — MKL multi-thread já é ótimo)
- Batch inference nativo: +8.52× em batch=128 (confirmada, implementada)
Ver doc 018 para guia prático de uso e análise de métricas.