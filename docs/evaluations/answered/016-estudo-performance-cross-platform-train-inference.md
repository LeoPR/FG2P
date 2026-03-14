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

### Decisão sobre o código

A implementação atual (flag `quantize` no `G2PPredictor.load()`, sem `if quantize` no loop de predição) é tecnicamente limpa: a decisão acontece uma vez no load, e o objeto `model` resultante é idêntico na interface. Não há ramificação no caminho quente.

Porém, o valor padrão `quantize=True` está errado para o hardware atual e pode causar regressão silenciosa.

**Ação a executar:** alterar padrão para `quantize=False` e manter o mecanismo desativado por padrão, com documentação clara de quando habilitá-lo (ARM, modelos maiores, após validação A/B positiva).

## Status

Respondida.
Documento de estudo técnico consolidado para orientar a próxima rodada de decisão.
Atualizado com resultado negativo do experimento de quantização dinâmica INT8 (2026-03-14).