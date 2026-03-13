# 015 — Consistencia e ganho real: Exp104d vs Exp104b

## Pergunta

A proposta de melhoria do Exp104d aconteceu de forma robusta quando comparada ao baseline correto (Exp104b)?

## Resposta curta

Parcialmente sim.

- Exp104d melhora as metricas pontuais de qualidade (PER/WER/Accuracy) frente ao Exp104b.
- Nao ha dominancia total: o ganho de WER ainda nao fecha como evidencia estatistica forte a 95% neste ciclo (IC95 sobreposto e teste aproximado com p > 0.05).
- Em custo computacional (especialmente CPU), Exp104d tende a ser pior por aumento de capacidade.

## Baseline e protocolo

- Baseline valido: Exp104b (`exp104b_intermediate_sep_da_custom_dist_fixed__20260311_022457`).
- Candidato: Exp104d (`exp104d_structural_tokens_correct__20260312_142940`).
- Mesmo tamanho de teste: `N = 28.782` palavras.
- Analise executada via `src/analyze_errors.py` (formato com Wilson 95%).

## Evidencias quantitativas

### Metricas classicas

- Exp104b:
  - PER = 0.51% [0.48%, 0.53%]
  - WER = 5.62% [5.36%, 5.89%]
  - Accuracy = 94.38%
- Exp104d:
  - PER = 0.48% [0.46%, 0.51%]
  - WER = 5.33% [5.07%, 5.59%]
  - Accuracy = 94.67%

### Metricas graduadas

- WER graduado: 0.58% (104b) -> 0.56% (104d)
- PER ponderado: 0.38% (104b) -> 0.38% (104d)

### Delta principal (104d - 104b)

- PER: -0.03 p.p.
- WER: -0.29 p.p.
- Accuracy: +0.29 p.p.

### Checagem estatistica rapida (WER)

- Aproximacao por diferenca de proporcoes:
  - delta = -0.2918 p.p.
  - z = -1.539
  - p (bicaudal) ~= 0.1237

Leitura: direcao favoravel ao Exp104d, mas sem fechamento forte em p < 0.05 neste ciclo.

## Consistencia em palavras longas / composicao continua

Teste qualitativo com palavras longas, hifenizadas e nomes compostos (simulando fala continua em token unico):

- Exp0 (`index 16`) mostrou instabilidade estrutural (loops e degradacao forte em sequencias longas).
- Exp104b (`index 15`) e Exp104d (`index 18`) mantiveram saida estruturalmente estavel, sem colapso semelhante ao Exp0.
- Entre 104b e 104d, as diferencas observadas foram locais (realizacoes foneticas pontuais), sem evidencia de perda de estabilidade global em longas entradas.

## Custo e inferencia

Benchmark consolidado (`results/benchmarks/benchmark_all_models_2026-03-13.txt`):

- Exp104b (`index:15`) e Exp104d (`index:18`) possuem throughput GPU similar no regime observado.
- Em CPU, Exp104d apareceu mais lento, consistente com aumento de parametros (~17.2M vs ~9.7M).
- O benchmark foi executado sob ruido de sistema (avisos de CV/thermal), portanto os numeros de velocidade devem ser lidos como indicativos.

## Conclusao operacional

1. Exp104d pode ser tratado como candidato de melhor qualidade media sobre Exp104b.
2. Nao tratar Exp104d como melhor "em todos os sentidos" neste momento.
3. Para promocao total, replicar comparacao com controle de ruido (CPU/GPU) e criterio explicito de trade-off qualidade vs custo.

## Arquivos de evidencia

- `results/exp104b_intermediate_sep_da_custom_dist_fixed/error_analysis_exp104b_intermediate_sep_da_custom_dist_fixed__20260311_022457.txt`
- `results/exp104d_structural_tokens_correct/error_analysis_exp104d_structural_tokens_correct__20260312_142940.txt`
- `results/exp104b_intermediate_sep_da_custom_dist_fixed/evaluation_exp104b_intermediate_sep_da_custom_dist_fixed__20260311_022457.txt`
- `results/exp104d_structural_tokens_correct/evaluation_exp104d_structural_tokens_correct__20260312_142940.txt`
- `results/benchmarks/benchmark_all_models_2026-03-13.txt`
