# 014 - Auditoria da Narrativa de Velocidade (Exp106)

Status: aberta

Objetivo:
- verificar se a narrativa atual de velocidade do Exp106 representa um ganho robusto e reproduzivel;
- distinguir resultado de ablation transitoria versus recomendacao pratica de deployment;
- registrar o que esta diferente do desenho original e o que falta para sustentar claim forte.

## Diagnostico Executivo

Leitura atual: a narrativa publica esta superenfatizando um speedup de 2,58x do Exp106.

Problema central: o proprio desenho do experimento define Exp106 como ablation de baixo impacto esperado (hifen), com prioridade baixa e recomendacao inicial de curiosidade cientifica, nao como candidato principal de latencia.

## O que esta diferente do esperado

1. O proposito original de Exp106 era isolar overhead do hifen, com ganho esperado pequeno (~0,5% a 1%), nao promover novo baseline de velocidade.
   - Evidencia: `conf/config_exp106_no_hyphen_50split.json` (campos `purpose`, `hypothesis`, `hyphen_analysis.expected_speedup`, `hyphen_analysis.recommendation`).

2. O texto em ARTICLE/EXPERIMENTS transformou esse resultado em narrativa de "descoberta critica" e recomendacao para latencia critica.
   - Evidencia: `docs/article/EXPERIMENTS.md` (secao Exp106, blocos "DESCOBERTA CRITICA" e "Recomendacao pratica").
   - Evidencia: `docs/article/ARTICLE.md` (secao de discussao 8.4 e tabela de Pareto com Exp106 como opcao de latencia).

3. Ha inconsistencia interna na propria secao de calculo de velocidade em EXPERIMENTS.
   - O bloco textual registra valores iguais de tempo/palavras para Exp105 e Exp106, ao mesmo tempo em que afirma speedup 2,58x.
   - Evidencia: `docs/article/EXPERIMENTS.md` (linhas de calculo logo apos a tabela de metricas de Exp106).

## O que falta para sustentar claim de velocidade forte

1. Baseline relevante e comparavel
   - Comparar Exp106 com media/mediana dos modelos de referencia de uso real (minimo: Exp9 e Exp104b).
   - Evitar baseline transitorio unico (Exp105) como ancora de claim de produto.

2. Protocolo de repeticao + incerteza
   - Repetir benchmark em multiplas rodadas e reportar IC 95% para throughput/latencia.
   - Exemplo minimo: n>=30 rodadas por modelo, mesmo hardware, mesmo batch, warmup e seeds controlados.

3. Criterio de significancia pratica
   - Definir regra antes do resultado (ex.: speedup >=2,0x sobre media dos modelos relevantes, com IC95% nao sobreposto).
   - Se nao cumprir criterio predefinido, classificar como observacao exploratoria.

4. Coerencia editorial entre arquivos
   - Harmonizar README, ARTICLE e EXPERIMENTS para evitar linguagem de impacto forte sem evidencias replicadas.
   - Enquanto a auditoria de speed nao fechar, usar wording prudente: "sinal preliminar" / "ablation exploratoria".

## Atualizacao do ciclo (2026-03-13)

1. Benchmark multi-model executado em CPU+GPU com metricas ampliadas (w/s, tok/s, chars/s, CI95).
   - Comando: `python scripts/benchmark_inference.py --all-models --warmup 8 --runs 40`
   - Artefato: `results/benchmarks/benchmark_all_models_2026-03-13.txt`

2. Resultado de auditoria deste run:
   - Cobertura: 19 checkpoints completos (inclui Exp104d).
   - Varios modelos emitiram alertas de CV alto e/ou thermal drift, indicando contention e necessidade de rodada controlada para fechamento definitivo.

3. Acao editorial aplicada:
   - README, ARTICLE e EXPERIMENTS rebaixaram o claim de speed do Exp106 para status preliminar/em auditoria.

## Conclusao desta auditoria

- A critica de "speedup possivelmente artificial por comparacao com experimento transitorio" e valida no estado atual do texto.
- O componente robusto de Exp106 hoje e: hifen parece fonologicamente irrelevante (impacto pequeno em PER/WER).
- O componente ainda nao robusto e: claim de ganho pratico de velocidade com forca de recomendacao de deployment.

## Proximo ciclo sugerido (sem reescrever tudo agora)

1. Reclassificar temporariamente Exp106 como ablation exploratoria de eficiencia, nao como perfil "speed-critical" principal.
2. Corrigir trechos com contradicao numerica de speed em `docs/article/EXPERIMENTS.md`.
3. Planejar benchmark dedicado de velocidade com IC95% e baseline medio de modelos relevantes (Exp9/Exp104b).
4. So apos repeticao, decidir se Exp106 volta a aparecer como recomendacao de latencia.
