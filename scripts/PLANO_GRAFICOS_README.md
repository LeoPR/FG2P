# Plano de Atualizacao dos Graficos do README (foco 104d)

## Objetivo
Organizar os scripts de geracao de figuras com o menor impacto possivel, reduzindo hardcodes e melhorando a selecao automatica do que mostrar no README principal.

Foco imediato:
- atualizar apenas os graficos usados no README da raiz;
- garantir que Exp104d apareca onde for pertinente;
- evitar refatoracao agressiva neste primeiro ciclo.

## Escopo deste ciclo (simples)
Entram agora:
- scripts/generate_comparative_visualizations.py
- scripts/generate_evolution_and_stability.py
- src/extract_visualization_data.py (somente ajustes minimos de criterio de selecao)

Nao entram agora:
- mudancas profundas em compile_article.py, generate_pdf.py, cross_eval.py, training_regime_analysis.py
- redesenho completo de pipeline

## Quais figuras do README dependem desses scripts
No README principal, os graficos relevantes sao:
- results/evolution_per_wer.png
- results/baseline_comparison.png
- results/da_loss_gain.png
- results/class_distribution_top5.png

Geradores:
- generate_evolution_and_stability.py gera:
  - evolution_per_wer.png
  - da_loss_gain.png
  - (e outros auxiliares)
- generate_comparative_visualizations.py gera:
  - baseline_comparison.png
  - class_distribution_top5.png
  - top_5_models_metrics.png (apoio)

## Diagnostico atual (resumo objetivo)
1. Best PER e Best WER estao hardcoded no comparative (exp104b/exp9), sem ranking dinamico.
2. Baseline FG2P no comparative usa valor fixo (0.49), enquanto README referencia 104d (0.48).
3. Timeline de evolucao ainda fecha em 104b e texto fixo 0.59 -> 0.49.
4. Extrator carrega metricas por experimento sem politica explicita para multiplos runs (sobrescrita por ordem).

## Plano de implementacao (incremental)

### Fase 1 - Criterios dinamicos minimos (sem quebrar layout)
Objetivo: manter os graficos quase iguais visualmente, mas trocar hardcodes por selecao automatica.

Acoes:
- comparative:
  - descobrir best_per e best_wer por ranking real dos dados carregados;
  - manter filtro de experimentos "biased" ja existente, mas com regra explicita;
  - highlights da tabela/grafico passam a ser dinamicos (prefixo removido);
  - baseline FG2P passa a usar experimento de referencia selecionado (104d quando for o melhor no criterio escolhido).
- evolution:
  - caminho principal deixa de forcar 104b final;
  - ultimo ponto passa a ser o melhor candidato no grupo pertinente (PER-centered para README);
  - titulo deixa de usar numeros fixos (0.59 -> 0.49) e passa a montar texto com valores reais.

Entrega esperada:
- graficos atualizados com 104d em destaque onde fizer sentido pelos dados.

### Fase 2 - Politica de run oficial por experimento (leve)
Objetivo: reduzir ambiguidade quando ha mais de um run do mesmo experimento.

Acoes:
- no extractor, definir regra simples e explicita para consolidacao por experimento:
  - preferencia por run mais recente (ou opcao configuravel);
  - registrar no log qual run foi escolhido.

Entrega esperada:
- rankings estaveis e reproduziveis.

### Fase 3 - Comando unico de geracao (orquestracao simples)
Objetivo: reduzir confusao de "varios generate" sem mexer pesado.

Acoes:
- criar um script leve de orquestracao (ex.: scripts/generate_readme_figures.py) que apenas chama:
  1) generate_comparative_visualizations.py
  2) generate_evolution_and_stability.py
- imprimir checklist final dos arquivos esperados em results/.

Entrega esperada:
- um comando para atualizar as figuras do README.

## Sequencia de execucao recomendada (apos ajustes)
1) Gerar comparativos:
   - python scripts/generate_comparative_visualizations.py
2) Gerar evolucao/estabilidade:
   - python scripts/generate_evolution_and_stability.py
3) Validar artefatos no README:
   - conferir as imagens em results/ e abrir README

No ciclo seguinte (quando Fase 3 existir):
- python scripts/generate_readme_figures.py

## Criterios de aceitacao deste ciclo
- 104d aparece como referencia de PER no que for PER-centered.
- badges "best" deixam de depender de nomes fixos.
- textos/rotulos numericos dos graficos principais nao ficam congelados em valores antigos.
- nenhum script fora do escopo precisa ser alterado neste primeiro passo.

## Mapa rapido dos scripts "generate" (para reduzir confusao)
- generate_comparative_visualizations.py
  - foco: comparacao entre modelos e distribuicao de classes
  - impacta diretamente 2 figuras do README
- generate_evolution_and_stability.py
  - foco: timeline, ganho de loss, estabilidade
  - impacta diretamente 2 figuras do README
- generate_pdf.py / compile_article.py
  - foco: consolidacao de documentacao (nao necessario para atualizar graficos do README)

## Proximo passo pratico
Executar somente Fase 1 agora (mudancas minimas e orientadas por dados), gerar as 4 figuras do README e validar visualmente o destaque do Exp104d onde pertinente.
