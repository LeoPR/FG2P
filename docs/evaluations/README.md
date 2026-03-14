# Avaliacoes do Projeto

Esta pasta organiza avaliacoes tecnicas do FG2P em formato incremental.

Objetivo:
- registrar perguntas de avaliacao que surgem durante a pesquisa;
- separar o que ja foi respondido do que ainda precisa de evidencias ou calculos;
- manter um historico de raciocinio que possa depois ser consolidado no artigo ou removido se ficar obsoleto.

Convencao:
- `answered/`: perguntas com resposta documentada, ainda que sujeita a refinamento;
- `open/`: perguntas, riscos, hipoteses e pendencias que merecem investigacao adicional.

## Indice de Perguntas

| ID | Tema | Status | Observacao |
|----|------|--------|------------|
| 001 | Robustez da pesquisa: pontos fortes e fracos | Respondida parcialmente | Sintese inicial ja consolidada; pode ser refinada depois |
| 002 | Intervalo de confianca do WER | Respondida | Implementado em analyze_errors.py; formulas centralizadas em docs/article/FORMULAS.md |
| 003 | Tamanho minimo de amostra para teste e treino | Respondida | Fechada como criterio metodologico pratico; minimo universal fica como trabalho futuro |
| 004 | Validade externa, comparacao classica e generalizacao | Respondida | Formula final: alta para comparacao no universo `ipa-dict`, moderada para generalizacao ampla |
| 005 | Estratificacao incompleta e fatores ocultos | Respondida | Fator de seguranca mantido como principio prudencial, sem valor universal fixado |
| 006 | Curva de estabilidade com menos dados | Respondida | Evidencia descritiva consolidada com Exp104b/105/106; pontos intermediarios ficam para estudo futuro |
| 007 | Por que LatPhon nao reporta WER | Respondida | Paper usa PER como metrica principal e estatistica no nivel de fonema |
| 008 | DA Loss: forca, limites e narrativa justa | Respondida | Narrativa final separa ganho da DA Loss em comparacoes limpas do efeito combinado com override estrutural |
| 009 | Regras para comparacoes empiricas justas | Respondida | Checklist aplicado nas comparacoes principais de README e ARTICLE |
| 010 | Matriz consolidada de reivindicacoes | Respondida | Claims frageis principais foram matizados e a delimitacao formal de originalidade foi registrada |
| 011 | Questionario de fechamento por etapas | Respondida | Decisoes editoriais principais aplicadas no README/ARTICLE e no indice de avaliacoes |
| 012 | Clareza narrativa e nomenclatura | Respondida | Regra geral para ordem da historia do projeto e uso consistente de rotulos (comparacao externa vs interna) |
| 013 | Checklist final de submissao (Ciclo 3.2) | Respondida | Coerencia final validada; consistencia numerica e referencias quebradas ajustadas |
| 014 | Auditoria da narrativa de velocidade (Exp106) | Aberta | Revisar claim de speedup 2,58x, baseline comparavel e CI95 de throughput/latencia antes de recomendacao forte |
| 015 | Consistencia e ganho real: Exp104d vs Exp104b | Respondida | Exp104d melhora estimativas pontuais de qualidade, mas nao domina em todos os eixos (IC95 de WER sobreposto e custo maior em CPU) |
| 016 | Estudo de performance cross-platform (train + inference) | Respondida | Auditoria tecnica sem alteracao de codigo; prioriza AMP, batching real de inferencia e trilha de quantizacao CPU |
| 017 | Benchmark por regime (latencia unitária vs volume CPU/GPU) | Respondida | Define decomposicao de overhead (Python/H2D/sync/decode), protocolo R1-R4 e plano de otimizacao para aceleracao real |
| 018 | Guia prático de inferência: CPU, batch e métricas de desempenho | Aberta | Sweep batch=1–128 confirmado (+8.52× em batch=128 vs unitário); GPU com batch pendente; análise de métricas w/s vs chars/s vs tokens/s |

## Proximas acoes (estado atual)

1. Estado global: pronto para submissao, com ressalva de auditoria textual da narrativa de velocidade do Exp106.
2. Manter MOS/ABX como trabalho futuro explicito (nao bloqueante).
3. Tratar claim de speedup do Exp106 como preliminar ate fechamento de benchmark replicado com IC95 (run inicial em `results/benchmarks/benchmark_all_models_2026-03-13.txt`).
4. Registrar Exp104d como candidato de melhor qualidade media vs Exp104b, sem promover como "melhor em todos os sentidos" ate replicacao com menor ruido e politica de custo/latencia definida.

## Publicacao: bloqueadores e pendencias

Leitura pragmatica para submissao/publicacao:

### Bloqueadores reais de publicacao

1. Nenhum bloqueador experimental aberto no estado documental atual.

Observacao: existe uma pendencia editorial de consistencia na narrativa de velocidade do Exp106 (nota 014), tratada como ajuste de rigor e nao como bloqueador de arquitetura/metodologia.

Observacao: a delimitacao formal de originalidade da DA Loss foi registrada em `docs/article/ORIGINALITY_ANALYSIS.md` e consolidada na nota 010.

### Pendencias importantes, mas que podem virar limitacao explicita

1. **Validacao perceptual (MOS/ABX)**
	- nao precisa bloquear o texto se ficar explicitamente como trabalho futuro e se as afirmacoes auditivas forem suavizadas

### O que da para fechar agora, sem novo experimento

1. Registrar MOS/ABX e curva de estabilidade intermediaria como trabalho futuro, nao como requisito imediato.
2. Fazer revisao cruzada curta de wording de limitacoes entre README, ARTICLE e evaluations.
3. Alinhar wording de speed do Exp106 (README/ARTICLE/EXPERIMENTS) com o status de evidencia atual (nota 014).

### O que exige trabalho adicional antes de chamar de "pronto para publicar"

1. Nenhum item bloqueante adicional no estado atual.

## Plano em ciclos ate concluir tudo

### Ciclo 1 - concluido

1. Revisao editorial de claims no README e ARTICLE.
2. Separacao explicita entre efeito da DA Loss e efeito do override estrutural.
3. Fechamento formal de 008, 009 e 010.

### Ciclo 2 - fechamento sem novo experimento (concluido)

1. Fechar 004 com redacao final de validade externa:
	- alta para comparacao no universo `ipa-dict`;
	- moderada para generalizacao ampla.
2. Fechar 003, 005 e 006 como bloco de limitacoes metodologicas + futuro trabalho.
3. Fechar 011 registrando as decisoes editoriais finais ja aplicadas.

### Ciclo 3 - pronto para publicar

1. Revisao curta de originalidade da DA Loss com fontes primarias no related work. (concluido)
2. Checklist final de submissao (coerencia de claims, limitacoes, reproducibilidade, referencias). (concluido; nota 013)
3. Marcar status global como "pronto para submissao". (concluido)

## Uso esperado

Quando surgir uma nova pergunta, adicionar primeiro uma nota em `open/` com:
- problema;
- por que importa;
- evidencias ja conhecidas;
- o que falta para responder com mais rigor.

Quando a pergunta estiver madura, mover o conteudo para `answered/` ou consolidar no artigo.