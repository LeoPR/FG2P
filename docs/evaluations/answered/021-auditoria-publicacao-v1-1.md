# 021 — Auditoria de Publicação: Revisão Geral v1.1

## Escopo

Revisão crítica completa do projeto FG2P para publicação no GitHub e submissão como artigo científico.
Auditoria realizada em 2026-03-16 cobrindo README.md, ARTICLE.md, REFERENCES.bib, BENCHMARK.md,
EXPERIMENTS.md, FORMULAS.md e todos os documentos de evaluations.

---

## Correções já aplicadas nesta sessão (não precisam de ação)

| Item | Arquivo | Correção aplicada |
|------|---------|-------------------|
| Exp9 PER 0.61% → 0.58% | `README.md` linhas 213, 373 | Corrigido para 0.58% (conforme ARTICLE/EXPERIMENTS) |
| Exp9 GPU speed ~31 → ~41 w/s | `README.md` linha 373 | Corrigido com dados do sweep formal |
| Exp104d CPU speed ~22 → ~24 | `README.md` linha 372 | Corrigido com dados do sweep formal |
| CPU pico ~184 → ~190 | `README.md` e BENCHMARK.md | Corrigido com dados formais 2026-03-15 |
| Exp9 CPU speed adicionado | `README.md` linha 373 | Agora inclui CPU (~41 w/s batch=1 / ~405 w/s) |
| Footnote †† CPU calibração | `README.md` linha 376 | Removida referência a calibração; atualizado para sweep formal |
| GPU pico 900→1,081–1,500 | `README.md` e BENCHMARK.md, ARTICLE.md | Range correto dos 19 modelos formais |
| Narrativa "GPU superior em todos" | BENCHMARK.md §1, ARTICLE.md §8.5 | Corrigido para "depende do modelo em batch=1" |
| Tabela formal Exp104d (CPU+GPU) | BENCHMARK.md, 018 | Adicionada seção sweep formal CPU 2026-03-15 |

---

## Problemas em aberto — requererem ação ou decisão

### Prioridade ALTA (afetam credibilidade científica)

#### A1. Exp9: dois modelos no registry (model_index 6 e 14)

**Problema:** O sweep de benchmark registra duas entradas para `exp9_intermediate_distance_aware`
(model_index=6 e model_index=14) com throughput levemente diferente. Isso indica que há dois
checkpoints de Exp9 no diretório de modelos.

**Risco:** Se o alias `best_wer` aponta para um checkpoint mas PER=0.58% veio do outro,
o modelo servido pode ter PER diferente do documentado.

**Evidência dos sweeps:**
- model_index=6: CPU batch=1 = 41.22 w/s; GPU batch=1 = 40.81 w/s
- model_index=14: CPU batch=1 = 41.25 w/s; GPU batch=1 = 39.98 w/s

**Ação necessária:** Verificar qual model_index é o oficial `best_wer` e documentar
explicitamente em `src/inference_light.py` ou em um registry JSON. Remover checkpoint
duplicado se desnecessário.

---

#### A2. Exp9 PER: origem da discrepância 0.61% vs 0.58%

**Problema:** README mostrava 0.61% (corrigido para 0.58%), mas a causa raiz não foi identificada.
As possibilidades são:
1. Exp9 foi retreinado com novo seed/split e novo checkpoint tem PER=0.58%
2. O valor 0.61% é de uma métrica diferente (ex: PER calculado sem estratificação)
3. O README simplesmente tinha um valor antigo desatualizado

**Ação necessária:** Confirmar qual é o PER oficial de Exp9 rodando:
```bash
python src/manage_experiments.py --run 6  # ou --index best_wer
```
e comparar com o valor em `models/exp9*/metadata.json`.

---

#### A3. Exp104d vs Exp104c: justificativa de escolha não documentada

**Problema:** O README escolhe Exp104d como `best_per`, mas:
- Exp104c tem WER melhor (4.92% vs 5.33%)
- Exp104c é mais simples (CE loss sem structural correction)
- Não há tabela comparativa explícita no README ou ARTICLE.md

**Ação necessária:** Adicionar parágrafo ou tabela em README §Model Selection:
- Exp104d: melhor PER (0.48%) — preferido para TTS/publicação
- Exp104c: melhor WER (4.92%) — pode ser útil para NLP sem separadores
- Por que Exp104d como `best_per` apesar do WER pior? Documentar explicitamente.

**Ref:** `docs/evaluations/answered/015-consistencia-exp104d-vs-exp104b.md` aborda 104b vs 104d;
falta análise 104c vs 104d.

---

#### A4. Citações no ARTICLE.md não usam formato \cite{}

**Problema:** Para submissão a conferência/revista, o padrão é `\cite{key}` (LaTeX) ou
`[@key]` (Pandoc/Markdown científico). O ARTICLE.md cita em prosa:
```
"A referência mais próxima é **LatPhon** (Chary et al., 2025)"
```
Mas o REFERENCES.bib tem `@article{chary2025latphon, ...}`.

**Consequência:** Incompatível com submissão direta a jornais que usam LaTeX.
Para publicação no GitHub como repositório de pesquisa, prosa é aceitável.

**Ação necessária (se artigo científico formal):**
- Converter para `\cite{}` / `[@key]`
- Verificar keys órfãs em REFERENCES.bib vs ARTICLE.md
- Verificar entradas citadas em prosa mas ausentes do .bib (Wilson 1927, Morris 2004, Brown-Cai-DasGupta 2001 — ver plano existente em `.claude/plans/`)

**Status:** Bloqueante apenas para submissão formal; não bloqueia publicação no GitHub.

---

### Prioridade MÉDIA (qualidade de publicação)

#### M1. Métricas PER_w e WER_g não documentadas no README

**Problema:** ARTICLE.md e EXPERIMENTS.md usam `PER_w` (weighted PER) e `WER_g` como métricas
graduadas. README menciona apenas PER e WER, sem explicar PER_w.

**Exemplo:** ARTICLE.md linha ~651: "PER_w **0,27%**" — o leitor do README não entende
o que isso significa nem por que é relevante.

**Ação:** Adicionar 2-3 linhas em README §"Performance & Generalization" explicando:
> PER_w is a weighted PER that penalizes phonetically distant substitutions more than
> near misses (using PanPhon articulatory distance). While PER treats all substitutions
> equally, PER_w ≈ 0.27% for Exp9 indicates most errors are phonetically adjacent.

---

#### M2. Status e speedup de Exp106 ainda ambíguo na documentação

**Problema:** `docs/evaluations/answered/014-auditoria-velocidade-exp106.md` conclui que
o speedup de Exp106 é "preliminar" e não pode ser reivindicado como claim forte. README
agora diz "(preliminary speed measurement)" mas o ARTICLE.md ainda pode ter linguagem forte.

**Ação necessária:** Verificar ARTICLE.md §8.4 trade-off table para Exp106 e garantir
que usa linguagem como "evidência preliminar" ou "estimativa exploratória".

**Ref:** Avaliação 014 já documentada.

---

#### M3. Tabela de erros por classe (A-D) inconsistente com PER do Exp9

**Problema:** README linha 241 mostra Class A (exact match) = 94.80% para Exp9.
Isso implica ~5.2% de tokens errados. Mas PER = 0.58% é muito menor.

**Possível explicação:** A tabela de classes A-D mede **distribuição de erros** apenas
para as palavras com erro (não todos os tokens), ou usa métrica diferente de PER padrão.
Isso não está explicado claramente.

**Ação:** Adicionar nota sob a tabela: "Percentages show the distribution of token-level
predictions across correctness classes. PER (0.58%) is computed at the word level via
edit distance and cannot be derived directly from Class A%."

---

#### M4. Formato de IC95 inconsistente no README (±0.03 vs [0.46–0.51%])

**Problema:** README usa dois formatos diferentes para o mesmo intervalo:
- Linha 29: "0.48% ± 0.03"
- Linha 55: "0.48% [0.46–0.51%]"

**Ação:** Padronizar para um único formato em todo o documento.
Recomendado: "0.48% [0.46–0.51%] (95% Wilson CI)" — mais informativo que ±.

---

#### M5. `best_per` e `best_wer` não têm registry explícito

**Problema:** `G2PPredictor.load("best_per")` não tem documentação formal do que esse
alias aponta. Se novos modelos forem adicionados, o alias pode mudar silenciosamente.

**Ação:** Criar ou verificar `src/inference_light.py` — onde o alias é resolvido —
e documentar explicitamente qual experiment_id cada alias mapeia. Opcional: criar
`models/registry.json` com timestamp e versão.

---

#### M6. Avaliação 019 (Real-World Use Case) está em answered mas não integrada

**Problema:** `019-real-world-use-case-g2p-escopo-taxonomia-e-metricas.md` está em
`answered/` mas não está claro se as conclusões foram integradas no README/ARTICLE.

**Ação:** Verificar se o texto final do README §"Use Cases" reflete as decisões de 019.

---

### Prioridade BAIXA (polish de publicação)

#### B1. evaluations/README.md tem acentuação inconsistente (PT-BR)

Vários títulos e palavras no README de evaluations estão sem acento por intenção
(para evitar encoding issues?). Verificar se é intencional ou se deve ser corrigido.
Ex: "Avaliacoes" → "Avaliações", "historico" → "histórico".

**Ação:** Se intencional (encoding safety), manter como está e documentar.
Se não, aplicar acentuação completa.

---

#### B2. README: falta "Quick Decision Guide" para seleção de modelo

**Problema:** A seção §"Model Selection" é detalhada, mas um leitor novo pode se
perder. Um bloco de decisão rápida ajudaria:

```
Minimizing PER (TTS, syllable structure)?  → best_per (Exp104d, 0.48% PER)
Minimizing WER (NLP, search, lookup)?      → best_wer (Exp9, 4.96% WER)
Clean phonemes, no separators?             → best_wer (Exp9, simpler output)
Maximum throughput (batch pipeline)?       → GPU batch=512, either model
```

---

#### B3. README menciona "RTX 4090 ~4.57× more cores" sem fonte formal

**Problema:** README linha 476:
> "An RTX 4090 (~4.57× more cores) would scale this advantage proportionally"

O número 4.57× não tem citação. Pode ser calculado de specs públicas da NVIDIA
(3584 vs 16384 CUDA cores), mas não está referenciado.

**Ação:** Adicionar nota: "(NVIDIA spec: RTX 3060 3,584 vs RTX 4090 16,384 CUDA cores)"
ou remover a estimativa quantitativa.

---

#### B4. BENCHMARK.md: calibração table tem valores sem footnote atualizado

A tabela de calibração em BENCHMARK.md ainda usa valores antigos (CPU 21.6 w/s etc.)
com nota de que são "calibração, não formal". Os dados formais agora existem (seção
"Sweep formal" adicionada acima da calibração). Considerar mover ou reordenar para que
a seção de dados formais apareça primeiro.

---

## Status geral do projeto

| Dimensão | Estado | Bloqueante para publicação? |
|----------|--------|----------------------------|
| Resultados experimentais | Sólidos, IC95 formais disponíveis | Não |
| Narrativa científica | Coerente; pequenas inconsistências residuais | Não (após A1-A3) |
| Comparação SOTA (LatPhon) | Honesta e bem calibrada | Não |
| Métricas e citações | Precisam de verificação (A1, A2, A4) | Parcialmente |
| Benchmark de performance | Atualizado e formal | Não |
| Reprodutibilidade | Boa; código disponível; dataset público | Não |
| Validação perceptual (MOS/ABX) | Ausente — declarado como trabalho futuro | Não (explícito) |
| Submissão para conferência | Requer \cite{} e verificação BibTeX | Sim, para submissão formal |

**Conclusão:** O projeto está em boas condições para publicação no GitHub como repositório
de pesquisa. Para submissão formal a conferência/revista, os itens A1-A4 precisam ser
resolvidos antes.

---

## Resolução dos itens principais (2026-03-16)

| Item | Resolução |
|------|-----------|
| A1 | `best_wer` atualizado para `exp9_intermediate_distance_aware` run_id=20260310_193733 (model_index=14, checkpoint canônico). Documentado em `models/model_registry.json` com nota sobre o checkpoint mais antigo (model_index=6). |
| A2 | Checkpoint canônico do Exp9 confirmado pelo usuário: run_id=20260310_193733 (mais recente). PER=0.58%, WER=4.96%. Registry atualizado. |
| A3 | Documentado via política de aliases: Exp104d = `best_per` por ter vocabulário IPA completo (sep + ˈ) e melhor PER. Exp104c é precursor sem um dos dois tokens estruturais — útil como grupo comparativo, mas não elegível para `best_per`. `best_per` → Exp104d exclusivamente (output com sep + ˈ é a entrega completa para TTS). |
| A4 | Citações em prosa convertidas para `[@key]` (Pandoc) em ARTICLE.md. Mapeamento: Chary→chary2025latphon, Wilson/Brown→wilson1927probable/brown2001interval, Morris/Bisani→morris2004cer/bisani2008joint, Mortensen→mortensen2016panphon, Bahdanau→bahdanau2014neural, Kohavi/Arlot→kohavi1995crossvalidation/arlot2010survey, Neyman/Cochran→neyman1934two/cochran1977sampling, Bottou→bottou2010large (Bottou 2012 removido: não está no .bib), Farias→removido (não está no .bib; mantido Tan→tan2021critical), HaoChen/Mishchenko→haochenSra2019shuffling/mishchenko2020reshuffling, Barbosa→barbosa2004brazilian, ByT5/Xue→byt5g2p, Reddi→reddi2020mlperf. Birkholz 2024 marcado como "em preparação" (não publicado). |
| M1 | PER_w documentado no README §DA Loss Effect, com definição, valores para Exp1/Exp9, e link para ARTICLE.md §5.3. |

## Status desta avaliação

Fechada — todos os itens de alta prioridade (A1–A4) e M1 resolvidos. Itens M2–M6 permanecem como polish opcional.
